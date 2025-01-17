import time
import httpx
import json
from pathlib import Path
from queue import Queue
from threading import Thread
from datetime import datetime
import uuid
import sys
import subprocess
import socket
import os
import platform
from typing import Dict, Any

SERVER_URL = None
LOCAL_ROOT = None

# Global variables to store run configuration
RUN_NAME = None
LOCAL_BACKUP_DIR = None

# Create queues for thread-safe logging
LOCAL_QUEUE = Queue()
SERVER_QUEUE = Queue()

def _get_valid_name(name: str | None) -> str:
    """Get a valid run name, checking both local and server storage.
    
    Args:
        name: Desired name or None for anonymous
    
    Returns:
        str: Valid run name
        
    Raises:
        RuntimeError: If no valid name can be found
    """
    # Generate anonymous name if none provided
    if name is None:
        base_name = f"anonymous/{uuid.uuid4().hex[:12]}"
    else:
        base_name = name
    
    # Check server first if enabled
    server_name = base_name
    if SERVER_URL:
        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{SERVER_URL}/check_name",
                    json={"run_name": base_name}
                )
                response.raise_for_status()
                result = response.json()
                if result["status"] == "error":
                    raise RuntimeError(result["message"])
                server_name = result["name"]
        except Exception as e:
            raise RuntimeError(f"Error checking name with server: {e}")
    
    # If local logging enabled, check local name availability
    if LOCAL_ROOT:
        local_path = LOCAL_ROOT / Path(server_name)
        if local_path.exists():
            # Try suffixes locally
            for i in range(100):
                suffixed_name = f"{server_name}-{i:02d}"
                if not (LOCAL_ROOT / Path(suffixed_name)).exists():
                    # If we found a good local name and server is enabled,
                    # verify it with server
                    if SERVER_URL:
                        try:
                            with httpx.Client() as client:
                                response = client.post(
                                    f"{SERVER_URL}/check_name",
                                    json={"run_name": suffixed_name}
                                )
                                response.raise_for_status()
                                result = response.json()
                                if result["status"] == "error":
                                    continue  # Try next suffix
                                return result["name"]
                        except Exception as e:
                            raise RuntimeError(f"Error checking name with server: {e}")
                    else:
                        return suffixed_name
            raise RuntimeError(f"Could not find available name for {server_name}")
    
    return server_name

def _get_git_info() -> Dict[str, str]:
    """Get git repository information if available."""
    try:
        # Get the git root directory
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        
        # Get the current commit hash
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=git_root,
            text=True
        ).strip()
        
        # Check if working directory is clean
        is_clean = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=git_root,
            text=True
        ).strip() == ""
        
        return {
            "commit": commit_hash,
            "clean": is_clean,
            "root": git_root
        }
    except subprocess.SubprocessError:
        return {}

def _get_gpu_info() -> Dict[str, Any]:
    """Get GPU information if available."""
    try:
        nvidia_smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", 
             "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        gpus = []
        for line in nvidia_smi.split('\n'):
            name, total, free = line.split(', ')
            gpus.append({
                "name": name,
                "memory_total_mb": float(total),
                "memory_free_mb": float(free)
            })
        return {"gpus": gpus}
    except (subprocess.SubprocessError, FileNotFoundError):
        return {}

def init(name=None, info=None, server_url="http://log-cabin:8000", local_root=os.path.expanduser("~/.logs")):
    """Initialize logging for a new run.
    
    Args:
        name (str|None): Name of the run, can include path separators. If None, generates anonymous name
        info (dict|None): Optional custom info to save with the run
        server_url (str|None|False): URL of logging server, or None/False to disable
        local_root (str|Path|None|False): Root directory for local logs, or None/False to disable
    
    Returns:
        str: The actual run name that was used (may differ from requested name)
        
    Raises:
        RuntimeError: If no valid name could be found or other initialization error
    """
    global RUN_NAME, LOCAL_BACKUP_DIR, SERVER_URL, LOCAL_ROOT
    
    SERVER_URL = server_url if server_url else None
    LOCAL_ROOT = Path(local_root) if local_root else None
    
    # Get valid name checking both local and server storage
    actual_name = _get_valid_name(name)
    RUN_NAME = actual_name
    # Gather system information
    with open("/proc/self/cmdline", "rb") as f:
        cmd_bytes = f.read()
    cmd_line = cmd_bytes.replace(b"\0", b" ").decode()
    # Add CUDA_VISIBLE_DEVICES to cmdline if set
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_devices is not None:
        cmd_line = f"CUDA_VISIBLE_DEVICES={cuda_devices} {cmd_line}"
    
    system_info = {
        "time": datetime.now().isoformat(),
        "command": {
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "executable": sys.executable,
            "cmdline": cmd_line
        },
        "system": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "CUDA_VISIBLE_DEVICES": cuda_devices
        }
    }
    
    # Add git information if available
    git_info = _get_git_info()
    if git_info:
        system_info["git"] = git_info
    
    # Add GPU information if available
    gpu_info = _get_gpu_info()
    if gpu_info:
        system_info["system"].update(gpu_info)
    
    # Combine with user-provided info (user info takes precedence)
    combined_info = system_info | (info or {})
    
    # Set up local logging if enabled
    if LOCAL_ROOT:
        LOCAL_BACKUP_DIR = LOCAL_ROOT / Path(actual_name).parent
        LOCAL_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        Thread(target=_log_to_file, daemon=True).start()
    
    # Initialize server connection if enabled
    if SERVER_URL:
        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{SERVER_URL}/init", 
                    json={
                        "run_name": actual_name,
                        "info": combined_info
                    }
                )
                response.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Error initializing server connection: {e}")
        Thread(target=_log_to_server, daemon=True).start()
    
    return actual_name

def log(entry_name, data):
    """Log data to both local file and server if enabled.
    
    Args:
        entry_name (str): Name of the entry/category for the log
        data (dict): Data to log
    """
    if RUN_NAME is None:
        raise RuntimeError("Must call init() before logging")
    
    # Add timestamp to the data
    if not isinstance(data, dict):
        raise ValueError("data must be a dictionary")
        
    data = {"timestamp": datetime.now().isoformat()} | data
    
    # Queue for local file if enabled
    if LOCAL_ROOT:
        LOCAL_QUEUE.put((entry_name, data))
    
    # Queue for server if enabled
    if SERVER_URL:
        SERVER_QUEUE.put({
            "run_name": RUN_NAME,
            "entry_name": entry_name,
            "data": data
        })

def _log_to_file():
    """Background thread for writing to local files."""
    while True:
        entry_name, data = LOCAL_QUEUE.get()  # Wait for data
        try:
            log_file = LOCAL_BACKUP_DIR / f"{entry_name}.jsonl"
            with log_file.open("a") as f:
                json.dump(data, f)
                f.write("\n")
        except Exception as e:
            print(f"WARNING: Failed to write to local file {entry_name}.jsonl: {e}")
        LOCAL_QUEUE.task_done()

def _log_to_server():
    """Background thread for sending data to the server."""
    while True:
        data = SERVER_QUEUE.get()  # Wait for data
        try:
            with httpx.Client() as client:
                response = client.post(f"{SERVER_URL}/log", json=data)
                response.raise_for_status()
                result = response.json()
                if result.get("status") == "error":
                    print(f"WARNING: Server rejected log for {data['entry_name']}: {result.get('message')}")
        except Exception as e:
            print(f"WARNING: Failed to send log for {data['entry_name']} to server: {e}")
        SERVER_QUEUE.task_done()

def wait_for_completion():
    """Wait for all queued logs to be written to disk and sent to server."""
    if LOCAL_ROOT:
        LOCAL_QUEUE.join()  # Wait for all local file writes to complete
    if SERVER_URL:
        SERVER_QUEUE.join()  # Wait for all server requests to complete


if __name__ == "__main__":
    init("example_experiment/run1")
    
    while True:
        # Simulate experiment data
        data = {
            "client_id": "client_1",
            "value": 42
        }
        
        log("results", data)
        time.sleep(5)  # Simulate time-consuming experiment

