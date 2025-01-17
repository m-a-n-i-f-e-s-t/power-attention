# saves the longcrawl64 dataset to a binary file for training. assumes access to pre-shuffled zarr shards. 
# following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
#
# Most people will just want to download the preprocessed version from the public bucket.
# If building from raw, needs the preprocessed shuffled longcrawl64 zarr shards in /storage/datasets/shuffled_zarrs.
# These can be acquired with the following command, which will take just the first 10 shards (convenient if you are making a smaller version of the dataset):
# $ GSUTIL_PARALLEL_THREAD_COUNT=8 GSUTIL_PARALLEL_PROCESS_COUNT=64 gsutil -m cp -r "gs://longcrawl-staging/output/shuffled_zarrs/[0-9]-of-128.zarr" /storage/datasets/shuffled_zarrs
# This data requires a login, email me and I will give you access.

import os
import numpy as np
from tqdm import tqdm
import zarr
import click

@click.command()
@click.option('--build-from-raw', is_flag=True, default=False, help='Build dataset from raw zarr shards')
@click.option('--destination-folder', type=str, default='~/mai_datasets/lc64', 
              help='Destination folder for the processed dataset')
@click.option('--source-folder', type=str, default=None,
              help='Source folder containing zarr shards')
@click.option('--num-shards', type=int, default=None,
              help='Number of zarr shards to process')
@click.option('--train-tokens', type=int, default=None,
              help='Target number of training tokens')
def main(build_from_raw, destination_folder, source_folder, num_shards, train_tokens):
    # Expand user path
    destination_folder = os.path.expanduser(destination_folder)
    os.makedirs(destination_folder, exist_ok=True)

    # Validate arguments
    if not build_from_raw and any([source_folder, num_shards, train_tokens]):
        raise click.UsageError(
            "Cannot specify source_folder, num_shards, or train_tokens without --build-from-raw flag"
        )

    if build_from_raw:
        if not source_folder:
            source_folder = '/storage/datasets/shuffled_zarrs'
        if not num_shards:
            num_shards = 128
        if not train_tokens:
            train_tokens = 50_000_000_000
        
        # Reserve 0.05% for validation on top of train tokens
        val_tokens = int(train_tokens * 0.0005)

        # Create memory-mapped files
        train_file = os.path.join(destination_folder, 'train.bin')
        val_file = os.path.join(destination_folder, 'val.bin')
        dtype = np.uint16

        train_arr = np.memmap(train_file, dtype=dtype, mode='w+', shape=(train_tokens,))
        val_arr = np.memmap(val_file, dtype=dtype, mode='w+', shape=(val_tokens,))

        # Process training data starting from shard 0
        train_idx, last_train_shard = process_shards(train_arr, train_tokens, source_folder, num_shards, 'Processing training data', start_shard=0)
        
        # Process validation data starting from the next shard
        val_start_shard = last_train_shard + 1
        val_idx, _ = process_shards(val_arr, val_tokens, source_folder, num_shards, 'Processing validation data', start_shard=val_start_shard)

        print(f'Wrote {train_idx} training tokens and {val_idx} validation tokens')
        print(f'train.bin is {os.path.getsize(train_file)/1e9:.1f}GB')
        print(f'val.bin is {os.path.getsize(val_file)/1e9:.1f}GB')
    else:
        # TODO: Implement non-raw processing logic
        pass



def process_shards(arr, target_tokens, source_folder, num_shards, desc, start_shard=0):
    """Process zarr shards and write to a memmap array until target_tokens is reached.
    Returns (tokens_written, last_shard_processed)"""
    np_idx = 0
    
    with tqdm(total=target_tokens, desc=desc) as pbar:
        for shard in range(start_shard, num_shards):
            pbar.set_description(f"{desc} (shard {shard}/{num_shards})")
            if np_idx >= target_tokens:
                break
                
            shard_path = os.path.join(source_folder, f'{shard}-of-{num_shards}.zarr')
            z = zarr.open(shard_path, mode='r')
            
            # Process in chunks to avoid loading entire shard into memory
            chunk_size = 10_000
            for zarr_row_idx in range(0, z.shape[0], chunk_size):
                if np_idx >= target_tokens:
                    break
                
                # load the chunk
                end_row_idx = min(zarr_row_idx + chunk_size, z.shape[0])
                chunk = z[zarr_row_idx:end_row_idx]
                # add the eot token (hardcoded for tiktoken's gpt2 bpe)
                chunk[:,-1] = 50256
                # flatten the chunk
                chunk = chunk.reshape(-1)
                
                # Calculate how many tokens we can still add
                remaining = target_tokens - np_idx
                chunk = chunk[:remaining]  # Truncate if needed
                
                arr[np_idx:np_idx + len(chunk)] = chunk
                np_idx += len(chunk)
                
                # Update progress bar with chunk info and token count
                pbar.update(len(chunk))
    
    arr.flush()
    return np_idx, shard

if __name__ == '__main__':
    main()
