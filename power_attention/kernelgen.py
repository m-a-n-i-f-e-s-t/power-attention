import inspect
import re
import os
import hashlib
import linecache
from collections import defaultdict
from jinja2 import Template, Environment, meta
from pathlib import Path
from itertools import product

def parse_val(s):
    """Try to parse a value from a string into appropriate Python types.
    
    Supports:
    - int (e.g. '42', '-17')
    - float (e.g. '3.14', '-0.001', '1e-10')
    - bool ('True', 'False', 'true', 'false')
    - tuple (e.g. '(1, 2, 3)', '()')
    - list (e.g. '[1, 2, 3]', '[]')
    - str (default fallback)
    
    Args:
        s: Input string to parse
        
    Returns:
        Parsed value as int, float, bool, tuple, list, or str
    """
    # Handle None/empty cases
    if s is None or s.strip() == '':
        return None
        
    # Clean the input
    s = s.strip()
    
    # Try boolean first (to avoid 'True' being parsed as a string)
    lower_s = s.lower()
    if lower_s in ('true', 'false'):
        return lower_s == 'true'
    
    # Try tuple
    if s.startswith('(') and s.endswith(')'):
        if s == '()':  # Handle empty tuple
            return tuple()
        try:
            # Remove parentheses and split by comma, handling empty last element
            items = [parse_val(item.strip()) for item in s[1:-1].split(',') if item.strip()]
            return tuple(items)
        except ValueError:
            pass
    
    # Try list
    if s.startswith('[') and s.endswith(']'):
        if s == '[]':  # Handle empty list
            return list()
        try:
            # Remove brackets and split by comma, handling empty last element
            items = [parse_val(item.strip()) for item in s[1:-1].split(',') if item.strip()]
            return items
        except ValueError:
            pass
    
    # Try integer
    try:
        return int(s)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(s)
    except ValueError:
        pass
    
    # Return as string if all else fails
    return s

def extract_constval_from_lines(line):
    """Extract constval from line of `var=(val1, val2, ...),...` into a dict[str, list]"""
    # Split by comma, but not within parentheses/brackets
    def split_outside_brackets(s):
        parts = []
        current = []
        bracket_count = 0
        
        for char in s:
            if char in '([':
                bracket_count += 1
            elif char in ')]':
                bracket_count -= 1
            elif char == ',' and bracket_count == 0:
                parts.append(''.join(current))
                current = []
                continue
            current.append(char)
        
        if current:
            parts.append(''.join(current))
        return parts

    pairs = split_outside_brackets(line)
    result = {}
    for pair in pairs:
        if '=' not in pair:
            continue
        key, value = pair.split('=', 1)
        result[key.strip()] = parse_val(value.strip())
    return result

def extract_template_from_docstring(func):
    """Extract template code from function docstring between <kernelgen> tags"""
    docstring = func.__doc__ or ""
    match_with_constval = re.search(r'<kernelgen (.*?)>(.*?)</kernelgen>', docstring, re.DOTALL)
    match_without_constval = re.search(r'<kernelgen>(.*?)</kernelgen>', docstring, re.DOTALL)
    
    if match_with_constval:
        constval_str = match_with_constval.group(1)
        template_str = match_with_constval.group(2)
    elif match_without_constval:
        constval_str = None
        template_str = match_without_constval.group(1)
    else:
        raise ValueError("No template found in function docstring")
    
    DEBUG = os.environ.get("KERNELGEN_DEBUG", None) is not None
    if DEBUG:
        print(f"\nExtracted template from {func.__name__}:")
        print("=" * 80)
        print(template_str)
        print("=" * 80)
    return template_str, extract_constval_from_lines(constval_str) if constval_str else {}

def get_template_variables(template_str):
    """Extract all variables used in the template"""
    DEBUG = os.environ.get("KERNELGEN_DEBUG", None) is not None
    env = Environment()
    try:
        ast = env.parse(template_str)
        variables = meta.find_undeclared_variables(ast)
        if DEBUG:
            print("\nTemplate variables found:", sorted(list(variables)))
        return variables
    except Exception as e:
        print("\nError parsing template:")
        print("-" * 80)
        # Print template with line numbers for easier debugging
        for i, line in enumerate(template_str.split('\n'), 1):
            print(f"{i:3d} | {line}")
        print("-" * 80)
        print(f"Error: {str(e)}")
        raise

def get_generated_file_path(func, config_hash):
    """Generate a unique file path for the rendered template"""
    module_path = Path(inspect.getmodule(func).__file__)
    generated_dir = module_path.parent / "generated"
    generated_dir.mkdir(exist_ok=True)
    
    # Create filename using function name and config hash
    filename = f"{func.__name__}_{config_hash}.py"
    return generated_dir / filename

def get_rendered_dir(func):
    """Get the _rendered directory path for the module"""
    module_path = Path(inspect.getmodule(func).__file__)
    rendered_dir = module_path.parent / "_rendered"
    rendered_dir.mkdir(exist_ok=True)
    return rendered_dir

def compute_config_hash(rendered_code):
    """Compute a deterministic hash of config values"""
    return hashlib.md5(rendered_code.encode()).hexdigest()[:8]

def serialize_dict(d):
    """Serialize a dictionary into a deterministic string representation.
    
    Args:
        d: Dictionary to serialize
        
    Returns:
        String representation of the dictionary with keys sorted alphabetically
    """
    if not isinstance(d, dict):
        return str(d)
        
    # Sort items by key for deterministic output
    items = []
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, dict):
            items.append(f"{k}:{serialize_dict(v)}")
        else:
            items.append(f"{k}:{v}")
            
    return "{" + ",".join(items) + "}"

class Condition:
    """A class representing a condition, useful for printing out triton-allowed conditions"""
    def __init__(self, conditions=None):
        if isinstance(conditions, dict):
            self.conditions = [conditions]
        elif isinstance(conditions, Condition):
            self.conditions = conditions.conditions
        elif isinstance(conditions, list):
            assert isinstance(conditions[0], dict) or len(conditions) == 0
            self.conditions = conditions
        elif conditions is None:
            self.conditions = []
        else:
            raise ValueError(f"Invalid conditions type: {type(conditions)}")
    
    def __or__(self, other):
        return Condition(self.conditions + other.conditions)
    
    @staticmethod
    def print_condition(condition: dict):
        """ Given a list of key-value pairs, print out a corresponding binary condition
        """
        if len(condition) == 0:
            return "True"
        elif len(condition) == 1:
            key, val = list(sorted(condition.items(), key=lambda x: x[0]))[0]
            return f"({key} == {val})"
        else:
            key, val = list(sorted(condition.items(), key=lambda x: x[0]))[0]
            condition.pop(key)
            res = f"({key} == {val}) and ({Condition.print_condition(condition)})"
            condition[key] = val
            return res
    
    def print(self):
        """ Print out the condition as a triton-allowed condition """
        if len(self.conditions) == 0:
            return "True"
        elif len(self.conditions) == 1:
            return Condition.print_condition(self.conditions[0])
        else:
            self.conditions = sorted(self.conditions, key=serialize_dict)
            c0 = self.conditions.pop(0)
            res = f"({Condition.print_condition(c0)}) or ({self.print()})"
            self.conditions.insert(0, c0)
            return res
    

def get_function_signature(func):
    """Extract function signature including type hints but excluding decorators"""
    source_lines = inspect.getsource(func).split('\n')
    sig_lines = []
    in_signature = False
    parentheses_count = 0
    
    for line in source_lines:
        stripped = line.strip()
        # Skip decorators
        if stripped.startswith('@'):
            continue
            
        # Start of function definition
        if stripped.startswith('def '):
            in_signature = True
            parentheses_count = line.count('(') - line.count(')')
            sig_lines.append(line)
            if parentheses_count == 0:
                break
            continue
            
        # Continue capturing multi-line signature
        if in_signature:
            sig_lines.append(line)
            parentheses_count += line.count('(') - line.count(')')
            if parentheses_count == 0:
                break
    
    return '\n'.join(sig_lines)

def extract_constexpr_declarations(template_str):
    """Extract all constexpr declarations from the template"""
    # Find all lines that contains ": tl.constexpr"
    lines = template_str.split('\n')
    constexpr_lines = [line for line in lines if ": tl.constexpr" in line or ":tl.constexpr" in line]
    rest_lines = [line for line in lines if line not in constexpr_lines]
    return constexpr_lines, "\n".join(rest_lines)

def render_template(template_str, context, constval_dict, func_signature):
    """Render template with given context and proper function definition"""
    # Create the full template with imports and function definition
    kw = {k: v if isinstance(v, (list, tuple)) else [v] for k, v in constval_dict.items()}
    constexpr_lines, rest_lines = extract_constexpr_declarations(template_str)
    first = True
    rendered_code = []
    if len(kw) == 0:
        template = Template(rest_lines)
        rendered_lines = template.render(**context)
        rendered_code.append(rendered_lines)
    else:
        rest_lines = "\n".join([f"    {line}" for line in rest_lines.split("\n")])
        template = Template(rest_lines)
        for vals in product(*(kw[k] for k in sorted(kw.keys()))):
            const_context = {k: v for k, v in zip(sorted(kw.keys()), vals)}
            rendered_lines = template.render(**context, **const_context)
            rendered_code.append(f"""
{'if' if first else 'elif'} {Condition(const_context).print()}: {rendered_lines}""")
            first = False

    return "\n".join(rendered_code), constexpr_lines

def kernelgen(configs):
    def decorator(func):
        DEBUG = os.environ.get("KERNELGEN_DEBUG", None) is not None

        template_str, constval_dict = extract_template_from_docstring(func)
        
        if not template_str:
            raise ValueError("No template found in function docstring")
        
        # Get template variables and dependent functions
        template_vars = get_template_variables(template_str)
        rendered_dir = get_rendered_dir(func)
        
        # Generate all variants up front
        variants = defaultdict(set) # dict of rendered_code -> match_dict
        
        if DEBUG:
            print(f"\nPre-rendering kernel variants for {func.__name__}:")
        constexpr_lines = []

        for config in sorted(configs, key=serialize_dict):
            # Create context for this config
            context = {
                var: config.kwargs[var]
                for var in template_vars 
                if var in config.kwargs
            }
            # Check for missing variables
            missing_vars = template_vars - set(context.keys()) - set(constval_dict.keys())
            if missing_vars:
                print(f"Warning: Config {config} missing variables: {missing_vars}")
                continue
                
            # Render template
            rendered_code, constexpr_lines = render_template(template_str, context, constval_dict, "")  # Empty signature since we'll add it later
            
            # Create match dict for this config
            match_dict = frozenset({
                key: val for key, val in config.kwargs.items()
                if isinstance(val, (int, float, bool, str))  # Only include static values
            }.items())
            
            variants[rendered_code].add(match_dict)
        
        # Generate the dispatcher function code
        dispatcher_code = []

        # Add imports
        dispatcher_code.append("import triton")
        dispatcher_code.append("import triton.language as tl")
        dispatcher_code.append("")
        
        # Add the main function signature and body
        dispatcher_code.append(get_function_signature(func))
        
        # Add constexpr declarations
        dispatcher_code.extend(f"    {line}" for line in constexpr_lines)
        
        # Add the if-else chain
        first = True
        for rendered_code, match_dicts in variants.items():
            condition = Condition()
            
            for match_dict in match_dicts:
                condition |= Condition(dict(match_dict))
            
            if first:
                dispatcher_code.append(f"    if {condition.print()}:")
                first = False
            else:
                dispatcher_code.append(f"    elif {condition.print()}:")
            
            # Indent the rendered code
            indented_code = "\n".join("        " + line for line in rendered_code.split("\n"))
            dispatcher_code.append(indented_code)
        
        # Add final else clause
        dispatcher_code.append("    else:")
        dispatcher_code.append('        tl.static_assert(False, "No matching config found")')
        
        # Create the final function
        final_code = "\n".join(dispatcher_code)
        
        # Save the generated code to _rendered directory
        filename = f"{func.__name__}_dispatcher.py"
        file_path = rendered_dir / filename
        with open(file_path, 'w') as f:
            f.write(final_code)
        if DEBUG:
            print(f"Generated dispatcher -> {file_path}")
        
        linecache.cache[filename] = (
            len(final_code),
            None,
            final_code.splitlines(keepends=True),
            filename
        )

        code_obj = compile(final_code, filename, "exec")
        namespace = {}
        exec(code_obj, func.__globals__, namespace)
        generated_func = namespace[func.__name__]

        # Get the generated function
        return generated_func
    
    return decorator