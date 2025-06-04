"""
Utility module for standardized Python executable discovery across environments.

This module provides a single source of truth for getting the correct Python executable
path across different platforms (Windows, Linux, WSL) with appropriate environment
detection, logging, and override capabilities.

This solves critical issues in the codebase related to:
1. Inconsistent Python environments for subprocesses
2. Hard-coded paths in WSL environments
3. Different approaches to finding the Python executable in different modules
4. Unexpected behavior when sys.executable doesn't match the virtual environment

Usage:
    from src.lora_training_pipeline.utils.process_executable import get_python_executable

    # Get the correct Python executable for the current environment
    python_exe = get_python_executable()

    # Use it in subprocess calls
    subprocess.Popen([python_exe, "-m", "your_module"])

    # Get a complete command list for subprocess
    cmd = get_python_command("script.py", "arg1", "arg2")
    subprocess.Popen(cmd)

    # Run a Python script directly
    process = run_python_script("script.py", "arg1", "arg2")
"""

import os
import sys
import logging
import platform
import subprocess
import shutil
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set debug level if environment variable is present
if os.environ.get("DEBUG_PROCESS_EXECUTABLE", "").lower() in ("true", "1", "yes"):
    logger.setLevel(logging.DEBUG)
    logger.debug("Debug logging enabled for process_executable module")

# Environment variable that can override the detected Python executable
PYTHON_EXECUTABLE_OVERRIDE = "LORA_PYTHON_EXECUTABLE"


def is_wsl() -> bool:
    """
    Detect if running in Windows Subsystem for Linux (WSL).

    Returns:
        bool: True if running in WSL, False otherwise.
    """
    try:
        if os.path.exists("/proc/version"):
            with open("/proc/version", "r") as f:
                version_info = f.read().lower()
                wsl_detected = "microsoft" in version_info or "wsl" in version_info
                logger.debug(f"WSL detection: {'Detected' if wsl_detected else 'Not detected'} in /proc/version")
                return wsl_detected
        else:
            logger.debug("WSL detection: /proc/version not found, not WSL")
    except Exception as e:
        logger.warning(f"Error during WSL detection: {e}")
        logger.debug(traceback.format_exc())
    return False


def get_windows_python_path() -> Optional[str]:
    """
    Find the Python executable path on Windows.

    This searches for python.exe in common locations including the current
    virtual environment if active.

    Returns:
        Optional[str]: Path to python.exe or None if not found.
    """
    logger.debug("Searching for Python executable on Windows")

    try:
        # Check if running in a virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        logger.debug(f"Running in virtual environment: {in_venv}")

        if in_venv:
            venv_path = os.path.join(sys.prefix, 'Scripts', 'python.exe')
            if os.path.exists(venv_path):
                logger.debug(f"Found Python in virtual environment: {venv_path}")
                return venv_path
            else:
                logger.warning(f"Virtual environment detected but python.exe not found at expected path: {venv_path}")

        # Try to find python in PATH
        python_exe = shutil.which('python')
        if python_exe:
            logger.debug(f"Found Python in PATH: {python_exe}")
            return python_exe

        # Common installation locations
        logger.debug("Searching common Windows Python installation locations")
        common_locations = [
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Python'),
            os.path.join(os.environ.get('PROGRAMFILES', ''), 'Python'),
            os.path.join(os.environ.get('PROGRAMFILES(X86)', ''), 'Python'),
        ]

        for location in common_locations:
            if os.path.exists(location):
                logger.debug(f"Checking Python directory: {location}")
                try:
                    python_dirs = [d for d in os.listdir(location) if d.startswith('Python')]
                    if python_dirs:
                        logger.debug(f"Found Python directories: {python_dirs}")
                        for py_dir in sorted(python_dirs, reverse=True):  # Prefer newer versions
                            python_exe = os.path.join(location, py_dir, 'python.exe')
                            if os.path.exists(python_exe):
                                logger.debug(f"Found Python executable: {python_exe}")
                                return python_exe
                except Exception as e:
                    logger.warning(f"Error listing directory {location}: {e}")

        logger.warning("No Python executable found in common Windows locations")
    except Exception as e:
        logger.error(f"Error while searching for Windows Python executable: {e}")
        logger.debug(traceback.format_exc())

    logger.warning("Failed to find Python executable on Windows")
    return None


def get_linux_python_path() -> Optional[str]:
    """
    Find the Python executable path on Linux.

    This checks for active virtual environments first, then falls back to system Python.

    Returns:
        Optional[str]: Path to python executable or None if not found.
    """
    logger.debug("Searching for Python executable on Linux")

    try:
        # Check if in virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        logger.debug(f"Running in virtual environment: {in_venv}")

        if in_venv:
            venv_path = os.path.join(sys.prefix, 'bin', 'python')
            if os.path.exists(venv_path):
                logger.debug(f"Found Python in virtual environment: {venv_path}")
                return venv_path
            else:
                logger.warning(f"Virtual environment detected but python not found at expected path: {venv_path}")

        # Try to find python in PATH
        logger.debug("Searching for Python in PATH")
        python_cmds_tried = []
        for python_cmd in ['python3', 'python']:
            python_cmds_tried.append(python_cmd)
            python_path = shutil.which(python_cmd)
            if python_path:
                logger.debug(f"Found Python in PATH using '{python_cmd}': {python_path}")

                # Verify it's executable
                if os.access(python_path, os.X_OK):
                    return python_path
                else:
                    logger.warning(f"Found Python at {python_path} but it is not executable")

        logger.warning(f"No Python found in PATH. Tried: {', '.join(python_cmds_tried)}")

        # As a last resort, try common locations
        common_paths = [
            "/usr/bin/python3",
            "/usr/bin/python",
            "/usr/local/bin/python3",
            "/usr/local/bin/python"
        ]

        logger.debug(f"Checking common Linux Python locations")
        for path in common_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                logger.debug(f"Found Python at common location: {path}")
                return path

        logger.warning("No Python executable found in common Linux locations")
    except Exception as e:
        logger.error(f"Error while searching for Linux Python executable: {e}")
        logger.debug(traceback.format_exc())

    logger.warning("Failed to find Python executable on Linux")
    return None


def get_wsl_python_path() -> Optional[str]:
    """
    Find the Python executable path in WSL environment.

    This handles the special case of WSL, checking for virtual environments
    and system Python installations within the WSL environment.

    Returns:
        Optional[str]: Path to python executable in WSL context or None if not found.
    """
    logger.debug("Searching for Python executable in WSL environment")

    try:
        # First check for hard-coded Windows path from run_pipeline.py
        win_path = r"C:\Users\arnon\Documents\dev\projects\incoming\LoRA_Training_Pipeline\.venv\Scripts\python.exe"
        wsl_path = convert_win_path_to_wsl(win_path)

        logger.debug(f"Checking for known Windows Python path: {win_path}")
        logger.debug(f"Converted to WSL path: {wsl_path}")

        # This is unlikely to work directly as WSL doesn't map Windows paths like this,
        # but we'll check anyway, and log the result
        if os.path.exists(win_path):
            logger.info(f"Found Windows Python path directly accessible in WSL: {win_path}")
            return win_path

        # In WSL, use the same approach as Linux
        logger.debug("Using Linux Python detection approach for WSL")
        linux_python = get_linux_python_path()

        if linux_python:
            return linux_python
        else:
            logger.warning("No Python executable found in WSL using Linux approach")

        # As a last resort, check if we can just use sys.executable
        if os.path.exists(sys.executable) and os.access(sys.executable, os.X_OK):
            logger.debug(f"Using fallback to sys.executable in WSL: {sys.executable}")
            return sys.executable

    except Exception as e:
        logger.error(f"Error while searching for WSL Python executable: {e}")
        logger.debug(traceback.format_exc())

    logger.warning("Failed to find Python executable in WSL environment")
    return None


def normalize_path(path: str) -> str:
    """
    Normalize path according to the current platform.

    Args:
        path (str): The path to normalize.

    Returns:
        str: Normalized path appropriate for the current platform.
    """
    if not path:
        logger.warning("Empty path provided to normalize_path")
        return ""

    try:
        logger.debug(f"Normalizing path: {path}")

        if platform.system() == "Windows":
            # Convert forward slashes to backslashes on Windows
            norm_path = os.path.normpath(path)
            logger.debug(f"Normalized Windows path: {norm_path}")
            return norm_path

        elif is_wsl():
            # Handle paths in WSL
            if path.startswith('/mnt/'):
                # Already a WSL path
                logger.debug(f"Path already in WSL format: {path}")
                return path

            elif ':' in path and path[1] == ':':  # Windows path like C:\...
                # Convert Windows path to WSL path
                drive = path[0].lower()
                wsl_path = f"/mnt/{drive}/{path[3:].replace('\\', '/')}"
                logger.debug(f"Converted Windows path to WSL format: {path} -> {wsl_path}")
                return wsl_path

            # Other path formats
            norm_path = os.path.normpath(path)
            logger.debug(f"Normalized WSL path: {norm_path}")
            return norm_path

        else:
            # Linux/Unix paths
            norm_path = os.path.normpath(path)
            logger.debug(f"Normalized Linux path: {norm_path}")
            return norm_path

    except Exception as e:
        logger.error(f"Error normalizing path '{path}': {e}")
        logger.debug(traceback.format_exc())
        # Return the original path on error - safer than returning nothing
        return path


def convert_win_path_to_wsl(win_path: str) -> str:
    """
    Convert a Windows path to WSL format.
    
    Args:
        win_path (str): Windows-style path (e.g., C:\\Users\\...)
        
    Returns:
        str: WSL-style path (e.g., /mnt/c/Users/...)
    """
    if not win_path or len(win_path) < 2 or win_path[1] != ':':
        return win_path  # Not a Windows path or invalid format
        
    drive = win_path[0].lower()
    path_part = win_path[2:].replace('\\', '/')
    if path_part.startswith('/'):
        path_part = path_part[1:]
    
    return f"/mnt/{drive}/{path_part}"


def convert_wsl_path_to_win(wsl_path: str) -> str:
    """
    Convert a WSL path to Windows format.
    
    Args:
        wsl_path (str): WSL-style path (e.g., /mnt/c/Users/...)
        
    Returns:
        str: Windows-style path (e.g., C:\\Users\\...)
    """
    if not wsl_path.startswith('/mnt/') or len(wsl_path) < 6:
        return wsl_path  # Not a WSL path or invalid format
    
    drive = wsl_path[5].upper()
    path_part = wsl_path[6:].replace('/', '\\')
    
    return f"{drive}:{path_part}"


def get_python_executable(override: Optional[str] = None) -> str:
    """
    Get the appropriate Python executable path for the current environment.

    This function serves as the main entry point for getting the Python executable
    across the codebase. It respects environment variable overrides and handles
    platform-specific detection logic.

    Args:
        override (Optional[str]): Direct override for the Python path.

    Returns:
        str: Path to the Python executable to use.

    Raises:
        RuntimeError: If no suitable Python executable could be found.
    """
    logger.info("Determining appropriate Python executable")

    try:
        # Check for direct override parameter
        if override:
            logger.info(f"Using provided Python executable override: {override}")
            if not os.path.exists(override):
                logger.warning(f"WARNING: Override Python path does not exist: {override}")
            return normalize_path(override)

        # Check for environment variable override
        env_override = os.environ.get(PYTHON_EXECUTABLE_OVERRIDE)
        if env_override:
            logger.info(f"Using Python executable from environment variable {PYTHON_EXECUTABLE_OVERRIDE}: {env_override}")
            if not os.path.exists(env_override):
                logger.warning(f"WARNING: Environment Python path does not exist: {env_override}")
            return normalize_path(env_override)

        # Get current OS and platform details for diagnostics
        logger.info(f"System: {platform.system()}, Platform: {platform.platform()}")
        logger.info(f"Current sys.executable: {sys.executable}")
        logger.info(f"Python version: {platform.python_version()}")

        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            logger.info(f"Virtual environment detected. Prefix: {sys.prefix}")
            if hasattr(sys, 'base_prefix'):
                logger.info(f"Base prefix: {sys.base_prefix}")

        # Detect the appropriate path based on platform
        python_path = None

        if is_wsl():
            logger.info("WSL environment detected")
            python_path = get_wsl_python_path()
        elif platform.system() == "Windows":
            logger.info("Windows environment detected")
            python_path = get_windows_python_path()
        else:
            logger.info(f"Unix/Linux environment detected: {platform.system()}")
            python_path = get_linux_python_path()

        # Fall back to sys.executable if all else fails
        if not python_path:
            logger.warning("Could not detect Python path through dedicated methods, using sys.executable")
            python_path = sys.executable

            # Verify sys.executable is usable
            if not os.path.exists(python_path):
                logger.error(f"sys.executable does not exist: {python_path}")

                # Last ditch attempt - try to find any Python
                logger.warning("Attempting desperate last-resort Python search")
                which_python = shutil.which("python") or shutil.which("python3")
                if which_python:
                    logger.warning(f"Found fallback Python via which: {which_python}")
                    python_path = which_python
                else:
                    logger.critical("Could not find any Python executable on the system")

        # Final verification
        if not python_path:
            raise RuntimeError("No Python executable found through any detection method")

        if not os.path.exists(python_path):
            raise RuntimeError(f"Selected Python executable does not exist: {python_path}")

        # Verify it's executable on Unix-like systems
        if platform.system() != "Windows" and not os.access(python_path, os.X_OK):
            logger.error(f"Selected Python path is not executable: {python_path}")

        # Verify the Python is working by running a simple command
        try:
            logger.debug(f"Verifying Python executable: {python_path}")
            result = subprocess.run(
                [python_path, "-c", "import sys; print(sys.version)"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Verified Python {python_path} is working. Version: {result.stdout.strip()}")
            else:
                logger.warning(f"Python verification command exited with code {result.returncode}")
                logger.warning(f"stderr: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout while verifying Python executable")
        except Exception as e:
            logger.error(f"Error verifying Python executable: {e}")

        # Return the path
        normalized_path = normalize_path(python_path)
        logger.info(f"Selected Python executable: {normalized_path}")
        return normalized_path

    except Exception as e:
        logger.critical(f"Critical error determining Python executable: {e}")
        logger.debug(traceback.format_exc())

        # In case of total failure, return sys.executable as a last resort
        # rather than completely crashing
        logger.critical(f"Using current sys.executable as emergency fallback: {sys.executable}")
        return sys.executable


def get_python_command(script_path: str, *args: str, **kwargs: Any) -> List[str]:
    """
    Get a properly formatted command list for subprocess execution.

    This function constructs a command list that can be passed directly to
    subprocess.Popen or similar functions.

    Args:
        script_path (str): Path to the Python script to execute.
        *args (str): Additional command-line arguments.
        **kwargs (Any): Additional options:
            - env_vars (Dict[str, str]): Environment variables to set.
            - python_path (str): Override for Python executable path.
            - module (bool): If True, run as a module (use -m)

    Returns:
        list: Command list ready for subprocess execution.
    """
    try:
        env_vars = kwargs.get('env_vars', {})
        python_path = kwargs.get('python_path')
        as_module = kwargs.get('module', False)

        logger.info(f"Creating Python command for: {script_path}")

        # Get the Python executable
        python_exe = get_python_executable(override=python_path)

        # Normalize the script path for the current platform
        if script_path:
            normalized_script = normalize_path(script_path)
            logger.debug(f"Normalized script path: {normalized_script}")
        else:
            normalized_script = ""
            logger.warning("Empty script path provided to get_python_command")

        # Construct the command
        cmd = [python_exe]

        # Check if running as module
        if as_module:
            if not normalized_script:
                raise ValueError("Script path required when running as module")
            cmd.extend(["-m", normalized_script])
            logger.debug(f"Running as module: {normalized_script}")
        elif normalized_script:
            cmd.append(normalized_script)

        # Add any additional arguments
        if args:
            cmd.extend(args)
            logger.debug(f"Added {len(args)} command line arguments")

        # Log the command being executed
        env_str = ' '.join(f'{k}={v}' for k, v in env_vars.items()) if env_vars else ''
        cmd_str = ' '.join(cmd)
        if env_str:
            logger.info(f"Command with env vars: {env_str} {cmd_str}")
        else:
            logger.info(f"Command: {cmd_str}")

        return cmd

    except Exception as e:
        logger.error(f"Error creating Python command: {e}")
        logger.debug(traceback.format_exc())

        # If we can't create a proper command, return a minimal working placeholder
        # that will just invoke Python without doing anything
        logger.critical(f"Returning emergency fallback Python command")
        return [sys.executable, "-c", "import sys; print('COMMAND CREATION ERROR')"]


def run_python_script(script_path: str, *args: str, **kwargs: Any) -> subprocess.Popen:
    """
    Run a Python script as a subprocess with the correct Python executable.

    Args:
        script_path (str): Path to the Python script to execute.
        *args (str): Additional command-line arguments.
        **kwargs (Any): Additional options:
            - env_vars (Dict[str, str]): Environment variables to set.
            - python_path (str): Override for Python executable path.
            - module (bool): If True, run as a module (use -m)
            - capture_output (bool): If True, capture stdout/stderr.
            - Any other kwargs are passed directly to subprocess.Popen.

    Returns:
        subprocess.Popen: The subprocess object for the running script.
    """
    try:
        env_vars = kwargs.pop('env_vars', {})
        python_path = kwargs.pop('python_path', None)
        as_module = kwargs.pop('module', False)
        capture_output = kwargs.pop('capture_output', False)

        logger.info(f"Running Python script: {script_path}")

        # Get the command list
        cmd = get_python_command(
            script_path,
            *args,
            env_vars=env_vars,
            python_path=python_path,
            module=as_module
        )

        # Set up the environment
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
            logger.debug(f"Added {len(env_vars)} environment variables")

        # Configure stdout/stderr
        if capture_output:
            kwargs['stdout'] = subprocess.PIPE
            kwargs['stderr'] = subprocess.PIPE
            logger.debug("Capturing process output")

        # Set up process group to enable proper cleanup
        kwargs.setdefault('start_new_session', True)

        # Run the subprocess with good exception handling
        try:
            logger.debug(f"Starting subprocess with Popen")
            process = subprocess.Popen(cmd, env=env, **kwargs)
            logger.info(f"Started Python process with PID {process.pid}")

            # Set up automatic cleanup on Python exit
            def cleanup_process():
                if process.poll() is None:  # Still running
                    logger.info(f"Cleaning up process {process.pid} on Python exit")
                    try:
                        process.terminate()
                    except Exception as cleanup_err:
                        logger.warning(f"Error terminating process {process.pid}: {cleanup_err}")

            # Register cleanup handler but don't overwrite existing ones
            try:
                import atexit
                atexit.register(cleanup_process)
                logger.debug(f"Registered cleanup handler for process {process.pid}")
            except Exception as e:
                logger.warning(f"Failed to register cleanup handler: {e}")

            return process

        except FileNotFoundError:
            logger.error(f"Python executable not found: {cmd[0]}")
            raise RuntimeError(f"Python executable not found: {cmd[0]}")

        except PermissionError:
            logger.error(f"Permission denied to execute: {cmd[0]}")
            raise RuntimeError(f"Permission denied to execute: {cmd[0]}")

        except subprocess.SubprocessError as e:
            logger.error(f"Subprocess error starting Python: {e}")
            raise RuntimeError(f"Subprocess error: {e}")

    except Exception as e:
        logger.error(f"Error running Python script {script_path}: {e}")
        logger.debug(traceback.format_exc())
        raise RuntimeError(f"Failed to run Python script: {e}")


def verify_python_version(min_version: tuple = (3, 8), python_path: Optional[str] = None) -> bool:
    """
    Verify that the detected Python executable meets minimum version requirements.

    Args:
        min_version (tuple): Minimum required Python version as (major, minor).
        python_path (Optional[str]): Optional custom Python path to check.

    Returns:
        bool: True if the Python version meets or exceeds the minimum, False otherwise.
    """
    try:
        # Get the Python executable to verify
        python_exe = python_path or get_python_executable()
        logger.info(f"Verifying Python version for: {python_exe}")

        if not os.path.exists(python_exe):
            logger.error(f"Python executable does not exist: {python_exe}")
            return False

        # Build min version string for display
        min_version_str = '.'.join(map(str, min_version))

        # Run Python with --version flag
        try:
            result = subprocess.run(
                [python_exe, "--version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10  # Add timeout to prevent hanging
            )

            if result.returncode != 0:
                logger.error(f"Python version check failed with code {result.returncode}")
                logger.error(f"stderr: {result.stderr}")
                return False

            # Parse version string (e.g., "Python 3.8.10" -> (3, 8, 10))
            version_output = result.stdout.strip() or result.stderr.strip()
            logger.debug(f"Python version output: {version_output}")

            # Handle different version formats
            if "Python" in version_output:
                parts = version_output.split()
                if len(parts) >= 2:
                    version_str = parts[1]  # Get "3.8.10" from "Python 3.8.10"
                else:
                    logger.error(f"Could not parse Python version from: {version_output}")
                    return False
            else:
                version_str = version_output.strip()

            logger.info(f"Detected Python version: {version_str}")

            # Parse version components
            version_parts = version_str.split('.')

            # Make sure we have at least major and minor version
            if len(version_parts) < 2:
                logger.error(f"Invalid version format: {version_str}")
                return False

            # Convert to tuple of integers for comparison
            try:
                version_tuple = tuple(int(part) for part in version_parts)
            except ValueError:
                logger.error(f"Could not convert version parts to integers: {version_parts}")
                return False

            # Check if version meets minimum requirement
            meets_requirement = version_tuple >= min_version

            if meets_requirement:
                logger.info(f"✓ Python version {version_str} meets minimum requirement {min_version_str}")
            else:
                logger.warning(f"✗ Python version {version_str} does not meet minimum requirement {min_version_str}")

            return meets_requirement

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout while checking Python version for {python_exe}")
            return False

        except subprocess.SubprocessError as e:
            logger.error(f"Subprocess error checking Python version: {e}")
            return False

    except (ValueError, IndexError) as e:
        logger.error(f"Error parsing Python version: {e}")
        logger.debug(traceback.format_exc())
        return False

    except Exception as e:
        logger.error(f"Unexpected error verifying Python version: {e}")
        logger.debug(traceback.format_exc())
        return False


# Command-line interface for testing
if __name__ == "__main__":
    import argparse

    # Configure root logger to show info level
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Python Executable Utility")
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--detect", action="store_true",
                     help="Detect and print the Python executable path")
    group.add_argument("--verify", action="store_true",
                     help="Verify the Python executable works and meets minimum version")
    group.add_argument("--test-run", action="store_true",
                     help="Test run a simple Python command")
    group.add_argument("--detect-wsl", action="store_true",
                     help="Check if running in WSL")
    group.add_argument("--path-convert", type=str, metavar="PATH",
                     help="Convert a path between Windows and WSL formats")

    parser.add_argument("--min-version", type=str, default="3.8",
                      help="Minimum Python version to check for (e.g., '3.8')")
    parser.add_argument("--python-path", type=str,
                      help="Specify a Python path to use or check")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Handle each command
    if args.detect:
        print("\n=== Python Executable Detection ===")
        try:
            python_path = get_python_executable(override=args.python_path)
            print(f"✅ Detected Python executable: {python_path}")
            print(f"Exists: {os.path.exists(python_path)}")
            if platform.system() != "Windows":
                print(f"Executable: {os.access(python_path, os.X_OK)}")
        except Exception as e:
            print(f"❌ Error detecting Python executable: {e}")
            exit(1)

    elif args.verify:
        print("\n=== Python Version Verification ===")
        try:
            # Parse min version
            min_version_parts = [int(x) for x in args.min_version.split('.')]
            min_version = tuple(min_version_parts)

            # Set Python path if provided
            python_path = args.python_path

            # Perform verification
            meets_req = verify_python_version(min_version, python_path)

            if meets_req:
                print(f"✅ Python meets minimum version requirement: {args.min_version}")
            else:
                print(f"❌ Python does not meet minimum version requirement: {args.min_version}")
                exit(1)
        except Exception as e:
            print(f"❌ Error verifying Python version: {e}")
            exit(1)

    elif args.test_run:
        print("\n=== Python Test Run ===")
        try:
            script = """
import sys
import platform
import os

print(f"Python version: {platform.python_version()}")
print(f"Python path: {sys.executable}")
print(f"Python prefix: {sys.prefix}")
print(f"Platform: {platform.platform()}")
print(f"Current directory: {os.getcwd()}")
"""
            # Create a temporary script file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
                f.write(script)
                script_path = f.name

            print(f"Running test script: {script_path}")
            process = run_python_script(script_path, python_path=args.python_path, capture_output=True)

            # Wait for process to finish
            stdout, stderr = process.communicate(timeout=10)

            # Print output
            if stdout:
                print("\nSTDOUT:")
                print(stdout.decode('utf-8', errors='replace'))

            if stderr:
                print("\nSTDERR:")
                print(stderr.decode('utf-8', errors='replace'))

            # Remove temp file
            os.unlink(script_path)

            # Check return code
            if process.returncode == 0:
                print(f"✅ Python test run succeeded")
            else:
                print(f"❌ Python test run failed with code {process.returncode}")
                exit(process.returncode)

        except Exception as e:
            print(f"❌ Error in test run: {e}")
            exit(1)

    elif args.detect_wsl:
        print("\n=== WSL Detection ===")
        wsl_detected = is_wsl()
        if wsl_detected:
            print("✅ Running in Windows Subsystem for Linux (WSL)")
        else:
            print("❌ Not running in WSL")

    elif args.path_convert:
        print("\n=== Path Conversion ===")
        path = args.path_convert

        if ":" in path and path[1] == ":":  # Windows path
            print(f"Detected Windows path: {path}")
            wsl_path = convert_win_path_to_wsl(path)
            print(f"Converted to WSL path: {wsl_path}")
        elif path.startswith("/mnt/"):  # WSL path
            print(f"Detected WSL path: {path}")
            win_path = convert_wsl_path_to_win(path)
            print(f"Converted to Windows path: {win_path}")
        else:
            print(f"Path format not recognized: {path}")

        print("\nCurrent platform path normalization:")
        norm_path = normalize_path(path)
        print(f"Normalized path: {norm_path}")