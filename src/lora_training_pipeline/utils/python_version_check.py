#!/usr/bin/env python
# LoRA_Training_Pipeline/src/lora_training_pipeline/utils/python_version_check.py
"""
Python Version Compatibility Check

This module provides utilities to check Python version compatibility with
different parts of the pipeline, particularly for Ray and PyTorch-Lightning
which have issues with Python 3.12.
"""

import sys
import platform
import warnings
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Set

# Define version compatibility ranges
VERSION_COMPATIBILITY = {
    "ray": {
        "min_python": (3, 7),
        "max_python": (3, 11, 99),  # Ray has issues with Python 3.12
        "recommended": (3, 10)
    },
    "lightning": {
        "min_python": (3, 8),
        "max_python": (3, 11, 99),  # PyTorch Lightning has issues with Python 3.12
        "recommended": (3, 10)
    },
    "torch": {
        "min_python": (3, 8),
        "max_python": (3, 11, 99),  # PyTorch stable releases have issues with Python 3.12
        "recommended": (3, 10)
    }
}

def get_python_version() -> Tuple[int, int, int]:
    """
    Get the current Python version as a tuple.
    
    Returns:
        tuple: (major, minor, micro) version numbers
    """
    version_info = sys.version_info
    return (version_info.major, version_info.minor, version_info.micro)

def check_python_version_compatibility(package: str) -> Dict[str, Union[bool, str]]:
    """
    Check if the current Python version is compatible with the specified package.
    
    Args:
        package: Name of the package to check compatibility for
        
    Returns:
        dict: Compatibility information
    """
    current_version = get_python_version()
    
    if package not in VERSION_COMPATIBILITY:
        return {
            "compatible": True,
            "message": f"No specific compatibility information for {package}",
            "warning": False
        }
    
    compat_info = VERSION_COMPATIBILITY[package]
    min_version = compat_info["min_python"]
    max_version = compat_info["max_python"]
    recommended = compat_info["recommended"]
    
    # Check if version is within range
    is_compatible = (
        current_version >= min_version and
        current_version <= max_version
    )
    
    # Check if version is recommended
    is_recommended = (
        current_version[0] == recommended[0] and
        current_version[1] == recommended[1]
    )
    
    # Create message based on compatibility
    if is_compatible:
        if is_recommended:
            message = f"Python {'.'.join(map(str, current_version))} is fully compatible with {package}"
            warning = False
        else:
            message = (
                f"Python {'.'.join(map(str, current_version))} is compatible with {package}, "
                f"but Python {'.'.join(map(str, recommended))} is recommended"
            )
            warning = True
    else:
        if current_version < min_version:
            message = (
                f"Python {'.'.join(map(str, current_version))} is too old for {package}. "
                f"Minimum required version is Python {'.'.join(map(str, min_version))}"
            )
            warning = True
        elif current_version > max_version:
            message = (
                f"Python {'.'.join(map(str, current_version))} is too new for {package}. "
                f"Maximum supported version is Python {'.'.join(map(str, max_version))}"
            )
            warning = True
        else:
            message = f"Unexpected version compatibility issue with {package}"
            warning = True
    
    return {
        "compatible": is_compatible,
        "recommended": is_recommended,
        "message": message,
        "warning": warning,
        "current_version": current_version,
        "min_version": min_version,
        "max_version": max_version,
        "recommended_version": recommended
    }

def check_all_package_compatibilities() -> Dict[str, Dict[str, Union[bool, str]]]:
    """
    Check compatibility with all packages in the VERSION_COMPATIBILITY dict.
    
    Returns:
        dict: Compatibility information for all packages
    """
    results = {}
    for package in VERSION_COMPATIBILITY:
        results[package] = check_python_version_compatibility(package)
    return results

def warn_if_incompatible(package: str) -> bool:
    """
    Check if the current Python version is compatible with the package and issue a warning if not.
    
    Args:
        package: Name of the package to check
        
    Returns:
        bool: True if compatible, False otherwise
    """
    result = check_python_version_compatibility(package)
    
    if not result["compatible"]:
        warnings.warn(
            f"\n⚠️ WARNING: {result['message']} ⚠️\n"
            f"This may cause errors or unexpected behavior.",
            RuntimeWarning, stacklevel=2
        )
        return False
    
    if result["warning"]:
        warnings.warn(
            f"\n⚠️ {result['message']} ⚠️",
            RuntimeWarning, stacklevel=2
        )
    
    return result["compatible"]

def is_module_installed(module_name: str) -> bool:
    """
    Check if a module is installed.
    
    Args:
        module_name: Name of the module to check
        
    Returns:
        bool: True if installed, False otherwise
    """
    return importlib.util.find_spec(module_name) is not None

def check_ray_compatibility() -> Dict[str, Union[bool, str]]:
    """
    Check compatibility specifically for Ray.
    Also checks if Ray is installed.
    
    Returns:
        dict: Compatibility information for Ray
    """
    # First check if Ray is installed
    is_installed = is_module_installed("ray")
    
    # Then check version compatibility
    compat_info = check_python_version_compatibility("ray")
    compat_info["is_installed"] = is_installed
    
    return compat_info

def fix_ray_import_issues():
    """
    Apply fixes for Ray import issues with Python 3.12.
    This includes modifying sys.path and applying patches if necessary.
    
    This is a no-op if Ray is not installed or Python version is compatible.
    """
    # Check if Ray is installed and there's a version compatibility issue
    ray_info = check_ray_compatibility()
    
    if not ray_info["is_installed"]:
        # Ray is not installed, nothing to fix
        return
    
    if ray_info["compatible"]:
        # No compatibility issue, nothing to fix
        return
    
    current_version = get_python_version()
    
    # Only apply fixes for Python 3.12
    if current_version[0] != 3 or current_version[1] != 12:
        return
    
    # At this point, we know Ray is installed and we're running Python 3.12
    # Apply necessary fixes
    
    import os
    
    # 1. Set environment variable to try to work around issues
    os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
    
    # 2. If using Ray with Lightning, try to work around their compatibility issues
    if is_module_installed("lightning"):
        # Set environment variables that might help with Lightning + Ray issues
        os.environ["PL_DISABLE_FORK"] = "1"
        os.environ["PYTHONPATH"] = f"{os.getcwd()}:{os.environ.get('PYTHONPATH', '')}"

def create_compatibility_env_file() -> Path:
    """
    Create a .env file with environment variables to help with compatibility issues.
    
    Returns:
        Path: Path to the created .env file
    """
    env_file = Path(".env.compatibility")
    
    current_version = get_python_version()
    
    # Determine which compatibility settings to include
    settings = []
    
    # Python 3.12 specific settings
    if current_version[0] == 3 and current_version[1] == 12:
        settings.extend([
            "# Python 3.12 compatibility settings",
            "RAY_DISABLE_IMPORT_WARNING=1",
            "PL_DISABLE_FORK=1",
            "",
            "# Adjust Python path to help with imports",
            "PYTHONPATH=${PWD}:${PYTHONPATH}",
            ""
        ])
    
    # Write settings to file
    with open(env_file, "w") as f:
        f.write("\n".join([
            "# LoRA Training Pipeline Compatibility Settings",
            "# Created by python_version_check.py",
            f"# Python version: {'.'.join(map(str, current_version))}",
            "",
            *settings,
            "# End of compatibility settings"
        ]))
    
    return env_file

def get_compatibility_instructions() -> str:
    """
    Get instructions for fixing compatibility issues based on the current Python version.
    
    Returns:
        str: Markdown-formatted instructions
    """
    current_version = get_python_version()
    results = check_all_package_compatibilities()
    
    # Check if there are any compatibility issues
    has_issues = any(not result["compatible"] for result in results.values())
    has_warnings = any(result["warning"] for result in results.values())
    
    if not has_issues and not has_warnings:
        return "✅ Your Python version is compatible with all required packages."
    
    # Start building instructions
    instructions = []
    
    # Python 3.12 specific instructions
    if current_version[0] == 3 and current_version[1] == 12:
        incompatible_packages = [pkg for pkg, result in results.items() if not result["compatible"]]
        
        if incompatible_packages:
            instructions.append("## Python 3.12 Compatibility Issues\n")
            instructions.append("Your Python version (3.12) is not fully compatible with the following packages:")
            for pkg in incompatible_packages:
                instructions.append(f"- {pkg}")
            
            instructions.append("\n### Option 1: Use a compatible Python version\n")
            instructions.append("The recommended solution is to use Python 3.10 or 3.11:")
            instructions.append("```bash")
            instructions.append("# Using conda")
            instructions.append("conda create -n lora-py311 python=3.11")
            instructions.append("conda activate lora-py311")
            instructions.append("")
            instructions.append("# Or using pyenv")
            instructions.append("pyenv install 3.11.7")
            instructions.append("pyenv local 3.11.7")
            instructions.append("```\n")
            
            instructions.append("### Option 2: Use compatibility mode\n")
            instructions.append("If you must use Python 3.12, you can try these workarounds:")
            instructions.append("```bash")
            instructions.append("# Source the compatibility environment variables")
            instructions.append("source .env.compatibility")
            instructions.append("")
            instructions.append("# Then run your command")
            instructions.append("python run_pipeline.py")
            instructions.append("```\n")
            
            instructions.append("### Option 3: Use Ray executors instead of Lightning+Ray\n")
            instructions.append("As a last resort, you can modify the code to use Ray executors directly without Lightning.")
            
        else:
            instructions.append("## Python 3.12 Compatibility Notes\n")
            instructions.append("Your Python version (3.12) is still new and might have some compatibility issues with some libraries.")
            instructions.append("Although our initial checks show no critical issues, you might encounter unexpected errors.")
            instructions.append("If you do, consider switching to Python 3.10 or 3.11 which are more thoroughly tested with these libraries.")
    
    # Return the instructions as a string
    return "\n".join(instructions)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Python Version Compatibility Checker")
    parser.add_argument("--check", action="store_true", help="Check compatibility with all packages")
    parser.add_argument("--fix", action="store_true", help="Apply compatibility fixes if possible")
    parser.add_argument("--env", action="store_true", help="Create compatibility .env file")
    parser.add_argument("--package", help="Check compatibility with a specific package")
    parser.add_argument("--instructions", action="store_true", help="Print compatibility instructions")
    
    args = parser.parse_args()
    
    if args.check or not any([args.fix, args.env, args.package, args.instructions]):
        # Default action is to check all packages
        results = check_all_package_compatibilities()
        
        print("\nPython Version Compatibility Check")
        print("=================================")
        current_version = get_python_version()
        print(f"Current Python version: {'.'.join(map(str, current_version))}")
        print("")
        
        all_compatible = True
        for package, result in results.items():
            status = "✅ Compatible" if result["compatible"] else "❌ Not compatible"
            if result["warning"]:
                status = "⚠️ Compatible but not recommended"
                
            print(f"{package}: {status}")
            print(f"  {result['message']}")
            print("")
            
            if not result["compatible"]:
                all_compatible = False
        
        if all_compatible:
            print("All packages are compatible with your Python version.")
        else:
            print("⚠️ Some packages are not compatible with your Python version.")
            print("Run with --instructions to see recommendations for fixing these issues.")
    
    if args.fix:
        fix_ray_import_issues()
        print("Applied compatibility fixes for the current session.")
    
    if args.env:
        env_file = create_compatibility_env_file()
        print(f"Created compatibility environment file: {env_file}")
        print("To use it, run: source .env.compatibility")
    
    if args.package:
        result = check_python_version_compatibility(args.package)
        
        print(f"\nPython Compatibility with {args.package}")
        print("=" * (25 + len(args.package)))
        print(f"Status: {'✅ Compatible' if result['compatible'] else '❌ Not compatible'}")
        print(f"Message: {result['message']}")
        
        if not result["compatible"]:
            print("\nRecommendation:")
            print(f"Use Python {'.'.join(map(str, result['recommended_version']))}")
    
    if args.instructions:
        instructions = get_compatibility_instructions()
        print("\nCompatibility Instructions")
        print("=========================")
        print(instructions)