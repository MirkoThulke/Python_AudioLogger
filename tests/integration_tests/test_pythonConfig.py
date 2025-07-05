import sys
import os
import platform
import subprocess


# Pytest section ########
PASSED      = True
FAILED      = False

result  = FAILED 
#########################


def test_python_version():
    """Ensure Python version is at least 3.8"""
    assert sys.version_info >= (3, 8), f"Python version too low: {platform.python_version()}"


def test_pip_installed():
    """Ensure pip is installed and working"""
    try:
        output = subprocess.check_output([sys.executable, "-m", "pip", "--version"])
        assert b"pip" in output
    except Exception as e:
        assert False, f"pip check failed: {e}"


def test_required_packages():
    """Ensure required packages are installed"""
    required = {"requests", "pytest"}
    try:
        output = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
        installed = set([line.split("==")[0].lower() for line in output.decode().splitlines()])
        missing = required - installed
        assert not missing, f"Missing packages: {missing}"
    except Exception as e:
        assert False, f"Package check failed: {e}"


if __name__ == "__main__":
    
    result1      = check_python_version()
    result2      = save_outdated_packages_to_json()
    
    if result1 == FAILED :
        result = FAILED
    elif result2 == FAILED :
        result = FAILED
    else :
        result = PASSED 

    sys.exit(result)  # this return code goes back to Jenkins
    