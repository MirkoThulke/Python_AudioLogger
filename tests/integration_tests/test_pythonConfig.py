import sys
import os
import platform
import subprocess


# Pytest section ########
PASSED      = True
FAILED      = False

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
        
def run_pip_check():
    
    try:
        result = subprocess.run(["pip", "check"], capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        assert result.returncode != 0
        
    except Exception as e:
        assert False, f"pip check failed: {e}"

def check_outdated():
    try:
        result = subprocess.run(
            ["pip", "list", "--outdated"],
            capture_output=True,
            text=True,
            check=False
        )
        
        output = result.stdout.strip()
        print(output)

        # Save output to a file (optional)
        with open("outdated-packages.txt", "w") as f:
            f.write(output + "\n")

        # If there's any outdated package listed (non-header lines), return 1
        lines = output.splitlines()
        if len(lines) > 2:  # Header is 2 lines
            assert False, f"Outdated packages found.: {lines}"
        else:
            assert True, f"No outdated packages found."
            
    except Exception as e:
        assert False, f"Error during pip list: {e}"

def test_export_installed_packages():
    with open('requirements_new.txt', 'w') as f:
        subprocess.run(['pip', 'freeze'], stdout=f)      

def load_requirementsFiles(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return set(line.strip() for line in lines if line.strip() and not line.startswith('#'))


def compare_requirementsFiles(file1, file2):
    reqs1 = load_requirementsFiles(file1)
    reqs2 = load_requirementsFiles(file2)

    only_in_1 = reqs1 - reqs2
    only_in_2 = reqs2 - reqs1

    if not only_in_1 and not only_in_2:
        print("The requirements files are functionally identical.")
    else:
        if only_in_1:
            print(f"Packages only in {file1}:")
            for pkg in only_in_1:
                print(f"  {pkg}")
        if only_in_2:
            print(f"Packages only in {file2}:")
            for pkg in only_in_2:
                print(f"  {pkg}")  
        assert False, f"Missing packages."   



def test_check_required_packages():
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
    
    test_python_version()
    test_pip_installed()
    run_pip_check()
    check_outdated()
    test_export_installed_packages()
    compare_requirementsFiles("requirements.txt","requirements_new.txt")
    test_check_required_packages()