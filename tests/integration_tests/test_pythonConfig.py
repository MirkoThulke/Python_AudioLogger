import sys
import platform
import subprocess
import os
import pipreqs

is_unix = os.getenv('IS_UNIX') == 'true'

# Pytest section ########
PASSED      = True
FAILED      = False

# For requirement.txt file check
if is_unix :
    file1 = "requirements_linux.txt"
    file2 = "requirements_linux_new.txt"
else :
    file1 = "requirements_windows.txt"
    file2 = "requirements_windows_new.txt"
    
#########################
# Pytest calls all functions starting with "test_" automatically
# hence, functions to be called by pytest MUST start with "test_"

# Unit test howto :
# https://youtu.be/6tNS--WetLI?feature=shared


def update_package(package):
    try:
        # Your existing update logic here, e.g.
        subprocess.run(['pip', 'install', '--upgrade', package], check=True)

    except Exception as e:
        assert False, f"Error during package update: {package}    {e}"

def update_outdated_packages():
    try:
        # Get list of outdated packages
        result = subprocess.run(
            ['pip', 'list', '--outdated'],
            capture_output=True,
            text=True,
            check=True
        )

        lines = result.stdout.splitlines()
        # Skip header (usually first 2 lines)
        # Example output:
        # Package    Version Latest Type
        # ---------- ------- ------ -----
        # certifi    2025.1.31 2025.6.15 wheel

        if len(lines) < 3:
            print("No outdated packages found.")
            return

        packages = []
        for line in lines[2:]:  # skip headers
            parts = line.split()
            if parts:
                packages.append(parts[0])

        for package in packages:
            print(f"Updating {package} ...")
            update_package(package)

    except Exception as e:
        assert False, f"Error during package update test: {e}"
    
    
    
def test_python_version():
    """Ensure Python version is at least 3.8"""
    print(f"Python version is:  {sys.version_info}\n")
    assert sys.version_info >= (3, 8), f"Python version too low: {platform.python_version()}"


def test_pip_installed():
    """Ensure pip is installed and working"""
    try:
        output = subprocess.check_output([sys.executable, "-m", "pip", "--version"])
        print(f"pip version is:  {output}\n")
        assert b"pip" in output, f"pip version not found.\n"
    except Exception as e:
        assert False, f"pip check failed: {e}"

    
def test_pip_check():
    
    try:
        result = subprocess.run(["pip", "check"], capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        assert result.returncode == 0, "pip check found broken requirements"
        
    except Exception as e:
        assert False, f"pip check failed: {e}"


def test_get_pip_version():
    try:
        result = subprocess.run(
            ['pip', '--version'],
            capture_output=True,
            text=True,
            check=True
            )
        print("pip version output:", result.stdout.strip())
        # Example output: pip 23.0.1 from /path/to/python/site-packages/pip (python 3.10)
        # You can parse version number if needed:
        assert "pip" in result.stdout.lower()
        
    except subprocess.CalledProcessError as e:
        print("Failed to get pip version.")
        print(e.stderr)
        assert False, f"pip check failed: {e}"
    
    


def test_update_outdated():
    try:
        # Step 1: Load packages from requirements file
        with open(file1, "r") as req_file:
            required_packages = {
                line.split("==")[0].strip()
                for line in req_file
                if line.strip() and not line.startswith("#")
            }

        # Step 2: Run pip list --outdated
        result = subprocess.run(
            ["pip", "list", "--outdated", "--format=freeze"],
            capture_output=True,
            text=True,
            check=False
        )

        output = result.stdout.strip()
        print(output)

        # Save to file (optional)
        with open("outdated-packages.txt", "w") as f:
            f.write(output + "\n")

        # Step 3: Filter outdated packages that are in requirements
        outdated_lines = output.splitlines()
        outdated_required = [
            line for line in outdated_lines
            if line.split("==")[0] in required_packages
        ]

        # Step 4: Update if any
        if outdated_required:
            for line in outdated_required:
                package = line.split("==")[0]
                subprocess.run(["pip", "install", "--upgrade", package], check=True)
            assert False, f"Outdated packages found and updated: {outdated_required}"
        else:
            assert True, "No outdated packages found from requirements."

    except Exception as e:
        assert False, f"Error during pip list or upgrade: {e}"


# Export only those packages that are relevant for this application
def test_export_installed_packages():
    try:
        result = subprocess.run(
            ['pipreqs', '--force', '--savepath', file2, '--encoding', 'utf-8'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            assert False, f"pipreqs failed with error: {result.stderr}"

        assert os.path.exists(file2), f"Requirements file not created: {file2}"

    except Exception as e:
        assert False, f"Error during pipreqs: {e}"


def load_requirementsFiles(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            return set(line.strip() for line in lines if line.strip() and not line.startswith('#'))
    except Exception as e:
        assert False, f"Error during loading of requirement files: {e}"


def test_compare_requirementsFiles():

    reqs1 = load_requirementsFiles(file1)
    reqs2 = load_requirementsFiles(file2)

    only_in_1 = reqs1 - reqs2
    only_in_2 = reqs2 - reqs1

    if not only_in_1 and not only_in_2:
        print("The requirements files are functionally identical.")
    else:
        msg = []
        if only_in_1:
            msg.append(f"Packages only in {file1}:")
            msg.extend(f"  {pkg}" for pkg in only_in_1)
        if only_in_2:
            msg.append(f"Packages only in {file2}:")
            msg.extend(f"  {pkg}" for pkg in only_in_2)
        # Print full context
        full_msg = "\n".join(msg)
        assert False, f"Missing or extra packages found:\n{full_msg}"



def test_check_required_packages():
    
    required = {"requests", "pytest", "wxpython", "pyaudio", "scipy"}

    try:
        output = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
        installed = set([line.split("==")[0].lower() for line in output.decode().splitlines()])
        missing = required - installed
        if missing:
            print(f"Missing Packages: {missing}")
        else:
            print("All required packages are installed.")
            
        assert not missing, f"Missing packages: {missing}"
        
    except Exception as e:
        assert False, f"Package check failed: {e}"





if __name__ == "__main__":
    # only used for debugging. Pytest calls all functions starting with "test_" automatically
    test_python_version()
    test_pip_installed()
    test_pip_check()
    test_get_pip_version()
    test_check_required_packages()
    test_update_outdated()
    test_export_installed_packages()
    test_compare_requirementsFiles(file1,file2)