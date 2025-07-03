import json
import sys
import subprocess



PASS        = False
FAILED      = True


error_flag = PASS  # No fault


def check_python_version():

    version_info = sys.version_info
    
    print(f"Detected Python version: {version_info.major}.{version_info.minor}.{version_info.micro}")
    
    return PASS



def save_outdated_packages_to_json():

    return PASS

        
        
if __name__ == "__main__":
    
    error_flag      = check_python_version() or save_outdated_packages_to_json()

    sys.exit(error_flag)  # this return code goes back to Jenkins
    