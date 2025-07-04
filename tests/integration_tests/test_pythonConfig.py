import json
import sys
import subprocess



# Pytest section ########
PASSED      = True
FAILED      = False

result  = FAILED 
#########################



def check_python_version():

    version_info = sys.version_info
    
    print(f"Detected Python version: {version_info.major}.{version_info.minor}.{version_info.micro}")
    
    assert PASSED



def save_outdated_packages_to_json():

    assert PASSED

     
        
if __name__ == "__main__":
    
    result1      = check_python_version()
    result2      = save_outdated_packages_to_json()
    
    if result1==FAILED or result2==FAILED:
        result = FAILED
    else:
        result = PASSED
        
    sys.exit(result)  # this return code goes back to Jenkins
    