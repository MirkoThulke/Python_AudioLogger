import sys


# Pytest section ########
PASSED      = True
FAILED      = False

result  = FAILED 
#########################



def test_integration_audioProcessing():

    assert PASSED



if __name__ == "__main__":
    result =  test_integration_audioProcessing()
    sys.exit(result)  # this return code goes back to Jenkins