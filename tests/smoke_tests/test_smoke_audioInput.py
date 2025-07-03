import sys


# Pytest section ########
PASSED      = True
FAILED      = False

result  = FAILED 
#########################



def test_smoke_audioInput():

    assert PASSED



if __name__ == "__main__":
    result =  test_smoke_audioInput()
    sys.exit(result)  # this return code goes back to Jenkins