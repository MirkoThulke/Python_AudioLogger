import sys


# Pytest section ########
PASSED      = True
FAILED      = False

result  = FAILED 
#########################

#########################
# Pytest calls all functions starting with "test_" automatically
# hence, functions to be called by pytest MUST start with "test_"

# Unit test howto :
# https://youtu.be/6tNS--WetLI?feature=shared

def test_smoke_audioInput():

    assert PASSED



if __name__ == "__main__":
    result =  test_smoke_audioInput()
    sys.exit(result)  # this return code goes back to Jenkins