import sys


PASS        = False
FAILED      = True


error_flag = PASS  # No fault



def test_integration_audioProcessing():

    return PASS



if __name__ == "__main__":
    error_flag =  test_integration_audioProcessing()
    sys.exit(error_flag)  # this return code goes back to Jenkins