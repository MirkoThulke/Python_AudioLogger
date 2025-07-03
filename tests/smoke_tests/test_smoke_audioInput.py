import sys


PASS        = False
FAILED      = True


error_flag = PASS  # No fault



def test_smoke_audioInput():

    return PASS



if __name__ == "__main__":
    error_flag =  test_smoke_audioInput()
    sys.exit(error_flag)  # this return code goes back to Jenkins