'''
https://github.com/endolith/waveform-analysis/tree/master/waveform_analysis/weighting_filters
The MIT License (MIT)
Copyright (c) 2016 endolith@gmail.com
'''

def _relative_degree(z, p):
    """
    Return relative degree of transfer function from zeros and poles
    """
    degree = len(p) - len(z)
    if degree < 0:
        raise ValueError("Improper transfer function. "
                         "Must have at least as many poles as zeros.")
    else:
        return degree
