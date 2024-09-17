import os
import numpy as np

def ADV1_schedule():
    fn = os.path.join(os.path.dirname(__file__), 'ADV1.csv')
    return np.loadtxt(fn).T

def ADV2_schedule():
    raise NotImplementedError

def model_schedule():
    return ADV1_schedule()
