
import os
import time

from load_dataset import load_test_data, NUM_CATEGORIES, IMG_WIDTH, IMG_HEIGHT
from model import numpy_to_tensor, tensor_to_numpy
os.add_dll_directory(r"C:\Users\HP\Vestacka Projekat\build")

import sys
sys.path.append(r"C:\Users\HP\Vestacka Projekat\build")
import cpp
import numpy as np


x=np.array([-3,-1,0,2,5],dtype=np.float32)

t=numpy_to_tensor(x)

r=cpp.ReLU()

print(tensor_to_numpy(r.forward(t)))