import ctypes as ct
import numpy as np
import pandas as pd

#the build file location
libfile = r"build/lib.linux-x86_64-3.8/logistic.cpython-38-x86_64-linux-gnu.so"

#loading it for use
our_lib = ct.CDLL(libfile)

#setting the return types for our C++ methods
our_lib.fit.argtypes = [ct.c_void_p]
our_lib.predict.argtypes = [ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.float64)]
our_lib.predict.restype = ct.c_double

#initializing the class
tree_obj = our_lib.LogisticRegression()

#the array to test the model
test_features = np.array((62.0,1.0,1.0,128.0,208.0,1.0,0.0,140.0,0.0,0.0,2.0,0.0,2.0))
test_features = test_features.astype(np.double)

#calling the fit method
predictions = our_lib.fit(tree_obj)

#predictiing
pred = our_lib.predict(tree_obj,test_features)
print("Predicted value:",pred)
