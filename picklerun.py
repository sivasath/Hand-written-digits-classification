import numpy as np
import pickle
a=pickle.load(open("diabetes.pickle","rb"))
np.set_printoptions(threshold=np.inf)
print a