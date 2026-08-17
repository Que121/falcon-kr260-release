import numpy as np,sys
d=np.load(sys.argv[1])
np.save(sys.argv[2], d[sys.argv[3]].astype(np.float32))
print("saved",sys.argv[2], d[sys.argv[3]].shape)
