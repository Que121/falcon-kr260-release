import numpy as np, sys
d = np.load(sys.argv[1])
for k in d.files:
    print(k, d[k].shape, d[k].dtype)
