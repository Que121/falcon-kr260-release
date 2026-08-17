import numpy as np
h = np.load("/home/ubuntu/bev/predicter_head.npz")
W0, b0, W2, b2 = h["0.weight"], h["0.bias"], h["2.weight"], h["2.bias"]
VRU = {2, 6, 7}
def occ(c):
    x = c[0].transpose(2, 1, 0).reshape(-1, 256)
    x = x @ W0.T + b0
    x = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
    x = (x @ W2.T + b2).reshape(200, 200, 16, 18)
    return x.argmax(-1).ravel()
def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
allpl = np.load("/home/ubuntu/bev_allpl/allpl_convonly.npy").astype(np.float32)
sim = np.load("/home/ubuntu/bev_allpl/bev_reluc_sim_convonly.npy").astype(np.float32)
fp = np.load("/home/ubuntu/bev/bev_clampref_fp32.npy").astype(np.float32)
oa, os_, of = occ(allpl), occ(sim), occ(fp)
def agree(a, b, t):
    vm = np.isin(b, list(VRU)); gr = b != 17
    print(f"  {t:26s} overall {(a==b).mean():.4f} | VRU {(a[vm]==b[vm]).mean():.4f} | geom {(a[gr]!=17).mean():.4f}")
print("=== conv_only cosine ===")
print(f"  all-PL board vs reluc-sim : {cos(allpl, sim):.4f}   <- all-PL deployment fidelity")
print(f"  all-PL board vs FP32      : {cos(allpl, fp):.4f}")
print("=== occupancy voxel agreement ===")
agree(oa, os_, "all-PL board vs reluc-sim")
agree(oa, of, "all-PL board vs FP32")
agree(os_, of, "reluc-sim vs FP32")
