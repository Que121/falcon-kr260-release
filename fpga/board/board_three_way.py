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

board = np.load("/home/ubuntu/bev/bev_onboard_convonly.npy").astype(np.float32)
sim   = np.load("/home/ubuntu/bev/bev_sim_int8_convonly.npy").astype(np.float32)
fp32  = np.load("/home/ubuntu/bev/bev_clampref_fp32.npy").astype(np.float32)

ob, os_, of = occ(board), occ(sim), occ(fp32)

def agree(a, b, tag):
    overall = float((a == b).mean())
    vm = np.isin(b, list(VRU)); gr = b != 17
    vru = float((a[vm] == b[vm]).mean()) if vm.sum() else float("nan")
    geom = float((a[gr] != 17).mean()) if gr.sum() else float("nan")
    print(f"  {tag:26s} overall {overall:.4f} | VRU {vru:.4f} | geom {geom:.4f}")

print("=== conv_only feature cosine ===")
print(f"  board vs sim-INT8 : {cos(board, sim):.4f}   <- on-board deployment fidelity")
print(f"  board vs FP32     : {cos(board, fp32):.4f}")
print(f"  sim-INT8 vs FP32  : {cos(sim, fp32):.4f}")
print("=== occupancy voxel-argmax agreement ===")
agree(ob, os_, "board vs sim-INT8")          # how faithfully the board runs the deployed model
agree(ob, of,  "board vs FP32")              # on-board retention vs FP32
agree(os_, of, "sim-INT8 vs FP32")           # simulator retention vs FP32 (same frame)
