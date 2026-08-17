import numpy as np

h = np.load("/home/ubuntu/bev/predicter_head.npz")
W0, b0, W2, b2 = h["0.weight"], h["0.bias"], h["2.weight"], h["2.bias"]   # (512,256)(512)(288,512)(288)
VRU = {2, 6, 7}


def occ_from_convonly(c):                       # c: (1,256,200,200) NCHW
    x = c[0].transpose(2, 1, 0)                 # permute(0,3,2,1): (C,H,W)->(W,H,C) = (200,200,256)
    x = x.reshape(-1, 256)
    x = x @ W0.T + b0                           # Linear 256->512
    x = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)   # softplus, stable
    x = x @ W2.T + b2                           # Linear 512->288
    x = x.reshape(200, 200, 16, 18)
    return x.argmax(-1)                         # (200,200,16)


walker = np.load("/home/ubuntu/bev/bev_onboard_convonly.npy").astype(np.float32)
gold = np.load("/home/ubuntu/bev/bev_clampref_fp32.npy").astype(np.float32)

ow = occ_from_convonly(walker).ravel()
og = occ_from_convonly(gold).ravel()

overall = float((ow == og).mean())
vm = np.isin(og, list(VRU))
geom_ref = og != 17
vru = float((ow[vm] == og[vm]).mean()) if vm.sum() else float("nan")
# geometry: of voxels the FP ref calls occupied (!=17 empty), how many does on-board also call occupied
geom = float((ow[geom_ref] != 17).mean()) if geom_ref.sum() else float("nan")

print("=== ON-BOARD occupancy (walker) vs FP32 reference occupancy ===")
print(f"overall voxel agreement : {overall:.4f}")
print(f"VRU-voxel agreement     : {vru:.4f}   (ref VRU voxels: {int(vm.sum())})")
print(f"geometry (occupied kept): {geom:.4f}   (ref occupied voxels: {int(geom_ref.sum())})")
print(f"ref class histogram top : empty(17)={int((og==17).sum())}  occupied={int(geom_ref.sum())}")
