"""All-PL recipe test: plain-ReLU graph (DPU-native, de-fragmented) + the clamp model's small
fix_points (so int8 saturation realizes the clamp). Transfers clamp fps onto the ReLU config by order."""
import os, sys, json, shutil, numpy as np, torch
sys.path.insert(0, "/work")
from bev_stage import BEVStage
from pytorch_nndct.apis import torch_quantizer

src, dst = "/work/frag_relu", "/work/frag_reluc"
shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst)

clamp = json.load(open("/work/quantize_result_clamp/quant_info.json"))
relu = json.load(open(dst + "/quant_info.json"))
moved = {}
for sect in ("output", "input", "param"):
    cvals = list(clamp.get(sect, {}).values())
    rkeys = list(relu.get(sect, {}).keys())
    n = 0
    for i, k in enumerate(rkeys):
        if i < len(cvals):
            relu[sect][k] = cvals[i]; n += 1
    moved[sect] = (n, len(rkeys), len(cvals))
json.dump(relu, open(dst + "/quant_info.json", "w"))
print("fps moved (set,relu_keys,clamp_vals):", moved)

sd = torch.load("/work/bev_stage_sd.pth", map_location="cpu")
def make():
    m = BEVStage(conv_only=True, clamp_c=None)
    m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("predicter.")}, strict=False)
    return m.eval()

q = torch_quantizer("test", make(), (torch.randn(1, 64, 200, 200),), output_dir=dst)
qm = q.quant_model
x = torch.from_numpy(np.load("/work/bev_test_input.npy").astype("float32"))
with torch.no_grad():
    conv = qm(x).numpy()
q.export_xmodel(deploy_check=False, output_dir=dst)

pred = BEVStage(conv_only=False); pred.load_state_dict(sd); pred = pred.predicter.eval()
def occ(c):
    t = torch.from_numpy(c.astype("float32")).permute(0, 3, 2, 1)
    with torch.no_grad():
        o = pred(t)
    return o.view(1, 200, 200, 16, 18).argmax(-1).numpy().ravel()
fp = np.load("/work/bev_clampref_fp32.npy").astype("float32")
ow, of = occ(conv), occ(fp); VRU = {2, 6, 7}
vm = np.isin(of, list(VRU)); gr = of != 17
print(f"reluc(forced-fp): conv max {conv.max():.1f} | overall {(ow==of).mean():.4f} "
      f"VRU {(ow[vm]==of[vm]).mean():.4f} geom {(ow[gr]!=17).mean():.4f}")
