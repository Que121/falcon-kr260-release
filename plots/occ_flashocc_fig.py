#!/usr/bin/env python3
"""Render the deployed FlashOcc model's 3D occupancy predictions (BEV top-down + 3D) for the paper.

Reads the dumped FlashOcc outputs (flashocc_io/frame_*.npz, key 'occ_out' = (1,200,200,16,18) logits),
argmax -> (200,200,16) class grid, and renders 3 diverse VRU-containing scenes in the canonical nuScenes
Occ3D colormap, with vulnerable-road-user voxels highlighted. -> docs/figs/flashocc_occ.{png,pdf}
"""
import sys, glob, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa

# Canonical nuScenes Occ3D colormap (CVPR'23). 0=others .. 16=vegetation, 17=free.
OCC_COLORS = np.array([
    [0,0,0],[255,120,50],[255,192,203],[255,255,0],[0,150,245],[0,255,255],[255,127,0],
    [255,0,0],[255,240,150],[135,60,0],[160,32,240],[255,0,255],[139,137,137],[75,0,75],
    [150,240,80],[230,230,250],[0,175,0],[255,255,255]], np.float32)/255.0
CLASS_NAMES = ['others','barrier','bicycle','bus','car','constr.veh','motorcycle','pedestrian',
               'traffic_cone','trailer','truck','drive_surf','other_flat','sidewalk','terrain',
               'manmade','vegetation','free']
VRU = {2,6,7}  # bicycle, motorcycle, pedestrian
IODIR = sys.argv[1] if len(sys.argv)>1 else "/home/ANON/flashocc_io"
OUT   = sys.argv[2] if len(sys.argv)>2 else "docs/figs/flashocc_occ"

def grid_of(f): return np.load(f)["occ_out"][0].argmax(-1).astype(np.int64)
def bevtop(g, zfree=17):
    H,W,Z=g.shape; top=np.full((H,W),-1,np.int64)
    for z in range(Z):
        m=(g[:,:,z]!=zfree)&(g[:,:,z]!=0); top[m]=g[:,:,z][m]
    img=np.ones((H,W,3),np.float32); mm=top>=0; img[mm]=OCC_COLORS[top[mm]]
    return img, top

frames=sorted(glob.glob(f"{IODIR}/frame_*.npz"))
cand=sorted(((int(np.isin(bevtop(grid_of(f))[1],list(VRU)).sum()), f) for f in frames), reverse=True)
pick=[]; idxs=[]
for nv,f in cand:
    i=int(f.split('_')[-1].split('.')[0])
    if all(abs(i-j)>=8 for j in idxs): pick.append((f,nv)); idxs.append(i)
    if len(pick)==3: break

fig=plt.figure(figsize=(15,4.6))
for k,(f,nv) in enumerate(pick):
    g=grid_of(f); img,top=bevtop(g); ax=fig.add_subplot(1,4,k+1)
    ax.imshow(np.transpose(img,(1,0,2)),origin="lower")
    ys,xs=np.nonzero(np.isin(top,list(VRU)).T)
    ax.scatter(xs,ys,s=26,facecolors="none",edgecolors="red",linewidths=1.1)
    ax.set_title(f"frame {f.split('_')[-1].split('.')[0]}  (BEV-VRU: {nv})",fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([])
g=grid_of(pick[0][0]); xs,ys,zs=np.nonzero((g!=17)&(g!=0)); cl=g[xs,ys,zs]
ax3=fig.add_subplot(1,4,4,projection="3d"); v=np.isin(cl,list(VRU))
ax3.scatter(xs[~v],ys[~v],zs[~v],c=OCC_COLORS[cl[~v]],s=2,marker="s",alpha=.55,depthshade=False)
ax3.scatter(xs[v],ys[v],zs[v],c="red",s=22,marker="s",depthshade=False)
ax3.set_title(f"frame {pick[0][0].split('_')[-1].split('.')[0]} — 3D (VRU red)",fontsize=9.5)
ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([]); ax3.view_init(elev=28,azim=-60); ax3.set_box_aspect((1,1,0.35))
present=sorted(set(np.unique(np.concatenate([grid_of(f).ravel() for f,_ in pick]))) - {0,17})
fig.legend(handles=[Patch(facecolor=OCC_COLORS[i],edgecolor='gray',label=CLASS_NAMES[i]) for i in present],
           loc="lower center",ncol=min(9,len(present)),fontsize=8.5,frameon=False,bbox_to_anchor=(0.5,-0.03))
fig.suptitle("FlashOcc 3D occupancy prediction on Occ3D-nuScenes (the deployed model) — BEV top-down + 3D; VRU voxels in red",fontsize=12,y=1.01)
plt.tight_layout()
plt.savefig(OUT+".png",dpi=150,bbox_inches="tight"); plt.savefig(OUT+".pdf",bbox_inches="tight")
print("picked frames:", idxs)
