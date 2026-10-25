"""Mask R-CNN (COCO person) masks for the person-in-stall test set (state P) -> align/person_P.npz"""
import glob, os, numpy as np, torch, cv2
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
m = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).cuda().eval()
paths = sorted(glob.glob("scans/evalsets_2026-09-27/aug19_bath3/*_rgb.jpg"))
names, M = [], []
for p in paths:
    im = cv2.imread(p)[:, :, ::-1].copy(); x = torch.tensor(im / 255., dtype=torch.float32, device="cuda").permute(2, 0, 1)
    with torch.no_grad(): o = m([x])[0]
    sel = (o["labels"] == 1) & (o["scores"] > 0.5); mk = np.zeros(im.shape[:2], bool)
    for k in o["masks"][sel]: mk |= k[0].cpu().numpy() > 0.5
    names.append(os.path.basename(p)[:-8]); M.append(mk)
M = np.stack(M); print("frames with person:", int((M.mean((1, 2)) > 0.002).sum()), "of", len(M), "mean cov %.2f%%" % (M.mean() * 100))
np.savez_compressed(os.path.dirname(os.path.abspath(__file__)) + "/align/person_P.npz", names=np.array(names), m=M)
