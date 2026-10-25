"""
gsplat rendering helpers for decoded .lcc splats.

One Renderer wraps both render paths: pinhole for training views and fisheye
(OPENCV_FISHEYE) for wide-FOV real-GT localization.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Intrinsics:
    w: int
    h: int
    fx: float
    fy: float
    cx: float
    cy: float
    k: tuple = ()
    model: str = "pinhole"

    @classmethod
    def pinhole(cls, w, h, f=None):
        f = f if f is not None else 0.7 * w
        return cls(w, h, f, f, w / 2, h / 2, (), "pinhole")

    @classmethod
    def from_dict(cls, d):
        return cls(int(d["w"]), int(d["h"]), d["fx"], d["fy"], d["cx"], d["cy"],
                   tuple(d.get("k", ())), d.get("model", "fisheye" if d.get("k") else "pinhole"))

    def scaled(self, sc: float) -> "Intrinsics":
        return Intrinsics(int(self.w * sc), int(self.h * sc), self.fx * sc, self.fy * sc,
                          self.cx * sc, self.cy * sc, self.k, self.model)

    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], float)


class Renderer:
    """Holds splats on the GPU once; renders many views cheaply."""

    def __init__(self, splats, device="cuda", near=0.05, far=60.0):
        import torch
        self.torch = torch
        self.device = device
        self.near, self.far = near, far
        self.M = torch.tensor(splats.means, device=device)
        self.S = torch.tensor(splats.scales, device=device)
        self.C = torch.tensor(splats.color, device=device)
        self.O = torch.tensor(splats.opacity, device=device)
        self.Q = torch.tensor(splats.quat, device=device, dtype=torch.float32)
        self.Q = self.Q / self.Q.norm(dim=1, keepdim=True)

    def render(self, w2c: np.ndarray, intr: Intrinsics, depth: bool = True):
        """Render one view. Returns (rgb_uint8 HxWx3, depth_float32 HxW | None).

        Set intr.model='fisheye' with k1..k4 for wide-FOV real-GT localization;
        otherwise a standard pinhole render.
        """
        import torch
        from gsplat import rasterization
        K = torch.tensor(intr.K(), device=self.device, dtype=torch.float32)[None]
        vm = torch.tensor(w2c, device=self.device, dtype=torch.float32)[None]
        mode = "RGB+ED" if depth else "RGB"
        kw = dict(near_plane=self.near, far_plane=self.far)
        if intr.model == "fisheye":
            kw.update(camera_model="fisheye",
                      radial_coeffs=torch.tensor([list(intr.k)], device=self.device, dtype=torch.float32),
                      with_ut=True, packed=False)
        out, _, _ = rasterization(self.M, self.Q, self.S, self.O, self.C, vm, K,
                                  intr.w, intr.h, render_mode=mode, **kw)
        rgb = (np.clip(out[0, ..., :3].detach().cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        d = out[0, ..., 3].detach().cpu().numpy().astype(np.float32) if depth else None
        return rgb, d
