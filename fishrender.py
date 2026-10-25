"""Full-FOV Kannala-Brandt fisheye rendering of a splat via a cube-map composite.

gsplat's fisheye camera model culls Gaussians at incidence >= 90 deg, so wide fisheye
images render with empty periphery. Here each fisheye pixel ray is sampled from one of
5 pinhole faces (front/left/right/up/down, ~100 deg FOV each).

Outputs: rgb uint8 HxWx3, ray distance (m, valid everywhere), incidence (deg), z-depth (< 90 deg only).
"""
import numpy as np, cv2
from render import Intrinsics


def kb_rays(it):
    """unit camera-frame ray per pixel (OpenCV camera: x right, y down, z forward) + incidence (deg)."""
    u, v = np.meshgrid(np.arange(it.w) + 0.5, np.arange(it.h) + 0.5)
    xd = (u - it.cx) / it.fx; yd = (v - it.cy) / it.fy; thd = np.hypot(xd, yd); k = it.k
    th = thd.copy()
    for _ in range(60):
        f = th * (1 + k[0] * th**2 + k[1] * th**4 + k[2] * th**6 + k[3] * th**8) - thd
        df = 1 + 3 * k[0] * th**2 + 5 * k[1] * th**4 + 7 * k[2] * th**6 + 9 * k[3] * th**8
        th = th - f / df
    phi = np.arctan2(yd, xd); s = np.sin(th)
    d = np.stack([s * np.cos(phi), s * np.sin(phi), np.cos(th)], -1)
    return d.astype(np.float64), np.degrees(th)


def _rot(axis):
    """R maps camera-frame vectors into the frame of a face looking along `axis` (z_face = axis)."""
    z = np.array(axis, float); up = np.array([0, -1, 0.]) if abs(z[1]) < 0.9 else np.array([0, 0, 1.])
    x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x)
    return np.stack([x, y, z])


FACES = [(0, 0, 1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]


class FishRenderer:
    def __init__(self, rend, it, face=768, fov_deg=100):
        self.r, self.it = rend, it
        self.d, self.inc = kb_rays(it)
        self.S = face; self.f = (face / 2) / np.tan(np.radians(fov_deg / 2))
        self.pin = Intrinsics(face, face, self.f, self.f, face / 2, face / 2, (), "pinhole")
        self.R = [_rot(a) for a in FACES]
        dots = np.stack([self.d @ R[2] for R in self.R], -1)
        self.face = np.argmax(dots, -1)
        self.maps = []
        for i, R in enumerate(self.R):
            p = self.d @ R.T
            pz = np.maximum(p[..., 2], 1e-6)
            mx = (self.f * p[..., 0] / pz + face / 2 - 0.5).astype(np.float32)
            my = (self.f * p[..., 1] / pz + face / 2 - 0.5).astype(np.float32)
            self.maps.append((mx, my, pz))

    def render(self, w2c):
        H, W = self.it.h, self.it.w
        rgb = np.zeros((H, W, 3), np.uint8); ray = np.zeros((H, W), np.float32)
        for i, R in enumerate(self.R):
            m = self.face == i
            if not m.any(): continue
            T = np.eye(4); T[:3, :3] = R
            fr, fd = self.r.render(T @ w2c, self.pin)
            mx, my, pz = self.maps[i]
            c = cv2.remap(fr, mx, my, cv2.INTER_LINEAR); z = cv2.remap(fd, mx, my, cv2.INTER_NEAREST)
            rgb[m] = c[m]; ray[m] = (z / pz)[m]
        ray[ray < 0.05] = 0
        zdep = np.where(self.inc < 89.5, ray * np.cos(np.radians(self.inc)), 0).astype(np.float32)
        return rgb, ray, zdep
