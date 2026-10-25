"""
Loader for XGRIDS Lixel CyberColor (.lcc) Gaussian-splat scans.

Decodes the 32-byte/splat data.bin; per-scan constants (splat count, LOD split,
scale range) come from the try.lcc manifest.

Byte layout:
    bytes  0-11 : position  float32 x3   (metric)
    bytes 12-14 : color RGB uint8 (/255)
    byte     15 : opacity   uint8 (/255)
    bytes 16-21 : scale     uint16 LE x3 -> smin + u16/65535 * (smax - smin)
    bytes 22-25 : rotation  uint8 x4, order (x,y,z,w) with w = byte 25 -> (b-128)/128
    bytes 26-31 : padding
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
import numpy as np

SPLAT_STRIDE = 32


def _find_lcc_json(root: str) -> str:
    """Return the path to the .lcc manifest inside a scan root (any *.lcc file)."""
    for name in os.listdir(root):
        if name.endswith(".lcc"):
            return os.path.join(root, name)
    raise FileNotFoundError(f"no *.lcc manifest found in {root}")


@dataclass
class Splats:
    """Decoded Gaussian arrays for one LOD block (float32, CPU numpy)."""
    means: np.ndarray
    scales: np.ndarray
    color: np.ndarray
    opacity: np.ndarray
    quat: np.ndarray

    def __len__(self) -> int:
        return self.means.shape[0]

    def crop_box(self, center: np.ndarray, half_extent) -> "Splats":
        """Keep splats within an axis-aligned box; returns a new Splats view."""
        half = np.asarray(half_extent, np.float32)
        keep = (np.abs(self.means - center) < half).all(1)
        return Splats(self.means[keep], self.scales[keep], self.color[keep],
                      self.opacity[keep], self.quat[keep])


@dataclass
class LccScan:
    """A decoded .lcc scan: manifest metadata + lazy access to splats and poses."""
    root: str
    manifest: dict = field(repr=False)

    @classmethod
    def open(cls, root: str) -> "LccScan":
        root = os.path.abspath(root)
        if os.path.exists(os.path.join(root, "point_cloud.ply")):
            return PlyScan(root)
        manifest = json.load(open(_find_lcc_json(root)))
        return cls(root=root, manifest=manifest)

    @property
    def total_splats(self) -> int:
        return int(self.manifest["totalSplats"])

    @property
    def lod_sizes(self) -> list[int]:
        """Splat counts per LOD block; block 0 is the finest."""
        return [int(x) for x in self.manifest["splats"]]

    def _scale_attr(self):
        for a in self.manifest["attributes"]:
            if a["name"] == "scale":
                return np.asarray(a["min"], np.float32), np.asarray(a["max"], np.float32)
        raise KeyError("no 'scale' attribute in .lcc manifest")

    @property
    def bounding_box(self):
        bb = self.manifest["boundingBox"]
        return np.asarray(bb["min"], np.float32), np.asarray(bb["max"], np.float32)

    def decode(self, lod: int = 0) -> Splats:
        """Decode one LOD block (default 0 = finest, the one to render for in-room views)."""
        raw = open(os.path.join(self.root, "data.bin"), "rb").read()
        n_total = len(raw) // SPLAT_STRIDE
        if n_total != self.total_splats:
            print(f"[lcc] warning: data.bin holds {n_total} splats but try.lcc says "
                  f"{self.total_splats}")
        a = np.frombuffer(raw, dtype=np.uint8)[: n_total * SPLAT_STRIDE].reshape(n_total, SPLAT_STRIDE)

        sizes = self.lod_sizes
        start = sum(sizes[:lod])
        end = start + sizes[lod]
        a = a[start:end]

        smin, smax = self._scale_attr()
        means = a[:, 0:12].copy().view(np.float32).reshape(-1, 3)

        def u16(off):
            return (a[:, off].astype(np.uint16) | (a[:, off + 1].astype(np.uint16) << 8)).astype(np.float32)

        scales = np.stack(
            [smin[i] + u16(16 + 2 * i) / 65535.0 * (smax[i] - smin[i]) for i in range(3)], axis=1
        )
        color = a[:, 12:15].astype(np.float32) / 255.0
        opacity = a[:, 15].astype(np.float32) / 255.0

        qb = a[:, 22:26].astype(np.float32)
        quat = (qb[:, [3, 0, 1, 2]] - 128.0) / 128.0
        norm = np.linalg.norm(quat, axis=1, keepdims=True)
        quat = quat / np.where(norm > 1e-8, norm, 1.0)

        return Splats(means, scales, color, opacity, quat)

    def has_sh(self) -> bool:
        return os.path.isfile(os.path.join(self.root, "shcoef.bin"))

    def decode_sh(self, lod: int = 0) -> np.ndarray:
        """Degree-1..3 SH colour coefficients for one LOD block, shape (N, 15, 3) float32.

        shcoef.bin (fileType=Quality) is 64 bytes/splat, same splat order as data.bin:
        16 little-endian uint32 words; words 0..14 = SH coefficients 1..15 (degree 1: 3,
        degree 2: 5, degree 3: 7), word 15 = zero pad. Each word bit-packs the three
        channels as R:11 bits (0-10), G:10 bits (11-20), B:11 bits (21-31), linearly
        quantized over the per-channel 'shcoef' min/max from the .lcc manifest.
        Coefficient 0 (DC) is NOT here: it is folded into data.bin colour,
        sh0 = (color - 0.5) / 0.28209479.
        """
        n = self.total_splats
        sizes = self.lod_sizes
        start, end = sum(sizes[:lod]), sum(sizes[:lod + 1])
        raw = np.fromfile(os.path.join(self.root, "shcoef.bin"), dtype=np.uint8)
        u = raw[: n * 64].reshape(n, 64)[start:end, :60].copy().view("<u4").reshape(-1, 15)
        lo = hi = None
        for a in self.manifest["attributes"]:
            if a["name"] == "shcoef":
                lo, hi = np.asarray(a["min"], np.float32), np.asarray(a["max"], np.float32)
        c = np.stack([(u & 2047) / 2047.0, ((u >> 11) & 1023) / 1023.0, (u >> 21) / 2047.0], -1)
        return (lo + c.astype(np.float32) * (hi - lo)).astype(np.float32)

    def sh_colors(self, splats: "Splats", lod: int = 0) -> np.ndarray:
        """(N,16,3) gsplat-ready SH colours (pass with sh_degree=3): DC from data.bin + shcoef.bin."""
        dc = ((splats.color - 0.5) / 0.28209479)[:, None, :]
        return np.concatenate([dc, self.decode_sh(lod)], 1).astype(np.float32)

    def load_poses(self) -> list[dict]:
        """Capture poses (R quaternion (w,x,y,z), T xyz) in the metric splat frame."""
        return json.load(open(os.path.join(self.root, "assets", "poses.json")))["poses"]

    def pose_centers(self) -> np.ndarray:
        return np.array([p["T"] for p in self.load_poses()], float)


class PlyScan:
    """Standard 3DGS PLY export (e.g. LCC Studio 'export/point_cloud/iteration_*/point_cloud.ply') with the same
    interface as LccScan (decode / load_poses / pose_centers). Folder layout:
        <root>/point_cloud.ply   (x y z, f_dc_0..2, opacity logit, scale_0..2 log, rot_0..3 = w x y z)
        <root>/poses.csv         (device trajectory: ts, x, y, z, qx, qy, qz, qw) — positions only are used."""
    SH_C0 = 0.28209479177387814

    def __init__(self, root):
        self.root = root

    def _read(self):
        path = os.path.join(self.root, "point_cloud.ply")
        with open(path, "rb") as f:
            props, n = [], 0
            while True:
                line = f.readline().decode("ascii", "ignore").strip()
                if line.startswith("element vertex"):
                    n = int(line.split()[-1])
                elif line.startswith("property"):
                    props.append(line.split()[-1])
                elif line == "end_header":
                    break
            data = np.fromfile(f, dtype=np.float32, count=n * len(props)).reshape(n, len(props))
        return {k: data[:, i] for i, k in enumerate(props)}

    @property
    def lod_sizes(self):
        return [len(self._read()["x"])]

    def decode(self, lod: int = 0) -> Splats:
        v = self._read()
        means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
        color = np.clip(0.5 + self.SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1), 0, 1).astype(np.float32)
        opacity = (1.0 / (1.0 + np.exp(-v["opacity"]))).astype(np.float32)
        scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)).astype(np.float32)
        quat = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
        quat /= np.clip(np.linalg.norm(quat, axis=1, keepdims=True), 1e-8, None)
        return Splats(means, scales, color, opacity, quat)

    def has_sh(self) -> bool:
        return False

    def load_poses(self) -> list[dict]:
        a = np.loadtxt(os.path.join(self.root, "poses.csv"), delimiter=",", ndmin=2)
        return [{"ts": float(r[0]), "T": [float(r[1]), float(r[2]), float(r[3])],
                 "R": [float(r[7]), float(r[4]), float(r[5]), float(r[6])]} for r in a]

    def pose_centers(self) -> np.ndarray:
        return np.array([p["T"] for p in self.load_poses()], float)


def quat2R(w, x, y, z) -> np.ndarray:
    """Quaternion (w,x,y,z) -> 3x3 rotation matrix."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


GRAVITY = np.array([0.0, 0.0, -1.0])


def forward_of_pose(pose: dict) -> np.ndarray:
    """Camera forward axis in the splat frame from a capture pose (flipYZ convention)."""
    R = quat2R(*pose["R"])
    return (R @ np.diag([1.0, -1.0, -1.0]))[:, 2]


def gravity_upright_c2w(forward: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Build an upright camera-to-world from a forward vector, de-rolled against gravity.

    image-down = +(G - proj_forward(G)); no rot90 needed.
    """
    f = np.asarray(forward, float)
    f = f / np.linalg.norm(f)
    down = GRAVITY - np.dot(GRAVITY, f) * f
    down /= np.linalg.norm(down)
    right = np.cross(down, f)
    right /= np.linalg.norm(right)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([right, down, f], axis=1)
    c2w[:3, 3] = position
    return c2w


def c2w_from_yaw_pitch(position: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    """Upright camera-to-world from yaw/pitch (radians) at a fixed position."""
    import math
    f = np.array([math.cos(pitch) * math.cos(yaw),
                  math.cos(pitch) * math.sin(yaw),
                  -math.sin(pitch)])
    return gravity_upright_c2w(f, position)
