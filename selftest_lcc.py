"""
Self-test for lcc.py against the office scan's known-good constants.

Runs in plain python3 (numpy only) — no CUDA/gsplat needed. Checks that the
manifest-driven loader reproduces the expected splat count, LOD split, scale range
and geometry for a known scan.
"""
import sys
import numpy as np
from lcc import LccScan, quat2R, gravity_upright_c2w, forward_of_pose

OFFICE = sys.argv[1] if len(sys.argv) > 1 else "scans/office/lcc-result"

EXP_TOTAL = 1_091_050
EXP_LOD0 = 727_184
EXP_SMIN = 4.5399949158309028e-05
EXP_SMAX = np.array([1.6504223346710205, 1.5844607353210449, 1.7457984685897827], np.float32)

checks = []


def check(name, cond, detail=""):
    checks.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


scan = LccScan.open(OFFICE)

check("total_splats", scan.total_splats == EXP_TOTAL, f"{scan.total_splats}")
check("lod0 size", scan.lod_sizes[0] == EXP_LOD0, f"{scan.lod_sizes}")

smin, smax = scan._scale_attr()
check("scale min", np.allclose(smin, EXP_SMIN), f"{smin.tolist()}")
check("scale max", np.allclose(smax, EXP_SMAX), f"{smax.tolist()}")

s = scan.decode(lod=0)
check("decoded count == lod0", len(s) == EXP_LOD0, f"{len(s)}")
check("means finite", np.isfinite(s.means).all())
check("means metric range", np.abs(s.means).max() < 1e4, f"max|mean|={np.abs(s.means).max():.1f}")
check("scales within [smin,smax]",
      (s.scales >= smin.min() - 1e-6).all() and (s.scales <= smax.max() + 1e-6).all(),
      f"[{s.scales.min():.2e},{s.scales.max():.3f}]")
check("color in [0,1]", (s.color >= 0).all() and (s.color <= 1).all())
check("opacity in [0,1]", (s.opacity >= 0).all() and (s.opacity <= 1).all())
check("quat normalized", np.allclose(np.linalg.norm(s.quat, axis=1), 1.0, atol=1e-4))

bmin, bmax = scan.bounding_box
inside = (s.means.min(0) >= bmin - 1.0).all() and (s.means.max(0) <= bmax + 1.0).all()
check("means inside bounding_box", inside,
      f"means[{s.means.min(0).round(1)}..{s.means.max(0).round(1)}] bb[{bmin.round(1)}..{bmax.round(1)}]")

R = quat2R(1, 0, 0, 0)
check("quat2R identity", np.allclose(R, np.eye(3)))
c2w = gravity_upright_c2w(np.array([1.0, 0, 0]), np.zeros(3))
check("upright cam orthonormal", np.allclose(c2w[:3, :3] @ c2w[:3, :3].T, np.eye(3), atol=1e-6))

poses = scan.load_poses()
check("poses loaded", len(poses) > 0, f"{len(poses)} poses")
fwd = forward_of_pose(poses[0])
check("forward is unit", np.isclose(np.linalg.norm(fwd), 1.0))

n_fail = sum(1 for _, ok in checks if not ok)
print(f"\n{len(checks) - n_fail}/{len(checks)} passed")
sys.exit(1 if n_fail else 0)
