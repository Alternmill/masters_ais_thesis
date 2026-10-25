#!/usr/bin/env python
"""Interactive viser viewer for an LCC splat + its capture trajectory.

Usage: python view_splat.py --scan scans/bathroom_2/lcc-result [--lod 0] [--port 8080]
"""
import argparse, sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lcc
import viser


def quats2R(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def R2quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return np.array([w, x, y, z], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="scans/bathroom_2/lcc-result")
    ap.add_argument("--lod", type=int, default=0, help="0=finest")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--frustum-every", type=int, default=50)
    ap.add_argument("--frustum-scale", type=float, default=0.12)
    a = ap.parse_args()

    sc = lcc.LccScan.open(a.scan)
    print(f"[view] {a.scan}  total={sc.total_splats}  lods={sc.lod_sizes}  using lod {a.lod}")
    sp = sc.decode(lod=a.lod)
    n = len(sp)
    print(f"[view] decoded {n} splats; building covariances...")

    sp.quat[:] = 0.0; sp.quat[:, 0] = 1.0
    R = quats2R(sp.quat)
    S2 = (sp.scales.astype(np.float32) ** 2)
    cov = np.einsum("nij,nj,nkj->nik", R, S2, R).astype(np.float32)
    rgbs = (np.clip(sp.color, 0, 1) * 255).astype(np.uint8)
    opac = sp.opacity.astype(np.float32)

    poses = sc.load_poses()
    centers = np.array([p["T"] for p in poses], np.float32)
    print(f"[view] {len(poses)} capture poses")

    server = viser.ViserServer(port=a.port)
    server.scene.set_up_direction("+z")

    server.scene._add_gaussian_splats(
        "/splat", centers=sp.means.astype(np.float32), covariances=cov,
        rgbs=rgbs, opacities=opac[:, None],
    )

    line_h = server.scene.add_spline_catmull_rom(
        "/traj/line", positions=centers, curve_type="centripetal",
        line_width=3.0, color=(255, 90, 0),
    )
    server.scene.add_icosphere("/traj/start", radius=0.05, color=(0, 220, 0), position=centers[0])
    server.scene.add_icosphere("/traj/end", radius=0.05, color=(220, 0, 0), position=centers[-1])

    idxs = list(range(0, len(poses), a.frustum_every))
    frusta = []
    for i in idxs:
        Rc = lcc.quat2R(*poses[i]["R"]) @ np.diag([1.0, -1.0, -1.0])
        wxyz = R2quat(Rc)
        frusta.append(server.scene.add_camera_frustum(
            f"/traj/cam/{i:05d}", fov=1.0, aspect=1.3333, scale=a.frustum_scale,
            color=(0, 160, 255), wxyz=wxyz, position=centers[i],
        ))
    print(f"[view] {len(idxs)} frustums drawn (every {a.frustum_every})")

    g_line = server.gui.add_checkbox("trajectory line", True)
    g_cams = server.gui.add_checkbox("camera frustums", True)
    g_line.on_update(lambda _: setattr(line_h, "visible", g_line.value))
    def _toggle_cams(_):
        for f in frusta: f.visible = g_cams.value
    g_cams.on_update(_toggle_cams)

    print(f"\n[view] READY -> http://localhost:{a.port}\n")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
