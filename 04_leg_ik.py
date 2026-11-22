"""
04_leg_ik.py  (Lesson 5)
Analytical IK of the leg (two branches: knee back / knee front), FK(IK(p)) check along the
foot path of the walking cycle, reachability, and a damped least-squares (DLS) check.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
import quadruped as qp

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
d = qp.DESIGNS["v2"]
l1, l2 = d["l1"], d["l2"]

# ---- all points of the required foot region (5 mm grid) and its corners ----
# (the real foot path comes with the gait, in the next part of the project)
X, Z = qp.region_grid(0.005)
pts = np.c_[X.ravel(), Z.ravel()]
corners = np.array([[x, z] for x in qp.REGION_X for z in qp.REGION_Z])
worst, inside = 0.0, True
for p in np.vstack([pts, corners]):
    q1, q2, ok = qp.ik_leg(p[0], p[1], l1, l2, branch=-1)
    x, z = qp.foot_rel(q1, q2, l1, l2)
    worst = max(worst, np.hypot(x - p[0], z - p[1]))
    inside = inside and ok and bool(qp.joint_ok(q1, q2))
print(f"{len(pts)} region points + 4 corners: worst |FK(IK(p)) - p| = {worst:.1e} m, "
      f"all inside joint limits with knee bend >= 20 deg: {inside}")

print("\nexample poses (knee-back branch, q2 < 0) and the other branch:")
print("  point                     q1 [deg]  q2 [deg]   | other branch q1, q2 [deg]")
examples = {"stance start (+S/2)": (d["S"] / 2, -qp.BODY_H), "stance end (-S/2)": (-d["S"] / 2, -qp.BODY_H),
            "top of the swing": (0.0, -qp.BODY_H + qp.STEP_H), "deep corner of R1": (qp.REGION_X[0], qp.REGION_Z[0])}
for name, (x, z) in examples.items():
    a1, a2, _ = qp.ik_leg(x, z, l1, l2, branch=-1)
    b1, b2, _ = qp.ik_leg(x, z, l1, l2, branch=+1)
    ok_b = bool(qp.joint_ok(b1, b2))
    print(f"  {name:24s} {np.rad2deg(a1):8.1f}  {np.rad2deg(a2):8.1f}   | {np.rad2deg(b1):6.1f}, {np.rad2deg(b2):6.1f}"
          f"  {'(allowed)' if ok_b else '(not allowed: knee limit)'}")

# ---- reachability with |l1 - l2| <= r <= l1 + l2 ----
print("\nreachability (r = distance hip -> foot):")
for p in [(0.0, -0.10), (0.0, -0.14), (0.0, -0.02), (0.06, -0.09)]:
    r = np.hypot(*p)
    cls = "reachable" if abs(l1 - l2) <= r <= l1 + l2 else ("too far" if r > l1 + l2 else "too close")
    print(f"  p = ({p[0]*1000:5.0f}, {p[1]*1000:5.0f}) mm  r = {r*1000:5.1f} mm -> {cls}"
          f"   (annulus {abs(l1-l2)*1000:.0f} .. {(l1+l2)*1000:.0f} mm)")

# ---- DLS numerical IK (Lesson 5 script 3) for the top of the swing ----
p_d = np.array([0.0, -qp.BODY_H + qp.STEP_H])
a1, a2, _ = qp.ik_leg(*p_d, l1, l2)
def dls(q0, lam, iters=40):
    q = np.array(q0, float); hist = []
    for _ in range(iters):
        e = p_d - np.array(qp.foot_rel(q[0], q[1], l1, l2))
        hist.append(np.linalg.norm(e))
        J = qp.jacobian(q, l1, l2)
        q = q + J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(2), e)
    return q, np.array(hist)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
dls_rows = []
print("\nDLS numerical IK, target = top of the swing:")
for q0, lam in [([0.3, -1.2], 0.01), ([0.3, -1.2], 0.05), ([0.0, 0.3], 0.01)]:
    q, h = dls(q0, lam)
    same = np.allclose(q, [a1, a2], atol=1e-6)
    dls_rows.append(dict(seed=q0, lam=lam, residual=float(h[-1]), q_deg=np.rad2deg(q).tolist()))
    print(f"  seed {q0}, lambda={lam}: residual {h[-1]:.1e} m, q = {np.round(np.rad2deg(q), 1)} deg, "
          f"knee-back branch: {same}")
    axes[1].semilogy(h + 1e-16, lw=2, label=f"seed {q0}, $\\lambda$ = {lam}")
axes[1].set_xlabel("iteration"); axes[1].set_ylabel("residual ||p_d - p(q)|| [m]")
axes[1].set_title("DLS converges, but a bad seed gives the other branch")
axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=8)

ax = axes[0]
for br, col, lab in [(-1, "tab:blue", "knee back (q2 < 0, used)"), (+1, "0.6", "knee front (q2 > 0)")]:
    q1, q2, _ = qp.ik_leg(-d["S"] / 2, -qp.BODY_H, l1, l2, branch=br)
    kx, kz = qp.knee_rel(q1, l1)
    ax.plot([0, kx * 1000, -d["S"] / 2 * 1000], [0, kz * 1000, -qp.BODY_H * 1000], "-o", lw=3, color=col, label=lab)
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((qp.REGION_X[0] * 1000, qp.REGION_Z[0] * 1000), (qp.REGION_X[1] - qp.REGION_X[0]) * 1000,
                       (qp.REGION_Z[1] - qp.REGION_Z[0]) * 1000, fc="none", ec="tab:orange", ls="--", lw=1.5,
                       label="required foot region (R1)"))
ax.plot(0, 0, "ks", ms=8)
ax.axhline(-qp.BODY_H * 1000, color="k", lw=0.8)
ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
ax.set_xlim(-120, 100); ax.set_ylim(-130, 30)
ax.set_xlabel("x [mm] (forward)"); ax.set_ylabel("z [mm]")
ax.set_title("Two IK branches for the stance end point")
ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()
plt.savefig("figures/04_ik_branches_dls.png", dpi=200)

json.dump({"worst_fk_ik_error": float(worst), "all_inside_limits": bool(inside), "dls": dls_rows},
          open("results/04_ik.json", "w"), indent=2)
plt.show()
