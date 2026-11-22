"""
01_frames_and_com.py  (Lessons 1-3 and 7)
Frames of the robot: world {W}, body {B} and the base frame {0} of every leg.
We find the feet in the world with a chain of transforms, check what a 5 deg body pitch
does to the legs, and compute the center of mass and the support polygon when standing.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import quadruped as qp

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
d = qp.DESIGNS["v2"]
l1, l2 = d["l1"], d["l2"]
H = qp.hips(d)

# standing pose: every foot right below its hip, body at 100 mm
q1, q2, _ = qp.ik_leg(0.0, -qp.BODY_H, l1, l2)
q_stand = np.array([q1, q2])
print(f"standing pose: q1 = {np.rad2deg(q1):.1f} deg, q2 = {np.rad2deg(q2):.1f} deg\n")

R = qp.R_B0
print("B_R_0 (leg base frame in the body frame), columns = x0, y0, z0:")
print(R)
print("R^T R = I ?", np.allclose(R.T @ R, np.eye(3)), "  det =", round(np.linalg.det(R), 6))

T_WB = qp.Trans(0, 0, qp.BODY_H)          # body frame 100 mm above the floor, level
print("\nfeet in the world frame:  W_p = W_T_B * B_T_0 * 0_p_foot")
feet = {}
for leg in qp.LEGS:
    p_W = (T_WB @ qp.fk_chain(q_stand, d, leg))[:3, 3]
    feet[leg] = p_W
    print(f"  {leg}: {np.round(p_W, 4)}")
print("  -> all feet have z = 0, so they touch the floor.")

# body pitch of 5 deg (walking up a small slope or a bad floor): what must the legs do?
pitch = np.deg2rad(5)
T_WB_p = qp.Trans(0, 0, qp.BODY_H) @ qp.Ry(pitch)
print("\nbody pitched 5 deg about y_B (nose down). Feet stay on the floor, so in {B}:")
dz_list = []
for leg in qp.LEGS:
    p_W = feet[leg]
    p_B = np.linalg.inv(T_WB_p) @ np.r_[p_W, 1]        # B_p = (W_T_B)^-1 W_p
    rel = p_B[:3] - np.array([H[leg][0], H[leg][1], 0])
    dz_list.append(rel[2] + qp.BODY_H)
    print(f"  {leg}: foot relative to hip x = {rel[0]*1000:6.1f} mm, z = {rel[2]*1000:7.1f} mm")
print(f"  leg height change needed: {min(dz_list)*1000:.1f} .. {max(dz_list)*1000:.1f} mm "
      f"(inside the +-10 mm we asked for in R1)")

# center of mass as a weighted average (Lesson 7)
qs = {leg: q_stand for leg in qp.LEGS}
c = qp.com(d, qs)
m = qp.mass_total(d)
m_th, lc1, _, m_sh, lc2, _ = qp.leg_params(d)
print(f"\nmass budget: core {qp.M_CORE*1000:.0f} g + payload {qp.M_PAYLOAD*1000:.0f} g + 8 servos "
      f"{8*qp.SERVOS[d['servo']]['mass']*1000:.0f} g + PLA legs/feet {4*(m_th+m_sh-qp.SERVOS[d['servo']]['mass'])*1000:.0f} g"
      f" = {m*1000:.0f} g")
x_sensor = d["x_body"] * qp.mass_body(d) / qp.M_PAYLOAD
print(f"the {qp.M_PAYLOAD*1000:.0f} g sensor is mounted {x_sensor*1000:.0f} mm in front of the body center,")
print(f"so the body parts have their COM at x = +{d['x_body']*1000:.1f} mm (this comes from the stability study)")
print(f"COM of the whole robot in {{B}}: {np.round(c*1000, 1)} mm  (knee-back legs pull it back and "
      f"{abs(c[2])*1000:.1f} mm down)")
feet_xy = np.array([feet[leg][:2] for leg in qp.LEGS])
marg = qp.stability_margin(feet_xy, c[:2])
print(f"standing on 4 feet: stability margin = {marg*1000:.1f} mm")
fz = qp.foot_forces(feet_xy, np.r_[c[:2], qp.BODY_H + c[2]], m, 0.0)
print("vertical foot forces [N]:", np.round(fz, 3), " sum =", round(fz.sum(), 3), " m g =", round(m * qp.G, 3))

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
ax.add_patch(Rectangle((-d["HX"] - 0.01, -d["FY"] + 0.02), 2 * d["HX"] + 0.02, 2 * d["FY"] - 0.04,
                       fc="#cfe2f3", ec="tab:blue", label="body"))
order = ["LF", "RF", "RH", "LH"]
ax.add_patch(Polygon(np.array([feet[l][:2] for l in order]), closed=True, fc="none", ec="tab:green",
                     lw=2, ls="--", label="support polygon (4 feet)"))
for leg in qp.LEGS:
    ax.plot(feet[leg][0], feet[leg][1], "o", color="tab:orange", ms=9)
    ax.text(H[leg][0] + 0.006, H[leg][1] + np.sign(H[leg][1]) * 0.008, leg)
ax.plot(*c[:2], "r*", ms=15, label=f"COM (margin {marg*1000:.0f} mm)")
for j, (lab, v) in enumerate([("x_B", [0.05, 0]), ("y_B", [0, 0.05])]):
    ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(v[0] * 1.15, v[1] * 1.15 + 0.004, lab)
ax.plot([], [], "o", color="tab:orange", label="foot (right below the hip)")
ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
ax.set_xlim(-0.14, 0.14); ax.set_ylim(-0.17, 0.13)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title("Top view: feet, COM and support polygon (standing)")
ax.legend(fontsize=8, loc="lower center", ncol=2)

ax = axes[1]   # side view of one leg with its frame {0}
kx, kz = qp.knee_rel(q_stand[0], l1)
fx, fz_ = qp.foot_rel(q_stand[0], q_stand[1], l1, l2)
ax.plot([0, kx, fx], [qp.BODY_H, qp.BODY_H + kz, qp.BODY_H + fz_], "-o", lw=4, color="tab:blue", ms=8)
ax.axhline(0, color="k", lw=1)
ax.annotate("", xy=(0, qp.BODY_H - 0.03), xytext=(0, qp.BODY_H), arrowprops=dict(arrowstyle="->", color="r", lw=2))
ax.text(0.003, qp.BODY_H - 0.03, "x0 (down)", color="r")
ax.annotate("", xy=(-0.03, qp.BODY_H), xytext=(0, qp.BODY_H), arrowprops=dict(arrowstyle="->", color="g", lw=2))
ax.text(-0.045, qp.BODY_H + 0.004, "y0 (back)", color="g")
ax.text(0.004, qp.BODY_H + 0.004, "hip (q1)")
ax.text(kx - 0.018, qp.BODY_H + kz, "knee (q2)")
ax.add_patch(Rectangle((qp.REGION_X[0], qp.BODY_H + qp.REGION_Z[0]), qp.REGION_X[1] - qp.REGION_X[0],
                       qp.REGION_Z[1] - qp.REGION_Z[0], fc="none", ec="crimson", ls="--", lw=1.5))
ax.text(qp.REGION_X[0], qp.BODY_H + qp.REGION_Z[1] + 0.004, "required foot region (R1)", color="crimson", fontsize=8)
ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
ax.set_xlim(-0.10, 0.08); ax.set_ylim(-0.02, 0.13)
ax.set_xlabel("x [m] (forward)"); ax.set_ylabel("z [m]")
ax.set_title(f"Side view of one leg (l1 = {l1*1000:.0f} mm, l2 = {l2*1000:.0f} mm), knee points back")
plt.tight_layout()
plt.savefig("figures/01_frames_com.png", dpi=200)

json.dump({"mass_total": m, "com_B": c.tolist(), "x_sensor": x_sensor, "standing_margin": marg, "q_stand_deg": np.rad2deg(q_stand).tolist(),
           "pitch5_leg_change_mm": [min(dz_list) * 1000, max(dz_list) * 1000]},
          open("results/01_frames.json", "w"), indent=2)
plt.show()
