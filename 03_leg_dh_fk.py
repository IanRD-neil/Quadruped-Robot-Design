"""
03_leg_dh_fk.py  (Lessons 3-4)
Modified DH table of one leg and the forward kinematics f: Q -> X.
Checks: (1) DH chain = direct geometry, (2) R^T R = I and det R = 1, (3) easy cases by hand.
Then we draw the whole robot in 3D with the frames of one leg.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
import quadruped as qp

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
d = qp.DESIGNS["v2"]
l1, l2 = d["l1"], d["l2"]
H = qp.hips(d)

print("Leg = planar 2R robot. Base frame {0} at the hip: x0 down, y0 backward, z0 = hip axis (y_B).")
print("Modified DH (Craig):  i-1Ti = Rx(alpha_{i-1}) Tx(a_{i-1}) Rz(theta_i) Tz(d_i)")
print(" i | a_{i-1} [mm] | alpha_{i-1} | d_i | theta_i")
print(" 1 |      0       |     0       |  0  |  q1  (hip servo)")
print(f" 2 |  l1 = {l1*1000:.0f}     |     0       |  0  |  q2  (knee servo)")
print(f" foot: 2T_F = Tx(l2 = {l2*1000:.0f} mm)")
print("Full chain:  B_T_F = B_T_0(leg) * 0_T_1(q1) * 1_T_2(q2) * Tx(l2)\n")

# ---- easy cases we can check by hand (foot relative to the hip, x forward, z up) ----
cases = [("leg straight down q=(0, 0)", [0, 0], [0, -(l1 + l2)]),
         ("thigh back 90 deg q=(90, 0)", [np.pi / 2, 0], [-(l1 + l2), 0]),
         ("knee 90 deg q=(0, -90)", [0, -np.pi / 2], [l2, -l1])]
for name, q, hand in cases:
    T = qp.fk_chain(q, d, "LF")
    rel = T[:3, 3] - [H["LF"][0], H["LF"][1], 0]
    print(f"{name:28s} foot (x, z) = ({rel[0]*1000:6.1f}, {rel[2]*1000:6.1f}) mm   "
          f"by hand = ({hand[0]*1000:6.1f}, {hand[1]*1000:6.1f}) mm")

# ---- 500 random configurations: DH chain vs closed form ----
rng = np.random.default_rng(5)
max_diff, max_orth, det_err = 0.0, 0.0, 0.0
for _ in range(500):
    q = [rng.uniform(*qp.Q1_LIM), rng.uniform(*qp.Q2_LIM)]
    leg = qp.LEGS[rng.integers(4)]
    T = qp.fk_chain(q, d, leg)
    x, z = qp.foot_rel(q[0], q[1], l1, l2)
    p_direct = np.array([H[leg][0] + x, H[leg][1], z])
    R = T[:3, :3]
    max_diff = max(max_diff, np.max(np.abs(T[:3, 3] - p_direct)))
    max_orth = max(max_orth, np.max(np.abs(R.T @ R - np.eye(3))))
    det_err = max(det_err, abs(np.linalg.det(R) - 1))
print(f"\n500 random q: max |p_chain - p_direct| = {max_diff:.1e} m")
print(f"              max |R^T R - I| = {max_orth:.1e},  max |det R - 1| = {det_err:.1e}")
print("The foot always stays in the plane y = y_hip: a 2-DOF leg can not move the foot sideways.")

# ---- 3D picture of the robot standing, frames of the LF leg ----
q1, q2, _ = qp.ik_leg(0.0, -qp.BODY_H, l1, l2)
T_WB = qp.Trans(0, 0, qp.BODY_H)
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
corners = [(-d["HX"], -d["FY"]), (d["HX"], -d["FY"]), (d["HX"], d["FY"]), (-d["HX"], d["FY"]), (-d["HX"], -d["FY"])]
ax.plot([c[0] for c in corners], [c[1] for c in corners], [qp.BODY_H] * 5, "k-", lw=2, label="body (hips)")
for leg in qp.LEGS:
    T0 = T_WB @ qp.T_B0(leg, d)
    T1 = T0 @ qp.mdh(*qp.dh_table([q1, q2], l1)[0])
    T2 = T1 @ qp.mdh(*qp.dh_table([q1, q2], l1)[1])
    TF = T2 @ qp.Tx(l2)
    P = np.array([T0[:3, 3], T2[:3, 3], TF[:3, 3]])
    ax.plot(P[:, 0], P[:, 1], P[:, 2], "-o", lw=3, color="tab:blue", ms=4)
    if leg == "LF":
        for Ti, name in [(T0, "{0}"), (T2, "{2}"), (TF, "{F}")]:
            for j, col in enumerate(["r", "g", "b"]):
                v = 0.03 * Ti[:3, j]
                ax.quiver(*Ti[:3, 3], *v, color=col, arrow_length_ratio=0.25, lw=1.2)
            ax.text(*(Ti[:3, 3] + [0.005, 0.01, 0.0]), name)
xx, yy = np.meshgrid([-0.15, 0.15], [-0.15, 0.15])
ax.plot_surface(xx, yy, 0 * xx, alpha=0.08, color="k")
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
ax.set_xlim(-0.15, 0.15); ax.set_ylim(-0.15, 0.15); ax.set_zlim(0, 0.15)
ax.set_box_aspect((1, 1, 0.5))
ax.view_init(22, -55)
ax.set_title("Robot standing (q1 = %.0f deg, q2 = %.0f deg)\nframes of the LF leg: red = x, green = y, blue = z"
             % (np.rad2deg(q1), np.rad2deg(q2)))
plt.tight_layout()
plt.savefig("figures/03_robot_3d.png", dpi=200)

json.dump({"max_chain_vs_direct": float(max_diff), "max_orthogonality_error": float(max_orth)},
          open("results/03_fk.json", "w"), indent=2)
plt.show()
