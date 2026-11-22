"""
quadruped.py
Model of our 4-legged walking robot. Every leg has 2 joints (hip pitch q1, knee pitch q2)
moved by hobby servos, so each leg is the planar 2R robot from the homeworks.
All numbered scripts import this file, so they all use the same numbers.
(work in progress: leg dynamics, gait and the servo checks come next)

Units: m, kg, s, rad, N, N*m
"""
import numpy as np

G = 9.81
LEGS = ["LF", "RF", "LH", "RH"]          # left/right, front/hind

# ------------------------------------------------------------------
# 1. Body and task numbers (fixed by the task)
# ------------------------------------------------------------------
# body length/width and stride are design variables -> they are in DESIGNS below
def hips(d):
    """hip positions (x, y) in the body frame for one design"""
    return {"LF": (+d["HX"], +d["FY"]), "RF": (+d["HX"], -d["FY"]),
            "LH": (-d["HX"], +d["FY"]), "RH": (-d["HX"], -d["FY"])}

BODY_H = 0.100      # nominal hip height above the floor
STEP_H = 0.025      # step height (to step over a 15 mm cable or edge)
OBSTACLE = 0.015

# required foot region relative to the hip (x forward, z up), requirement R1:
# strides up to 70 mm (+-35 mm) + 10 mm margin, body height 100 +-10 mm, plus the step height
REGION_X = (-0.045, 0.045)
REGION_Z = (-0.110, -0.065)

M_PAYLOAD = 0.050   # payload on top of the body (e.g. ultrasonic sensor or ESP32-CAM), R3
M_CORE = 0.160      # 3D printed body (~80 g) + ESP32 + PCA9685 + 2S 500 mAh LiPo + regulator + wires
SPEED_REQ = 1.0 / 30.0   # walk 1 m in 30 s (R4)
MARGIN_REQ = 0.005       # static stability margin >= 5 mm (R6)
SAG_REQ = 0.002          # body height error in stance <= 2 mm (R7)
CLEAR_REQ = 0.020        # swing foot must lift >= 20 mm (15 mm obstacle + 5 mm) (R8)

# ------------------------------------------------------------------
# 2. Leg constants
# ------------------------------------------------------------------
Q1_LIM = (np.deg2rad(-60), np.deg2rad(120))   # hip servo range (180 deg, mounted with offset)
Q2_LIM = (np.deg2rad(-150), np.deg2rad(0))    # knee can not bend the other way
Q2_MARGIN = np.deg2rad(20)                     # knee always bent >= 20 deg (singularity margin, R2)
KNEE = -1                                      # IK branch q2 < 0: knee points backward on all legs
MIN_LINK = 0.040                               # a link must be >= 40 mm to fit the servo and horn

RHO_THIGH = 0.12    # PLA thigh, mass per length [kg/m]
RHO_SHANK = 0.08    # PLA shank [kg/m]
M_FOOT = 0.003      # rubber foot pad

# hobby servos (typical datasheet values, check with the real part)
#   tau_stall [N*m], w0 = no-load speed [rad/s]
SERVOS = {
    "SG90":  dict(mass=0.009, volt=4.8, tau_stall=1.8 * 0.0981, w0=(np.pi / 3) / 0.10, gears="plastic"),
    "MG90S": dict(mass=0.0134, volt=6.0, tau_stall=2.2 * 0.0981, w0=(np.pi / 3) / 0.08, gears="metal"),
}
TORQUE_USE = 0.50   # rule of thumb for hobby servos: use at most 50 % of stall torque
SPEED_USE = 0.70    # and at most 70 % of the no-load speed
E_BAND = np.deg2rad(10)   # servo gives full torque at 10 deg position error (proportional band)
J_ROTOR = 1e-4      # reflected motor inertia at the servo output, estimated: 12 mm motor
                    # rotor ~6e-9 kg m^2 times gear ratio^2 (~120^2)  [kg m^2]

# ------------------------------------------------------------------
# 3. Design versions (design variables x_d)
#    v1 = first guess, v2 = final design after the verification loop
# ------------------------------------------------------------------
DESIGNS = {
    "v1": dict(l1=0.055, l2=0.055, servo="SG90", HX=0.080, FY=0.065, S=0.060,
               gait="continuous", T=0.80, beta=0.75, compensation=False),
    "v2": dict(l1=0.085, l2=0.050, servo="MG90S", HX=0.080, FY=0.090, S=0.070,
               gait="discontinuous", T_shift=0.25, T_swing=0.30, x_body=0.0125, compensation=True),
}


# ------------------------------------------------------------------
# 4. Elementary transforms and modified DH (Lessons 2-4)
# ------------------------------------------------------------------
def Rx(a):
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4); T[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]; return T

def Ry(a):
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4); T[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]; return T

def Rz(t):
    c, s = np.cos(t), np.sin(t)
    T = np.eye(4); T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]; return T

def Trans(x, y, z):
    T = np.eye(4); T[:3, 3] = [x, y, z]; return T

def Tx(a):
    return Trans(a, 0, 0)

def mdh(a_prev, alpha_prev, d_i, theta_i):
    """Modified DH block (Craig): Rx(alpha) Tx(a) Rz(theta) Tz(d)"""
    return Rx(alpha_prev) @ Tx(a_prev) @ Rz(theta_i) @ Trans(0, 0, d_i)

# leg base frame {0} at the hip: x0 points down, y0 points backward, z0 = hip axis (body y)
R_B0 = np.array([[0, -1, 0],
                 [0, 0, 1],
                 [-1, 0, 0]], dtype=float)

def T_B0(leg, d):
    """pose of the leg base frame {0} of one leg in the body frame {B}"""
    hx, hy = hips(d)[leg]
    T = np.eye(4); T[:3, :3] = R_B0; T[:3, 3] = [hx, hy, 0.0]
    return T

def dh_table(q, l1):
    """rows (a_{i-1}, alpha_{i-1}, d_i, theta_i) of one leg, tool = Tx(l2) to the foot"""
    return [(0.0, 0.0, 0.0, q[0]),
            (l1, 0.0, 0.0, q[1])]


# ------------------------------------------------------------------
# 5. Leg kinematics (written relative to the hip, in body directions x fwd, z up)
# ------------------------------------------------------------------
def fk_chain(q, d, leg="LF"):
    """foot pose in {B}: B_T_0 * 0_T_1 * 1_T_2 * Tx(l2)"""
    T = T_B0(leg, d)
    for row in dh_table(q, d["l1"]):
        T = T @ mdh(*row)
    return T @ Tx(d["l2"])

def foot_rel(q1, q2, l1, l2):
    """foot position relative to the hip: x (forward), z (up). Works with arrays."""
    x = -(l1 * np.sin(q1) + l2 * np.sin(q1 + q2))
    z = -(l1 * np.cos(q1) + l2 * np.cos(q1 + q2))
    return x, z

def knee_rel(q1, l1):
    return -l1 * np.sin(q1), -l1 * np.cos(q1)

def ik_leg(x, z, l1, l2, branch=KNEE):
    """analytical IK of the leg (planar 2R in frame {0}: x0 = -z, y0 = -x)."""
    x0, y0 = -z, -x
    c2 = (x0**2 + y0**2 - l1**2 - l2**2) / (2 * l1 * l2)
    ok = np.abs(c2) <= 1.0
    c2 = np.clip(c2, -1.0, 1.0)
    s2 = branch * np.sqrt(1 - c2**2)
    q2 = np.arctan2(s2, c2)
    q1 = np.arctan2(y0, x0) - np.arctan2(l2 * s2, l1 + l2 * c2)
    return q1, q2, ok

def joint_ok(q1, q2, use_margin=True):
    ok = (q1 >= Q1_LIM[0]) & (q1 <= Q1_LIM[1]) & (q2 >= Q2_LIM[0]) & (q2 <= Q2_LIM[1])
    if use_margin:
        ok = ok & (np.abs(q2) >= Q2_MARGIN)
    return ok

def jacobian(q, l1, l2):
    """J = d(x, z)/d(q1, q2) of the foot, relative to the hip. det J = -l1 l2 sin(q2)"""
    q1, q2 = q
    s1, c1, s12, c12 = np.sin(q1), np.cos(q1), np.sin(q1 + q2), np.cos(q1 + q2)
    return np.array([[-(l1 * c1 + l2 * c12), -l2 * c12],
                     [l1 * s1 + l2 * s12, l2 * s12]])

def region_grid(step=0.005):
    xs = np.arange(REGION_X[0], REGION_X[1] + 1e-9, step)
    zs = np.arange(REGION_Z[0], REGION_Z[1] + 1e-9, step)
    return np.meshgrid(xs, zs)


# ------------------------------------------------------------------
# 6. Mass properties (Lesson 7)
# ------------------------------------------------------------------
def leg_params(d):
    """masses and COM distances of thigh (+ knee servo at the knee) and shank (+ foot pad)"""
    l1, l2 = d["l1"], d["l2"]
    ms = SERVOS[d["servo"]]["mass"]
    m_th = RHO_THIGH * l1 + ms
    lc1 = (RHO_THIGH * l1 * l1 / 2 + ms * l1) / m_th
    I1 = RHO_THIGH * l1 * l1**2 / 12 + RHO_THIGH * l1 * (l1 / 2 - lc1)**2 + ms * (l1 - lc1)**2
    m_sh = RHO_SHANK * l2 + M_FOOT
    lc2 = (RHO_SHANK * l2 * l2 / 2 + M_FOOT * l2) / m_sh
    I2 = RHO_SHANK * l2 * l2**2 / 12 + RHO_SHANK * l2 * (l2 / 2 - lc2)**2 + M_FOOT * (l2 - lc2)**2
    return m_th, lc1, I1, m_sh, lc2, I2

def mass_body(d, m_pay=None):
    """everything that does not move with the legs: core, payload and the 4 hip servos"""
    m_pay = M_PAYLOAD if m_pay is None else m_pay
    return M_CORE + m_pay + 4 * SERVOS[d["servo"]]["mass"]

def mass_total(d, m_pay=None):
    m_th, _, _, m_sh, _, _ = leg_params(d)
    return mass_body(d, m_pay) + 4 * (m_th + m_sh)

def com(d, qs, m_pay=None):
    """center of mass in {B} as a weighted average (Lesson 7). qs = {leg: (q1, q2)}"""
    m_th, lc1, _, m_sh, lc2, _ = leg_params(d)
    l1 = d["l1"]
    m_sum = mass_body(d, m_pay)
    num = m_sum * np.array([d.get("x_body", 0.0), 0.0, 0.0])   # body parts COM (battery position sets x)
    H = hips(d)
    for leg in LEGS:
        q1, q2 = qs[leg]
        hx, hy = H[leg]
        c_th = np.array([hx - lc1 * np.sin(q1), hy, -lc1 * np.cos(q1)])
        c_sh = np.array([hx - l1 * np.sin(q1) - lc2 * np.sin(q1 + q2), hy,
                         -l1 * np.cos(q1) - lc2 * np.cos(q1 + q2)])
        num += m_th * c_th + m_sh * c_sh
        m_sum += m_th + m_sh
    return num / m_sum


# ------------------------------------------------------------------
# 7. Support polygon, stability margin and foot forces (Lesson 7 statics)
# ------------------------------------------------------------------
def stability_margin(feet_xy, p):
    """smallest distance from point p to the edges of the support polygon (negative = outside)"""
    c = feet_xy.mean(0)
    P = feet_xy[np.argsort(np.arctan2(feet_xy[:, 1] - c[1], feet_xy[:, 0] - c[0]))]
    m = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        e = b - a
        n = np.array([-e[1], e[0]]) / np.linalg.norm(e)
        m.append((p - a) @ n)
    return min(m)

def foot_forces(feet_xy, c, m, ax):
    """vertical ground forces from Newton-Euler of the whole robot (feet on the floor, z = 0):
    sum f = m g,  sum f x = m g xc - m ax hc,  sum f y = m g yc.
    3 feet: exact. 4 feet: minimum-norm solution (statically indeterminate)."""
    A = np.vstack([np.ones(len(feet_xy)), feet_xy[:, 0], feet_xy[:, 1]])
    b = np.array([m * G, m * G * c[0] - m * ax * c[2], m * G * c[1]])
    return A.T @ np.linalg.solve(A @ A.T, b)
