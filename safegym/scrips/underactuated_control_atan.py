import gymnasium as gym
import numpy as np

# register all SafeGym envs
from safegym import register_all

register_all()

Kp = np.diag([1.0, 1.0])
Kv = np.diag([1.0, 1.0])
k_phi, k_omega = 5.0, 2.0
m = 40.0


def ctrl(obs: np.ndarray) -> np.ndarray:
    # SafeGym SE2 absolute observation layout:
    # [x, y, cos(theta), sin(theta), cos(phi_t), sin(phi_t), vx, vy, omega, phi_dot_t]
    x = float(obs[0])
    y = float(obs[1])
    vx = float(obs[6])
    vy = float(obs[7])
    phi = float(np.arctan2(obs[3], obs[2]))  # chaser heading
    omega = float(obs[8])  # chaser angular rate

    p = np.array([x, y], dtype=np.float32)
    v = np.array([vx, vy], dtype=np.float32)
    p_star = np.zeros(2, dtype=np.float32)
    v_star = np.zeros(2, dtype=np.float32)

    a_d = -Kp @ (p - p_star) - Kv @ (v - v_star)
    phi_d = float(np.arctan2(a_d[1], a_d[0] + 1e-12))

    # attitude control (wrap angle error to [-pi, pi])
    e_phi = np.arctan2(np.sin(phi - phi_d), np.cos(phi - phi_d))
    tau_des = -k_phi * e_phi - k_omega * omega  # desired torque (phys units)

    # Map to unit Box [-1, 1] expected by env when unit_action_space=True.
    # The env will scale by env.max_action inside __action_filter.
    # Use base env scaling (OrderEnforcing wrapper -> unwrapped)
    max_act = float(getattr(env.unwrapped, "max_action", 1.0))
    u1_unit = float(np.clip((m * np.linalg.norm(a_d)) / (max_act + 1e-12), 0.0, 1.0))
    u2_unit = float(np.clip(tau_des / (max_act + 1e-12), -1.0, 1.0))
    return np.array([u1_unit, u2_unit], dtype=np.float32)


if __name__ == "__main__":
    # Use absolute observations to avoid manual de-normalization
    env = gym.make(
        "Satellite-SE",
        render_mode=None,
        normalized_obs=False,
        underactuated=True,
    )
    obs, info = env.reset()

    done = False
    while not done:
        u = ctrl(obs)
        obs, r, term, trunc, info = env.step(u)
        done = term or trunc
    env.close()
