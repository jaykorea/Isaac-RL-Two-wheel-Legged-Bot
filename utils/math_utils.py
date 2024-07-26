import numpy as np
from scipy.spatial.transform import Rotation as R


class MathUtils:
    @staticmethod
    def quaternion_to_euler_array(quat):
        x, y, z, w = quat
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return np.array([roll_x, pitch_y, yaw_z])

    @staticmethod
    def quat_rotate_inverse(quaternion, vectors):
        q = np.array(quaternion)
        v = np.array(vectors)
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        t = 2 * np.cross(q_conj[1:], v)
        return v + q_conj[0] * t + np.cross(q_conj[1:], t)

    @staticmethod
    def wrap_to_2pi(angle):
        wrapped_angle = np.fmod(angle, 4 * np.pi)
        if wrapped_angle > 2 * np.pi:
            wrapped_angle -= 4 * np.pi
        elif wrapped_angle < -2 * np.pi:
            wrapped_angle += 4 * np.pi
        return wrapped_angle

    @staticmethod
    def quat_to_base_vel(quat, qvel):
        r = R.from_quat(quat)
        v = r.apply(qvel, inverse=True).astype(np.double)  # In the base frame
        return v
