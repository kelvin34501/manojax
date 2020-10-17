from jax import numpy as np
import os


class AxisLayer:
    def __init__(self):
        self.joints_mapping = [5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]
        up_axis_base = np.vstack((np.array([[0, 1, 0]]).repeat(12, axis=0), np.array([[1, 1, 1]]).repeat(3, axis=0)))
        self.up_axis_base = np.expand_dims(up_axis_base.astype(np.float32), 0)

    @staticmethod
    def _recov_axis_batch(hand_joints, transf, joints_mapping, up_axis_base):
        """
        input: hand_joints[B, 21, 3], transf[B, 16, 4, 4]
        output: b_axis[B, 15, 3], u_axis[B, 15, 3], l_axis[B, 15, 3]
        """
        bs = transf.shape[0]

        b_axis = hand_joints[:, joints_mapping] - hand_joints[:, [i + 1 for i in joints_mapping]]
        b_axis = (np.transpose(transf[:, 1:, :3, :3], (0, 1, 3, 2)) @ np.expand_dims(b_axis, -1)).squeeze(-1)

        l_axis = np.cross(b_axis, up_axis_base)

        u_axis = np.cross(l_axis, b_axis)

        return (
            b_axis / np.expand_dims(np.linalg.norm(b_axis, axis=2), -1),
            u_axis / np.expand_dims(np.linalg.norm(u_axis, axis=2), -1),
            l_axis / np.expand_dims(np.linalg.norm(l_axis, axis=2), -1),
        )

    def __call__(self, hand_joints, transf):
        return self._recov_axis_batch(hand_joints, transf, self.joints_mapping, self.up_axis_base)
