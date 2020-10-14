"""
Jax MANO layer
Cite: Embodied Hands: Modeling and Capturing Hands and Bodies Together
Adapted from "https://github.com/hassony2/manopth"
Author: Lv Jun
Modified by kelvin34501, linxiny
"""
import os
import jax.numpy as np
from jax import jit
import pickle
import cv2
from manopth.quatutils import (
    quaternion_to_angle_axis,
    quaternion_inv,
    quaternion_mul,
    quaternion_to_rotation_matrix,
    normalize_quaternion,
)


class ManoLayer:
    __constants__ = [
        "use_pca",
        "rot",
        "ncomps",
        "kintree_parents",
        "check",
        "side",
        "center_idx",
        "joint_rot_mode",
        "root_rot_mode",
    ]

    def __init__(
        self,
        center_idx=None,
        flat_hand_mean=True,
        ncomps=6,
        side="right",
        mano_root="mano/models",
        use_pca=True,
        root_rot_mode="axisang",
        joint_rot_mode="axisang",
        robust_rot=False,
        return_transf=False,
        return_full_pose=False,
    ):
        """
        Args:
            center_idx: index of center joint in our computations,
                if -1 centers on estimate of palm as middle of base
                of middle finger and wrist
            flat_hand_mean: if True, (0, 0, 0, ...) pose coefficients match
                flat hand, else match average hand pose
            mano_root: path to MANO pkl files for left and right hand
            ncomps: number of PCA components form pose space (<45)
            side: 'right' or 'left'
            use_pca: Use PCA decomposition for pose space.
            joint_rot_mode: 'axisang' or 'rotmat', ignored if use_pca
        """
        super().__init__()

        self.center_idx = center_idx
        self.robust_rot = robust_rot

        # check root_rot_mode feasible, and set self.rot
        if root_rot_mode == "axisang":
            self.rot = 3
        elif root_rot_mode == "rotmat":
            self.rot = 6
        elif root_rot_mode == "quat":
            self.rot = 4
        else:
            raise KeyError(
                "root_rot_mode not found. shoule be one of 'axisang' or 'rotmat' or 'quat'. got {}".format(root_rot_mode)
            )

        # todo: flat_hand_mean have issues
        self.flat_hand_mean = flat_hand_mean

        # toggle extra return information
        self.return_transf = return_transf
        self.return_full_pose = return_full_pose

        # record side of hands
        self.side = side

        # check use_pca and joint_rot_mode
        if use_pca and joint_rot_mode != "axisang":
            raise TypeError("if use_pca, joint_rot_mode must be 'axisang'. got {}".format(joint_rot_mode))
        # record use_pca flag and joint_rot_mode
        self.use_pca = use_pca
        self.joint_rot_mode = joint_rot_mode
        # self.ncomps only work in axisang mode
        if use_pca:
            self.ncomps = ncomps
        else:
            self.ncomps = 45

        # do more checks on root_rot_mode, in case mode error
        if self.joint_rot_mode == "axisang":
            # add restriction to root_rot_mode
            if root_rot_mode not in ["axisang", "rotmat"]:
                err_msg = "rot_mode not compatible, "
                err_msg += "when joint_rot_mode is 'axisang', root_rot_mode should be one of "
                err_msg += "'axisang' or 'rotmat', got {}".format(root_rot_mode)
                raise KeyError(err_msg)
        else:
            # for 'rotmat' or 'quat', there rot_mode must be same for joint and root
            if root_rot_mode != self.joint_rot_mode:
                err_msg = "rot_mode not compatible, "
                err_msg += "should get the same rot mode for joint and root, "
                err_msg += "got {} for root and {} for joint".format(root_rot_mode, self.joint_rot_mode)
                raise KeyError(err_msg)
        # record root_rot_mode
        self.root_rot_mode = root_rot_mode

        # load model according to side flag
        if side == "right":
            self.mano_path = os.path.join(mano_root, "MANO_RIGHT.pkl")
        elif side == "left":
            self.mano_path = os.path.join(mano_root, "MANO_LEFT.pkl")

        # parse and register stuff
        smpl_data = self._ready_arguments(self.mano_path)

        hands_components = smpl_data["hands_components"]  # 45*45

        self.smpl_data = smpl_data
        self.betas = np.array(smpl_data["betas"])[np.newaxis, ...]
        self.shapedirs = np.array(smpl_data["shapedirs"])
        self.posedirs = np.array(smpl_data["posedirs"])
        self.v_template = np.array(smpl_data["v_template"])[np.newaxis, ...]
        self.J_regressor = np.array(smpl_data["J_regressor"].toarray())
        self.weights = np.array(smpl_data["weights"])
        self.faces = np.array(smpl_data["f"]).astype(np.int32)

        # Get hand mean
        hands_mean = np.zeros(hands_components.shape[1]) if flat_hand_mean else smpl_data["hands_mean"]
        if self.use_pca or self.joint_rot_mode == "axisang":
            # Save as axis-angle
            self.hands_mean = hands_mean.copy()[np.newaxis, ...]  # 45 all zeros
            selected_components = hands_components[:ncomps]
            self.selected_comps = np.array(selected_components)
        elif self.joint_rot_mode == "rotmat":
            self.hands_mean_rotmat = self._batch_rodrigues(hands_mean.reshape((15, 3))).reshape(15, 3, 3)
        elif self.joint_rot_mode == "quat":
            # TODO deal with flat hand mean
            self.hands_mean_quat = None
        else:
            raise KeyError(
                "joint_rot_mode not found. shoule be one of 'axisang' or 'rotmat' or 'quat'. got {}".format(
                    self.joint_rot_mode
                )
            )

        # Kinematic chain params
        self.kintree_table = smpl_data["kintree_table"]
        parents = list(self.kintree_table[0].tolist())
        self.kintree_parents = parents

    @staticmethod
    def _ready_arguments(fname_or_dict, posekey4vposed="pose"):
        dd = pickle.load(open(fname_or_dict, "rb"), encoding="latin1")

        want_shapemodel = "shapedirs" in dd
        nposeparms = dd["kintree_table"].shape[1] * 3

        if "trans" not in dd:
            dd["trans"] = np.zeros(3)
        if "pose" not in dd:
            dd["pose"] = np.zeros(nposeparms)
        if "shapedirs" in dd and "betas" not in dd:
            dd["betas"] = np.zeros(dd["shapedirs"].shape[-1])

        for s in [
            "v_template",
            "weights",
            "posedirs",
            "pose",
            "trans",
            "shapedirs",
            "betas",
            "J",
        ]:
            if (s in dd) and not hasattr(dd[s], "dterms"):
                dd[s] = np.array(dd[s])

        assert posekey4vposed in dd
        if want_shapemodel:
            dd["v_shaped"] = dd["shapedirs"].dot(dd["betas"]) + dd["v_template"]
            v_shaped = dd["v_shaped"]
            J_tmpx = dd["J_regressor"] * v_shaped[:, 0]
            J_tmpy = dd["J_regressor"] * v_shaped[:, 1]
            J_tmpz = dd["J_regressor"] * v_shaped[:, 2]
            dd["J"] = np.vstack((J_tmpx, J_tmpy, J_tmpz)).T
            pose_map_res = ManoLayer._lrotmin(dd[posekey4vposed])
            dd["v_posed"] = v_shaped + dd["posedirs"].dot(pose_map_res)
        else:
            pose_map_res = ManoLayer._lrotmin(dd[posekey4vposed])
            dd_add = dd["posedirs"].dot(pose_map_res)
            dd["v_posed"] = dd["v_template"] + dd_add

        return dd

    @staticmethod
    def _lrotmin(p):
        p = p.ravel()[3:]
        return np.concatenate([(cv2.Rodrigues(pp)[0] - np.eye(3)).ravel() for pp in p.reshape((-1, 3))]).ravel()

    @staticmethod
    def _posemap_axisang(pose_vectors):
        rot_nb = int(pose_vectors.shape[1] / 3)
        pose_vec_reshaped = pose_vectors.reshape(-1, 3)
        rot_mats = ManoLayer._batch_rodrigues(pose_vec_reshaped)
        rot_mats = rot_mats.reshape(pose_vectors.shape[0], rot_nb * 9)
        pose_maps = ManoLayer._subtract_flat_id(rot_mats)
        return pose_maps, rot_mats

    @staticmethod
    def _subtract_flat_id(rot_mats):
        # Subtracts identity as a flattened tensor
        rot_nb = int(rot_mats.shape[1] / 9)
        id_flat = np.tile(np.eye(3, dtype=rot_mats.dtype).reshape(1, 9), (1, rot_nb))
        # id_flat.requires_grad = False
        results = rot_mats - id_flat
        return results

    @staticmethod
    def _quat2mat(quat):
        """Convert quaternion coefficients to rotation matrix.
        Args:
            quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
        Returns:
            Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
        """
        norm_quat = quat
        norm_quat = norm_quat / np.linalg.norm(norm_quat + 1e-8, axis=1, keepdims=True)
        w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

        batch_size = quat.shape[0]

        w2, x2, y2, z2 = w ** 2, x ** 2, y ** 2, z ** 2
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        rotMat = np.stack(
            [
                w2 + x2 - y2 - z2,
                2 * xy - 2 * wz,
                2 * wy + 2 * xz,
                2 * wz + 2 * xy,
                w2 - x2 + y2 - z2,
                2 * yz - 2 * wx,
                2 * xz - 2 * wy,
                2 * wx + 2 * yz,
                w2 - x2 - y2 + z2,
            ],
            axis=1,
        ).reshape(batch_size, 3, 3)
        return rotMat

    @staticmethod
    def _batch_rodrigues(axisang):
        # axisang N x 3
        axisang_norm = np.linalg.norm(axisang + 1e-8, axis=1)
        angle = axisang_norm[..., np.newaxis]
        axisang_normalized = axisang / angle
        angle = angle * 0.5
        v_cos = np.cos(angle)
        v_sin = np.sin(angle)
        quat = np.concatenate((v_cos, v_sin * axisang_normalized), 1)
        rot_mat = ManoLayer._quat2mat(quat)
        rot_mat = rot_mat.reshape(rot_mat.shape[0], 9)
        return rot_mat

    @staticmethod
    def _with_zeros(tensor):
        batch_size = tensor.shape[0]
        padding = np.array([0.0, 0.0, 0.0, 1.0])

        concat_list = (tensor, np.tile(padding.reshape(1, 1, 4), (batch_size, 1, 1)))
        cat_res = np.concatenate(concat_list, 1)
        return cat_res

    @staticmethod
    def _compute_rotation_matrix_from_ortho6d(poses):
        """
        Code from
        https://github.com/papagina/RotationContinuity
        On the Continuity of Rotation Representations in Neural Networks
        Zhou et al. CVPR19
        https://zhouyisjtu.github.io/project_rotation/rotation.html
        """
        x_raw = poses[:, 0:3]  # batch*3
        y_raw = poses[:, 3:6]  # batch*3

        x = ManoLayer._normalize_vector(x_raw)  # batch*3
        z = ManoLayer._cross_product(x, y_raw)  # batch*3
        z = ManoLayer._normalize_vector(z)  # batch*3
        y = ManoLayer._cross_product(z, x)  # batch*3

        x = x.reshape((-1, 3, 1))
        y = y.reshape((-1, 3, 1))
        z = z.reshape((-1, 3, 1))
        matrix = np.concatenate((x, y, z), 2)  # batch*3*3
        return matrix

    @staticmethod
    def _robust_compute_rotation_matrix_from_ortho6d(poses):
        """
        Instead of making 2nd vector orthogonal to first
        create a base that takes into account the two predicted
        directions equally
        """
        x_raw = poses[:, 0:3]  # batch*3
        y_raw = poses[:, 3:6]  # batch*3

        x = ManoLayer._normalize_vector(x_raw)  # batch*3
        y = ManoLayer._normalize_vector(y_raw)  # batch*3
        middle = ManoLayer._normalize_vector(x + y)
        orthmid = ManoLayer._normalize_vector(x - y)
        x = ManoLayer._normalize_vector(middle + orthmid)
        y = ManoLayer._normalize_vector(middle - orthmid)
        # Their scalar product should be small !
        # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
        z = ManoLayer._normalize_vector(ManoLayer._cross_product(x, y))

        x = x.reshape((-1, 3, 1))
        y = y.reshape((-1, 3, 1))
        z = z.reshape((-1, 3, 1))
        matrix = np.concatenate((x, y, z), axis=2)  # batch*3*3
        # Check for reflection in matrix ! If found, flip last vector TODO
        assert (np.stack([np.linalg.det(mat) for mat in matrix]) < 0).sum() == 0
        return matrix

    @staticmethod
    def _normalize_vector(v):
        batch = v.shape[0]
        v_mag = np.sqrt(np.power(v, 2).sum(1))  # batch
        v_mag = np.max(v_mag, np.array([1e-8]))
        v_mag = np.broadcast_to(v_mag.reshape((batch, 1)), (batch, v.shape[1]))
        v = v / v_mag
        return v

    @staticmethod
    def _cross_product(u, v):
        batch = u.shape[0]
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

        out = np.concatenate((i.reshape(batch, 1), j.reshape(batch, 1), k.reshape(batch, 1)), 1)

        return out

    @staticmethod
    def _batch_rotprojs(batches_rotmats):
        proj_rotmats = []
        for batch_idx, batch_rotmats in enumerate(batches_rotmats):
            proj_batch_rotmats = []
            for rot_idx, rotmat in enumerate(batch_rotmats):
                # GPU implementation of svd is VERY slow
                # ~ 2 10^-3 per hit vs 5 10^-5 on cpu
                U, S, V = np.linalg.svd(rotmat)
                rotmat = np.matmul(U, V.transpose((0, 1)))
                orth_det = np.linalg.det(rotmat)
                # Remove reflection
                if orth_det < 0:
                    rotmat[:, 2] = -1 * rotmat[:, 2]

                proj_batch_rotmats.append(rotmat)
            proj_rotmats.append(np.stack(proj_batch_rotmats))
        return np.stack(proj_rotmats)

    def __call__(
        self, pose_coeffs, betas=np.zeros(1),
    ):
        batch_size = pose_coeffs.shape[0]
        if self.use_pca or self.joint_rot_mode == "axisang":
            # Get axis angle from PCA components and coefficients
            # Remove global rot coeffs
            hand_pose_coeffs = pose_coeffs[:, self.rot : self.rot + self.ncomps]
            if self.use_pca:
                full_hand_pose = hand_pose_coeffs @ self.selected_comps
            else:
                full_hand_pose = hand_pose_coeffs

            # Concatenate back global rot
            full_pose = np.concatenate((pose_coeffs[:, : self.rot], self.hands_mean + full_hand_pose), 1)
            if self.root_rot_mode == "axisang":
                # compute rotation matrixes from axis-angle while skipping global rotation
                pose_map, rot_map = self._posemap_axisang(full_pose)
                root_rot = rot_map[:, :9].reshape(batch_size, 3, 3)
                rot_map = rot_map[:, 9:]
                pose_map = pose_map[:, 9:]
            else:
                # th_posemap offsets by 3, so add offset or 3 to get to self.rot=6
                pose_map, rot_map = self._posemap_axisang(full_pose[:, 6:])
                if self.robust_rot:
                    root_rot = self._robust_compute_rotation_matrix_from_ortho6d(full_pose[:, :6])
                else:
                    root_rot = self._compute_rotation_matrix_from_ortho6d(full_pose[:, :6])
        elif self.joint_rot_mode == "rotmat":
            full_pose = pose_coeffs  # ! Dummy Assignment
            pose_rots = self._batch_rotprojs(pose_coeffs)
            rot_map = pose_rots[:, 1:].reshape((batch_size, -1))
            pose_map = self._subtract_flat_id(rot_map)
            root_rot = pose_rots[:, 0]
        elif self.joint_rot_mode == "quat":
            # we need th_rot_map, th_pose_map, root_rot
            # though do no assertion
            # th_pose_coeffs should be [B, 4 + 15 * 4] = [B, 64]
            full_pose = pose_coeffs  # ! Dummy Assignment
            batch_size = pose_coeffs.shape[0]
            pose_coeffs = pose_coeffs.reshape((batch_size, 16, 4))  # [B. 16, 4]
            all_rots = quaternion_to_rotation_matrix(pose_coeffs)  # [B, 16, 3, 3]
            # flatten things out
            root_rot = all_rots[:, 0, :, :]  # [B, 3, 3]
            rot_map = all_rots[:, 1:, :].reshape((batch_size, -1))  # [B, 15 * 9]
            pose_map = self._subtract_flat_id(rot_map)
        else:
            raise KeyError(
                "joint_rot_mode not found. shoule be one of 'axisang' or 'rotmat' or 'quat'. got {}".format(
                    self.joint_rot_mode
                )
            )

        # Full axis angle representation with root joint
        if betas is None or betas.size == 1:
            v_shaped = np.matmul(self.shapedirs, self.betas.transpose(1, 0)).transpose((2, 0, 1)) + self.v_template
            j = np.matmul(self.J_regressor, v_shaped).tile((batch_size, 1, 1))
        else:
            v_shaped = np.matmul(self.shapedirs, betas.transpose((1, 0))).transpose((2, 0, 1)) + self.v_template
            j = np.matmul(self.J_regressor, v_shaped)
            # th_pose_map should have shape 20x135

        v_posed = v_shaped + np.matmul(self.posedirs, pose_map.transpose((1, 0))[np.newaxis, ...]).transpose((2, 0, 1))
        # Final T pose with transformation done !

        # Global rigid transformation

        root_j = j[:, 0, :].reshape(batch_size, 3, 1)
        root_trans = self._with_zeros(np.concatenate((root_rot, root_j), 2))

        all_rots = rot_map.reshape(rot_map.shape[0], 15, 3, 3)
        lev1_idxs = [1, 4, 7, 10, 13]
        lev2_idxs = [2, 5, 8, 11, 14]
        lev3_idxs = [3, 6, 9, 12, 15]
        lev1_rots = all_rots[:, [idx - 1 for idx in lev1_idxs]]
        lev2_rots = all_rots[:, [idx - 1 for idx in lev2_idxs]]
        lev3_rots = all_rots[:, [idx - 1 for idx in lev3_idxs]]
        lev1_j = j[:, lev1_idxs]
        lev2_j = j[:, lev2_idxs]
        lev3_j = j[:, lev3_idxs]

        # From base to tips
        # Get lev1 results
        all_transforms = [root_trans[:, np.newaxis, ...]]
        lev1_j_rel = lev1_j - root_j.transpose((0, 2, 1))
        lev1_rel_transform_flt = self._with_zeros(
            np.concatenate((lev1_rots, lev1_j_rel[..., np.newaxis]), 3).reshape(-1, 3, 4)
        )
        root_trans_flt = np.tile(root_trans[:, np.newaxis, ...], (1, 5, 1, 1)).reshape(root_trans.shape[0] * 5, 4, 4)
        lev1_flt = np.matmul(root_trans_flt, lev1_rel_transform_flt)
        all_transforms.append(lev1_flt.reshape(all_rots.shape[0], 5, 4, 4))

        # Get lev2 results
        lev2_j_rel = lev2_j - lev1_j
        lev2_rel_transform_flt = self._with_zeros(
            np.concatenate((lev2_rots, lev2_j_rel[..., np.newaxis]), 3).reshape(-1, 3, 4)
        )
        lev2_flt = np.matmul(lev1_flt, lev2_rel_transform_flt)
        all_transforms.append(lev2_flt.reshape(all_rots.shape[0], 5, 4, 4))

        # Get lev3 results
        lev3_j_rel = lev3_j - lev2_j
        lev3_rel_transform_flt = self._with_zeros(
            np.concatenate((lev3_rots, lev3_j_rel[..., np.newaxis]), 3).reshape(-1, 3, 4)
        )
        lev3_flt = np.matmul(lev2_flt, lev3_rel_transform_flt)
        all_transforms.append(lev3_flt.reshape(all_rots.shape[0], 5, 4, 4))

        reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
        results = np.concatenate(all_transforms, 1)[:, reorder_idxs]
        results_global = results

        joint_js = np.concatenate((j, np.zeros((j.shape[0], 16, 1))), 2)

        tmp2 = np.matmul(results, joint_js[..., np.newaxis])
        results2 = (results - np.concatenate([np.zeros(*tmp2.shape[:2], 4, 3), tmp2], 3)).transpose((0, 2, 3, 1))

        T = np.matmul(results2, self.weights.transpose((1, 0)))

        rest_shape_h = np.concatenate((v_posed.transpose((0, 2, 1)), np.ones((batch_size, 1, v_posed.shape[1]))), 1,)

        verts = (T * rest_shape_h[:, np.newaxis, ...]).sum(2).transpose((0, 2, 1))
        verts = verts[:, :, :3]
        jtr = results_global[:, :, :3, 3]
        # In addition to MANO reference joints we sample vertices on each finger
        # to serve as finger tips
        if self.side == "right":
            tips = verts[:, [745, 317, 444, 556, 673]]
        else:
            tips = verts[:, [745, 317, 445, 556, 673]]
        jtr = np.concatenate((jtr, tips), 1)

        # Reorder joints to match visualization utilities
        jtr = jtr[
            :, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
        ]

        # deal with center joint
        if self.center_idx is not None:
            center_joint = jtr[:, self.center_idx][:, np.newaxis, ...]
        else:  # ! Dummy Center Joint (B, 1, 3)
            center_joint = np.zeros_like(np.expand_dims(jtr[:, 0], 1))
        jtr = jtr - center_joint
        verts = verts - center_joint

        global_rot = results_global[:, :, :3, :3]  # (B, 16, 3, 3)
        global_t = results_global[:, :, :3, 3:]  # (B, 16, 3, 1)
        global_t = global_t - np.expand_dims(center_joint, -1)  # (B, [16], 3, 1)
        transf_global = np.concatenate([global_rot, global_t], axis=3)  # (B, 16, 3, 4)
        transf_global = self._with_zeros(transf_global.reshape((-1, 3, 4)))
        transf_global = transf_global.reshape((batch_size, 16, 4, 4))

        # Scale to milimeters
        # th_verts = th_verts * 1000
        # th_jtr = th_jtr * 1000
        results = [verts, jtr]  # (V, J)

        if self.return_transf:
            results = results + [transf_global]  # (V, J, T)
            if self.return_full_pose:
                results = results + [full_pose]  # (V, J, T, so3)
        elif self.return_full_pose:
            results = results + [full_pose]  # (V, J, so3)

        return tuple(results)


if __name__ == "__main__":
    manolayer = ManoLayer()
