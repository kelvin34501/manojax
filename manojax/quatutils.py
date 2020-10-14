# from https://github.com/lixiny/bihand/blob/master/bihand/utils/quatutils.py
# modify quat format from x,y,z,w to w,x,y,z
from jax import numpy as np


def normalize_quaternion(quaternion: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    r"""Normalizes a quaternion.
    The quaternion should be in (w, x, y, z) format.

    Args:
        quaternion (torch.Tensor): a tensor containing a quaternion to be
          normalized. The tensor can be of shape :math:`(*, 4)`.
        eps (Optional[bool]): small value to avoid division by zero.
          Default: 1e-12.

    Return:
        torch.Tensor: the normalized quaternion of shape :math:`(*, 4)`.

    Example:
        >>> quaternion = torch.tensor([0., 1., 0., 1.])
        >>> kornia.normalize_quaternion(quaternion)
        tensor([0.000, 0.7071, 0.0000, 0.7071])
    """
    if not isinstance(quaternion, np.ndarray):
        raise TypeError(
            "Input type is not a np.ndarray. Got {}".format(type(quaternion))
        )

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a ndarray of shape (*, 4). Got {}".format(quaternion.shape)
        )
    quat_norm = np.linalg.norm(quaternion, ord=2, axis=-1, keepdims=True)
    quat_norm = np.clip(quat_norm, eps, None)
    return quaternion / quat_norm


def quaternion_inv(q):
    """
    inverse quaternion(s) q
    The quaternion should be in (w, x, y, z) format.
    Expects  tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4

    q_conj = q[..., 1:] * -1.0
    q_conj = np.concatenate((q[..., 0:1], q_conj), axis=-1)
    q_norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q_conj / q_norm


def quaternion_mul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    The quaternion should be in (w, x, y, z) format.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    # terms; ( * , 4, 4)
    terms = np.matmul(r.reshape((-1, 4, 1)), q.reshape((-1, 1, 4)))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] + terms[:, 2, 3] - terms[:, 3, 2]
    y = terms[:, 0, 2] - terms[:, 1, 3] + terms[:, 2, 0] + terms[:, 3, 1]
    z = terms[:, 0, 3] + terms[:, 1, 2] - terms[:, 2, 1] + terms[:, 3, 0]
    return np.stack((w, x, y, z), axis=1).reshape(original_shape)


def quaternion_to_angle_axis(quaternion: np.ndarray) -> np.ndarray:
    """Convert quaternion vector to angle axis of rotation.
    The quaternion should be in (w, x, y, z) format.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = kornia.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not isinstance(quaternion, np.ndarray):
        raise TypeError(
            "Input type is not a np.ndarray. Got {}".format(type(quaternion))
        )

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape Nx4 or 4. Got {}".format(quaternion.shape)
        )
    # unpack input and compute conversion
    q1: np.ndarray = quaternion[..., 1]
    q2: np.ndarray = quaternion[..., 2]
    q3: np.ndarray = quaternion[..., 3]
    sin_squared_theta: np.ndarray = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: np.ndarray = np.sqrt(sin_squared_theta)
    cos_theta: np.ndarray = quaternion[..., 0]
    two_theta: np.ndarray = 2.0 * np.where(
        cos_theta < 0.0,
        np.arctan2(-sin_theta, -cos_theta),
        np.arctan2(sin_theta, cos_theta),
    )

    k_pos: np.ndarray = two_theta / sin_theta
    k_neg: np.ndarray = 2.0 * np.ones_like(sin_theta)
    k: np.ndarray = np.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: np.ndarray = np.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def angle_axis_to_quaternion(angle_axis: np.ndarray) -> np.ndarray:
    r"""Convert an angle axis to a quaternion.
    The quaternion vector has components in (w, x, y, z) format.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        angle_axis (torch.Tensor): tensor with angle axis.

    Return:
        torch.Tensor: tensor with quaternion.

    Shape:
        - Input: :math:`(*, 3)` where `*` means, any number of dimensions
        - Output: :math:`(*, 4)`

    Example:
        >>> angle_axis = torch.rand(2, 4)  # Nx4
        >>> quaternion = kornia.angle_axis_to_quaternion(angle_axis)  # Nx3
    """
    if not isinstance(angle_axis, np.ndarray):
        raise TypeError(
            "Input type is not a np.ndarray. Got {}".format(type(angle_axis))
        )

    if not angle_axis.shape[-1] == 3:
        raise ValueError(
            "Input must be a tensor of shape Nx3 or 3. Got {}".format(angle_axis.shape)
        )
    # unpack input and compute conversion
    a0: np.ndarray = angle_axis[..., 0:1]
    a1: np.ndarray = angle_axis[..., 1:2]
    a2: np.ndarray = angle_axis[..., 2:3]
    theta_squared: np.ndarray = a0 * a0 + a1 * a1 + a2 * a2

    theta: np.ndarray = np.sqrt(theta_squared)
    half_theta: np.ndarray = theta * 0.5

    mask: np.ndarray = theta_squared > 0.0
    ones: np.ndarray = np.ones_like(half_theta)

    k_neg: np.ndarray = 0.5 * ones
    k_pos: np.ndarray = np.sin(half_theta) / theta
    k: np.ndarray = np.where(mask, k_pos, k_neg)
    w: np.ndarray = np.where(mask, np.cos(half_theta), ones)

    quaternion: np.ndarray = np.zeros_like(angle_axis)
    quaternion[..., 0:1] += a0 * k
    quaternion[..., 1:2] += a1 * k
    quaternion[..., 2:3] += a2 * k
    return np.concatenate([w, quaternion], axis=-1)


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    r"""Convert a quaternion to a rotation matrix.
    The quaternion vector has components in (w, x, y, z) format.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternion.

    Return:
        torch.Tensor: tensor with quaternion.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3, 3)`

    Example:
        >>> q = torch.rand(2, 4)  # Nx4
        >>> rotmat = quaternion_to_rotation_matrix(q)  # Nx3x3
    """
    original_shape = quaternion.shape  # (*, 4)
    asterisk_shape = original_shape[:-1]  # (*, )
    # split cols of q
    w, x, y, z = (
        quaternion[..., 0],
        quaternion[..., 1],
        quaternion[..., 2],
        quaternion[..., 3],
    )
    # convenient terms
    ww, wx, wy, wz = w * w, w * x, w * y, w * z
    xx, xy, xz = x * x, x * y, x * z
    yy, yz = y * y, y * z
    zz = z * z
    # compute normalizer
    q_norm_squared = ww + xx + yy + zz  # (*, )
    q_norm_squared = q_norm_squared.unsqueeze(-1)  # (*, 1) for broadcasting
    # stack
    rotation_matrix = np.stack(
        (
            ww + xx - yy - zz,  # (0, 0)
            2 * (xy - wz),  # (0, 1)
            2 * (wy + xz),  # (0, 2)
            2 * (wz + xy),  # (1, 0)
            ww - xx + yy - zz,  # (1, 1)
            2 * (yz - wx),  # (1, 2)
            2 * (xz - wy),  # (2, 0)
            2 * (wx + yz),  # (2, 1)
            ww - xx - yy + zz,  # (2, 2)
        ),
        axis=-1,
    )  # (*, 9)
    # normalize
    rotation_matrix = rotation_matrix / q_norm_squared  # (*, 9)
    # reshape
    target_shape = tuple(list(asterisk_shape) + [3, 3])  # value = (*, 3, 3)
    rotation_matrix = rotation_matrix.reshape(target_shape)

    return rotation_matrix


def quaternion_norm(quaternion):
    r"""Computes norm of quaternion.
    The quaternion should be in (w, x, y, z) format.

    Args:
        quaternion (torch.Tensor): a tensor containing a quaternion to be
          normalized. The tensor can be of shape :math:`(*, 4)`.

    Return:
        torch.Tensor: the norm of shape :math:`(*)`.

    Example:
        >>> quaternion = torch.tensor([0., 1., 0., 1.])
        >>> quaternion_norm(quaternion)
        tensor(1.4142)
    """
    return np.sqrt(np.sum(np.power(quaternion, 2), axis=-1))


def quaternion_norm_squared(quaternion):
    r"""Computes norm of quaternion.
    The quaternion should be in (w, x, y, z) format.

    Args:
        quaternion (torch.Tensor): a tensor containing a quaternion to be
          normalized. The tensor can be of shape :math:`(*, 4)`.

    Return:
        torch.Tensor: the squared norm of shape :math:`(*)`.

    Example:
        >>> quaternion = torch.tensor([0., 1., 0., 1.])
        >>> quaternion_norm(quaternion)
        tensor(2.0)
    """
    return np.sum(np.power(quaternion, 2), axis=-1)


def quaternion_to_angle(quaternion: np.ndarray) -> np.ndarray:
    """Convert quaternion vector to angle of rotation.
    The quaternion should be in (w, x, y, z) format.

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*)`
    """
    if not isinstance(quaternion, np.ndarray):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(quaternion))
        )

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape Nx4 or 4. Got {}".format(quaternion.shape)
        )
    # unpack input and compute conversion
    q1: np.ndarray = quaternion[..., 1]
    q2: np.ndarray = quaternion[..., 2]
    q3: np.ndarray = quaternion[..., 3]
    sin_squared_theta: np.ndarray = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: np.ndarray = np.sqrt(sin_squared_theta)
    cos_theta: np.ndarray = quaternion[..., 0]
    two_theta: np.ndarray = 2.0 * np.where(
        cos_theta < 0.0,
        np.arctan2(-sin_theta, -cos_theta),
        np.arctan2(sin_theta, cos_theta),
    )

    return two_theta
