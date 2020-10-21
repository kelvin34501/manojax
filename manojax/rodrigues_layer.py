import jax.numpy as jnp

def quat2mat_jax(quat):
    norm_quat = quat
    norm_quat = norm_quat / jnp.linalg.norm(norm_quat, axis=1, keepdims=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    batch_size = quat.shape[0]

    w2, x2, y2, z2 = jnp.power(w, 2), jnp.power(x, 2), jnp.power(y, 2), jnp.power(z, 2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = jnp.stack(
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


def batch_rodrigues_jax(axisang):
    # axisang N x 3
    axisang_norm = jnp.linalg.norm(axisang + 1e-8, axis=1)
    angle = jnp.expand_dims(axisang_norm, axis=-1)
    axisang_normalized = axisang / angle
    angle = angle * 0.5
    v_cos = jnp.cos(angle)
    v_sin = jnp.sin(angle)
    quat = jnp.concatenate([v_cos, v_sin * axisang_normalized], axis=1)
    rot_mat = quat2mat_jax(quat)
    rot_mat = rot_mat.reshape(-1, 3, 3)
    return rot_mat
