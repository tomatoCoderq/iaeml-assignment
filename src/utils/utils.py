from typing import Tuple
import chex
from jax import numpy as jnp

# auto reset


# Rotations


def rotate_2d(vec: chex.Array, theta: float) -> jnp.ndarray:
    """Rotate a 2D vector by `theta` radians counterclockwise."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    R = jnp.array([[c, -s], [s, c]], dtype=vec.dtype)
    return R @ vec


def convert_to_map_view(vec: chex.Array, map_shape: Tuple[int, int]) -> jnp.ndarray:
    h, w = map_shape
    displacement = jnp.array([h - 1, 0])
    directions = jnp.array([-1, 1])
    map_view = displacement + directions*jnp.flip(vec)  # from (x, y) to (h, w)

    return map_view


def convert_to_world_view(vec: chex.Array, map_shape: Tuple[int, int]) -> jnp.ndarray:
    h, w = map_shape
    displacement = jnp.array([0, h - 1])
    directions = jnp.array([1, -1])
    world_view = displacement + directions*jnp.flip(vec)  # from (h, w) to (x, y)

    return world_view
