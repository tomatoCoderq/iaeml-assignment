"""Vector rotation and conversion between world (x, y) and map (row, column)."""

import chex
from jax import numpy as jnp


def rotate_2d(vec: chex.Array, theta: float) -> jnp.ndarray:
    """Rotate a 2D vector by `theta` radians counterclockwise."""
    cosine, sine = jnp.cos(theta), jnp.sin(theta)
    rotation = jnp.array([[cosine, -sine], [sine, cosine]], dtype=vec.dtype)
    return rotation @ vec


def convert_to_map_view(vec: chex.Array, map_shape: tuple[int, int]) -> jnp.ndarray:
    """Convert world (x, y) into map (row, column), with row zero at the top."""
    height, _ = map_shape
    displacement = jnp.array([height - 1, 0])
    directions = jnp.array([-1, 1])
    map_view = displacement + directions * jnp.flip(vec)  # from (x, y) to (h, w)

    return map_view


def convert_to_world_view(vec: chex.Array, map_shape: tuple[int, int]) -> jnp.ndarray:
    """Convert map (row, column) into world (x, y), with y increasing upward."""
    height, _ = map_shape
    displacement = jnp.array([0, height - 1])
    directions = jnp.array([1, -1])
    world_view = displacement + directions * jnp.flip(vec)  # from (h, w) to (x, y)

    return world_view
