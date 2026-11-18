"""Collision-aware navigation references for imitation learning."""

import heapq
import math
from functools import partial

import equinox
import jax
import jax.numpy as jnp
import numpy as np

from env.robotaxi import IDLE_PENALTY_PER_SECOND, IDLE_SPEED_THRESHOLD, SPEED_TRACKING_WEIGHT
from env.route_rewards import RouteRewardEnv, RouteRewardParams

# Base features + three navigation fields. Keep ordering compatible with saved actors.
TRACKING_OBS_DIM = 72
VELOCITY_INDEX = -5
TARGET_SPEED_INDEX = -4
LOOKAHEAD_SLICE = slice(-3, -1)


class TrackingParams(RouteRewardParams):
    reference_points: tuple
    reference_lengths: tuple
    lookahead_distance: float
    cruise_speed: float
    safety_horizon: float
    prediction_clearance: float


def make_reference(state, params, margin):
    """Find a static shortest path with clearance, then store its cumulative lengths."""
    resolution = 4
    height, width = params.map_height_width
    yy, xx = np.mgrid[: (height - 1) * resolution + 1, : (width - 1) * resolution + 1]
    positions = np.stack((xx, yy), axis=-1) / resolution
    clearance = np.linalg.norm(
        positions[:, :, None, :] - np.asarray(state.static_obstacles), axis=-1
    ).min(axis=-1)
    legal = clearance >= margin
    cells = np.argwhere(legal)
    if not len(cells):
        raise ValueError("Obstacle margin leaves no navigable cells")
    world = cells[:, ::-1] / resolution
    start = tuple(cells[np.linalg.norm(world - np.asarray(state.agent_pos), axis=1).argmin()])
    goal = tuple(cells[np.linalg.norm(world - np.asarray(state.goal_pos), axis=1).argmin()])
    distance = np.full(legal.shape, np.inf)
    distance[goal] = 0
    parent = {}
    queue = [(0.0, goal)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost > distance[node]:
            continue
        y, x = node
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            ny, nx = (y + dy, x + dx)
            if not (0 <= ny < legal.shape[0] and 0 <= nx < legal.shape[1] and legal[ny, nx]):
                continue
            if dy and dx and (not (legal[y, nx] and legal[ny, x])):
                continue
            candidate = cost + math.hypot(dy, dx)
            if candidate < distance[ny, nx]:
                distance[ny, nx] = candidate
                parent[ny, nx] = node
                heapq.heappush(queue, (candidate, (ny, nx)))
    if not np.isfinite(distance[start]):
        raise ValueError("No route at the requested obstacle margin")
    points = [np.asarray(state.agent_pos)]
    node = start
    points.append(np.asarray(node[::-1]) / resolution)
    while node != goal:
        node = parent[node]
        points.append(np.asarray(node[::-1]) / resolution)
    points = np.asarray(points)
    keep = np.concatenate(([True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-06))
    points = points[keep]
    lengths = np.concatenate(([0.0], np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()))
    return (tuple(map(tuple, points.tolist())), tuple(lengths.tolist()))


def navigation_reference(state, params):
    """Supply a lookahead point and speed reference without overriding actions."""
    points = jnp.asarray(params.reference_points, dtype=jnp.float32)
    lengths = jnp.asarray(params.reference_lengths, dtype=jnp.float32)
    segments = points[1:] - points[:-1]
    fraction = jnp.clip(
        jnp.sum((state.agent_pos - points[:-1]) * segments, axis=-1)
        / jnp.sum(segments**2, axis=-1),
        0,
        1,
    )
    projections = points[:-1] + fraction[:, None] * segments
    index = jnp.argmin(jnp.linalg.norm(projections - state.agent_pos, axis=-1))
    progress = lengths[index] + fraction[index] * (lengths[index + 1] - lengths[index])
    ahead = jnp.minimum(progress + params.lookahead_distance, lengths[-1])
    target = jnp.array([jnp.interp(ahead, lengths, points[:, i]) for i in (0, 1)])
    offset = target - state.agent_pos
    forward = state.agent_forward_dir
    lateral = jnp.array([-forward[1], forward[0]])
    local = jnp.array([jnp.dot(offset, forward), jnp.dot(offset, lateral)])
    curvature = 2 * local[1] / jnp.maximum(jnp.dot(offset, offset), 0.05)
    speed = params.cruise_speed / (1 + 0.5 * jnp.abs(curvature))
    times = jnp.linspace(0.0, params.safety_horizon, 25)
    future_car = state.agent_pos + times[:, None] * speed * forward
    bounds = jnp.asarray(params.map_height_width[::-1])
    future_obstacles = jnp.mod(
        state.kinematic_obstacles[None, :, :]
        + times[:, None, None] * state.kinematic_obst_velocities[None, :, :],
        bounds,
    )
    separations = jnp.linalg.norm(future_car[:, None, :] - future_obstacles, axis=-1)
    blocked = jnp.any(separations < params.prediction_clearance)
    speed = jnp.where(blocked, 0.0, speed)
    return (local, speed, progress / lengths[-1])


class TrackingEnv(RouteRewardEnv):
    @staticmethod
    def init_params(
        *args,
        tracking_margin=1.25,
        lookahead_distance=1.2,
        cruise_speed=1.0,
        safety_horizon=2.0,
        prediction_clearance=1.3,
        **kwargs,
    ):
        for value in (
            tracking_margin,
            lookahead_distance,
            cruise_speed,
            safety_horizon,
            prediction_clearance,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Tracking parameters must be finite and positive")
        kwargs["observe_navigation_target"] = True
        params, state = RouteRewardEnv.init_params(*args, **kwargs)
        points, lengths = make_reference(state, params, tracking_margin)
        fields = {name: getattr(params, name) for name in params.__dataclass_fields__}
        params = TrackingParams(
            **fields,
            reference_points=points,
            reference_lengths=lengths,
            lookahead_distance=lookahead_distance,
            cruise_speed=cruise_speed,
            safety_horizon=safety_horizon,
            prediction_clearance=prediction_clearance,
        )
        return (params, state)

    @staticmethod
    def get_observation(state, params):
        obs = RouteRewardEnv.get_observation(state, params)
        return TrackingEnv.with_reference(obs, state, params)

    @staticmethod
    def with_reference(obs, state, params):
        local, speed, progress = navigation_reference(state, params)
        return equinox.tree_at(
            lambda item: (item.navigation_target, item.target_speed),
            obs,
            (jnp.concatenate((local / params.perception_radius, jnp.atleast_1d(progress))), speed),
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def reset(key, env_params, init_state):
        return (TrackingEnv.get_observation(init_state, env_params), init_state)

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def step(key, env_state, action, env_params):
        obs, state, reward, done, info = RouteRewardEnv.step(key, env_state, action, env_params)
        obs = TrackingEnv.with_reference(obs, state, env_params)
        speed_error = state.velocity - obs.target_speed
        speed_penalty = SPEED_TRACKING_WEIGHT * speed_error**2 * env_params.step_size
        idle = (state.velocity < IDLE_SPEED_THRESHOLD) & (obs.target_speed > IDLE_SPEED_THRESHOLD)
        idle_penalty = jnp.where(idle, IDLE_PENALTY_PER_SECOND * env_params.step_size, 0.0)
        adjusted = (
            reward + info["speed_penalty"] + info["idle_penalty"] - speed_penalty - idle_penalty
        )
        reward = jnp.where(info["goal_done"] | info["collision_done"], reward, adjusted)
        info = dict(
            info,
            target_speed=obs.target_speed,
            speed_error=speed_error,
            speed_penalty=speed_penalty,
            idle_penalty=idle_penalty,
            idle_penalty_applied=idle,
        )
        return (obs, state, reward, done, info)


def normalized_teacher(observations, params):
    """Compute bounded acceleration/steering labels from the tracking observation.

    The final five fields are velocity, target speed, local lookahead x/y and progress.
    """
    velocity = observations[..., VELOCITY_INDEX]
    desired_speed = observations[..., TARGET_SPEED_INDEX]
    local = observations[..., LOOKAHEAD_SLICE] * params.perception_radius
    curvature = 2 * local[..., 1] / jnp.maximum(jnp.sum(local**2, axis=-1), 0.05)
    steering = jnp.arctan(params.wheelbase_length * curvature)
    limit = jnp.deg2rad(params.max_steering_angle)
    acceleration = 3 * (desired_speed - velocity) + params.friction_coefficient * velocity
    return jnp.stack(
        (
            jnp.clip(acceleration / params.max_acceleration, -1, 1),
            jnp.clip(steering / limit, -1, 1),
        ),
        axis=-1,
    )


def teacher_action(observation, params):
    limits = jnp.array([params.max_acceleration, jnp.deg2rad(params.max_steering_angle)])
    return normalized_teacher(observation.as_vector(), params) * limits
