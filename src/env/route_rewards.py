"""Route progress, stagnation costs and ordered intermediate goals for RoboTaxi."""

import heapq
import math
from functools import partial

import equinox
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates
from scipy.ndimage import distance_transform_edt

from env.robotaxi import (
    RoboTaxi,
    RoboTaxiObservation,
    RobotaxiParams,
    RoboTaxiState,
)
from env.robotaxi import (
    get_reward_config as base_reward_config,
)

GRID_RESOLUTION = 4


def build_route_reference(initial, params, fractions):
    """Compute a fixed collision-aware static route and its distance field."""
    height, width = params.map_height_width
    yy, xx = np.mgrid[: (height - 1) * GRID_RESOLUTION + 1, : (width - 1) * GRID_RESOLUTION + 1]
    positions = np.stack((xx, yy), axis=-1) / GRID_RESOLUTION
    obstacles = np.asarray(initial.static_obstacles)
    clearance = np.linalg.norm(positions[:, :, None, :] - obstacles, axis=-1).min(axis=-1)
    legal = clearance >= 1.0 - 1e-4
    distances = np.full(legal.shape, np.inf)
    goal_x, goal_y = np.rint(np.asarray(initial.goal_pos) * GRID_RESOLUTION).astype(int)
    distances[goal_y, goal_x] = 0.0
    queue = [(0.0, goal_y, goal_x)]
    parent = {}
    directions = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    while queue:
        cost, y, x = heapq.heappop(queue)
        if cost > distances[y, x]:
            continue
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < legal.shape[0] and 0 <= nx < legal.shape[1]):
                continue
            if not legal[ny, nx]:
                continue
            if dy and dx and not (legal[y, nx] and legal[ny, x]):
                continue
            candidate = cost + math.hypot(dy, dx) / GRID_RESOLUTION
            if candidate < distances[ny, nx]:
                distances[ny, nx] = candidate
                parent[(ny, nx)] = (y, x)
                heapq.heappush(queue, (candidate, ny, nx))

    start_x, start_y = np.rint(np.asarray(initial.agent_pos) * GRID_RESOLUTION).astype(int)
    if not legal[start_y, start_x] or not np.isfinite(distances[start_y, start_x]):
        raise ValueError("The initial position has no legal static route to the goal")
    node = (start_y, start_x)
    route = [np.asarray(node[::-1]) / GRID_RESOLUTION]
    while node != (goal_y, goal_x):
        node = parent[node]
        route.append(np.asarray(node[::-1]) / GRID_RESOLUTION)
    route = np.asarray(route)
    arclength = np.concatenate(([0.0], np.linalg.norm(np.diff(route, axis=0), axis=1).cumsum()))
    targets = []
    for fraction in fractions:
        position = np.array(
            [np.interp(fraction * arclength[-1], arclength, route[:, axis]) for axis in (0, 1)]
        )
        if np.linalg.norm(obstacles - position, axis=1).min() < 1.0 - 1e-4:
            position = route[np.abs(arclength - fraction * arclength[-1]).argmin()]
        targets.append(tuple(float(value) for value in position))
    missing = ~np.isfinite(distances)
    nearest = distance_transform_edt(missing, return_distances=False, return_indices=True)
    distances[missing] = distances[tuple(nearest[:, missing])]
    grid = tuple(tuple(float(value) for value in row) for row in distances)
    return grid, tuple(targets), route


class RouteRewardParams(RobotaxiParams):
    route_distance_grid: tuple
    use_route_progress: bool
    route_progress_weight: float
    stagnation_rate: float
    stagnation_seconds: float
    subgoal_positions: tuple
    subgoal_bonus_total: float
    subgoal_radius: float
    observe_navigation_target: bool


class RouteRewardState(RoboTaxiState):
    best_route_distance: jax.Array
    steps_without_progress: jax.Array
    subgoal_index: jax.Array


class RouteRewardObservation(RoboTaxiObservation):
    navigation_target: jax.Array

    def as_vector(self):
        return jnp.concatenate([super().as_vector(), self.navigation_target])


def route_distance(position, params):
    grid = jnp.asarray(params.route_distance_grid, dtype=jnp.float32)
    return map_coordinates(
        grid, (position[::-1] * GRID_RESOLUTION)[:, None], order=1, mode="nearest"
    )[0]


def get_reward_config(env_kwargs=None):
    config = env_kwargs or {}
    return dict(
        base_reward_config(),
        version="v4-route-subgoals",
        use_route_progress=config.get("use_route_progress", True),
        route_progress_weight=config.get("route_progress_weight", 0.2),
        stagnation_rate=config.get("stagnation_rate", 0.2),
        stagnation_seconds=config.get("stagnation_seconds", 3.0),
        stagnation_progress_threshold=0.1,
        subgoal_fractions=list(config.get("subgoal_fractions", ())),
        subgoal_bonus_total=config.get("subgoal_bonus_total", 10.0),
        subgoal_radius=config.get("subgoal_radius", 0.75),
        observe_navigation_target=bool(
            config.get("observe_navigation_target", False) or config.get("subgoal_fractions", ())
        ),
        subgoal_reward_once_in_order=True,
        subgoals_terminate_episode=False,
    )


class RouteRewardEnv(RoboTaxi):
    @staticmethod
    def init_params(
        *args,
        use_route_progress=True,
        route_progress_weight=0.2,
        stagnation_rate=0.2,
        stagnation_seconds=3.0,
        subgoal_fractions=(),
        subgoal_bonus_total=10.0,
        subgoal_radius=0.75,
        observe_navigation_target=False,
        **kwargs,
    ):
        fractions = tuple(float(value) for value in subgoal_fractions)
        if any(not 0.0 < value < 1.0 for value in fractions):
            raise ValueError("Subgoal fractions must lie strictly between 0 and 1")
        if any(a >= b for a, b in zip(fractions, fractions[1:], strict=False)):
            raise ValueError("Subgoal fractions must be strictly increasing")
        if not all(
            math.isfinite(value) and value >= 0
            for value in (route_progress_weight, stagnation_rate, subgoal_bonus_total)
        ):
            raise ValueError("Reward weights must be finite and nonnegative")
        if not math.isfinite(stagnation_seconds) or stagnation_seconds <= 0:
            raise ValueError("Stagnation duration must be finite and positive")
        if not math.isfinite(subgoal_radius) or subgoal_radius <= 0:
            raise ValueError("Subgoal radius must be finite and positive")
        params, initial = RoboTaxi.init_params(*args, **kwargs)
        grid, positions, _ = build_route_reference(initial, params, fractions)
        fields = {name: getattr(params, name) for name in params.__dataclass_fields__}
        params = RouteRewardParams(
            **fields,
            route_distance_grid=grid,
            use_route_progress=use_route_progress,
            route_progress_weight=route_progress_weight,
            stagnation_rate=stagnation_rate,
            stagnation_seconds=stagnation_seconds,
            subgoal_positions=positions,
            subgoal_bonus_total=subgoal_bonus_total,
            subgoal_radius=subgoal_radius,
            observe_navigation_target=bool(observe_navigation_target or fractions),
        )
        fields = {name: getattr(initial, name) for name in initial.__dataclass_fields__}
        initial = RouteRewardState(
            **fields,
            best_route_distance=route_distance(initial.agent_pos, params),
            steps_without_progress=jnp.asarray(0),
            subgoal_index=jnp.asarray(0),
        )
        return params, initial

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def reset(key, env_params, init_state):
        if init_state is None:
            raise ValueError("RouteRewardEnv.reset requires an initial state")
        return RouteRewardEnv.get_observation(init_state, env_params), init_state

    @staticmethod
    def get_observation(state, params):
        obs = RoboTaxi.get_observation(state, params)
        return RouteRewardEnv._with_navigation_target(obs, state, params)

    @staticmethod
    def _with_navigation_target(obs, state, params):
        navigation = jnp.zeros((0,), dtype=jnp.float32)
        if params.observe_navigation_target:
            targets = jnp.asarray(params.subgoal_positions + ((0.0, 0.0),), dtype=jnp.float32)
            targets = targets.at[-1].set(state.goal_pos)
            target = targets[state.subgoal_index]
            offset = target - state.agent_pos
            forward = state.agent_forward_dir
            side = jnp.array([-forward[1], forward[0]])
            local = jnp.array([jnp.dot(offset, forward), jnp.dot(offset, side)])
            local = local / jnp.maximum(jnp.linalg.norm(offset), params.perception_radius)
            remaining = (len(params.subgoal_positions) - state.subgoal_index) / max(
                len(params.subgoal_positions), 1
            )
            navigation = jnp.concatenate([local, jnp.atleast_1d(remaining)])
        fields = {name: getattr(obs, name) for name in obs.__dataclass_fields__}
        return RouteRewardObservation(**fields, navigation_target=navigation)

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def step(key, env_state, action, env_params):
        base_obs, state, base_reward, done, info = RoboTaxi.step(key, env_state, action, env_params)
        remaining = route_distance(state.agent_pos, env_params)
        old_remaining = route_distance(env_state.agent_pos, env_params)
        improved = remaining < env_state.best_route_distance - 0.1
        counter = jnp.where(improved, 0, env_state.steps_without_progress + 1)
        best = jnp.where(improved, remaining, env_state.best_route_distance)
        progress_reward = info["progress_reward"]
        if env_params.use_route_progress:
            progress_reward = env_params.route_progress_weight * (old_remaining - remaining)
        stalled = counter >= env_params.stagnation_seconds * env_params.fps
        stagnation_penalty = jnp.where(
            stalled, env_params.stagnation_rate * env_params.step_size, 0.0
        )
        hit = jnp.asarray(False)
        count = len(env_params.subgoal_positions)
        if count:
            positions = jnp.asarray(env_params.subgoal_positions, dtype=jnp.float32)
            target = positions[jnp.minimum(env_state.subgoal_index, count - 1)]
            hit = (
                (env_state.subgoal_index < count)
                & (jnp.linalg.norm(state.agent_pos - target) < env_params.subgoal_radius)
                & ~info["collision_done"]
                & ~info["goal_done"]
            )
        subgoal_reward = hit * (env_params.subgoal_bonus_total / max(count, 1))
        state = equinox.tree_at(
            lambda item: (
                item.best_route_distance,
                item.steps_without_progress,
                item.subgoal_index,
            ),
            state,
            (best, counter, env_state.subgoal_index + hit.astype(jnp.int32)),
        )
        reward = (
            progress_reward
            - info["time_penalty"]
            - info["idle_penalty"]
            - info["speed_penalty"]
            - stagnation_penalty
            + subgoal_reward
        )
        terminal = info["goal_done"] | info["collision_done"]
        reward = jnp.where(terminal, base_reward, reward)
        obs = RouteRewardEnv._with_navigation_target(base_obs, state, env_params)
        info = dict(
            info,
            progress_reward=progress_reward,
            route_distance=remaining,
            stagnation_penalty=stagnation_penalty,
            subgoal_reward=subgoal_reward,
            subgoal_hit=hit,
            subgoals_reached=state.subgoal_index,
        )
        return obs, state, reward, done, info
