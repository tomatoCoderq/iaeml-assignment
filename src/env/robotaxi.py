from functools import partial

import equinox
from env.base import BaseEnv, BaseEnvParams, BaseEnvState, BaseEnvObservation

import jax
from jax import numpy as jnp
import chex

from typing import Any, Dict, Tuple

from env.tasks import BaseTaskSampler
from utils import utils

RADIUS = 1.0  # radius of each circle
NUM_RAY_SENSORS = 16
FOV = jnp.pi  # used to get 360 degree vision
OBSTACLE_COST = 1000  # we are using int16, so choose this carefully
FPS = 60

# Reward v3. Rates are per simulated second; distances use world units.
# Keep progress symmetric: paying more for approach than retreat would reward loops.
REWARD_VERSION = "v3"
PROGRESS_REWARD_WEIGHT = 0.2        # was 2.0: 10x weaker distance shaping
TIME_PENALTY_PER_SECOND = 0.01     # was 0.1: 10x smaller movement/time cost
IDLE_SPEED_THRESHOLD = 0.05        # forward speed below this counts as standing
IDLE_PENALTY_PER_SECOND = 3.0      # extra idle cost; at 60 FPS this is 0.05/step
GOAL_REWARD = 100.0
COLLISION_REWARD = -100.0
TARGET_CRUISE_SPEED = 1.0
TARGET_TURN_SPEED = 0.3
SPEED_TRACKING_WEIGHT = 0.2        # penalty = 0.2 * (v - target_speed)^2 * dt
TURN_SLOW_ANGLE_DEG = 30.0
TURN_SHARP_ANGLE_DEG = 60.0
FRONT_CONE_HALF_ANGLE_DEG = 30.0
OBSTACLE_STOP_DISTANCE = 0.25      # ray clearance to obstacle boundary
OBSTACLE_SLOW_DISTANCE = 1.5
GOAL_STOP_DISTANCE = 0.5           # inside the existing goal radius (1.0)
GOAL_SLOW_DISTANCE = 3.0


def get_reward_config():
    """Serializable description of the exact reward used in this code version."""
    return {
        "version": REWARD_VERSION,
        "progress_reward_weight": PROGRESS_REWARD_WEIGHT,
        "time_penalty_per_second": TIME_PENALTY_PER_SECOND,
        "idle_speed_threshold": IDLE_SPEED_THRESHOLD,
        "idle_penalty_per_second": IDLE_PENALTY_PER_SECOND,
        "goal_reward": GOAL_REWARD,
        "collision_reward": COLLISION_REWARD,
        "target_cruise_speed": TARGET_CRUISE_SPEED,
        "target_turn_speed": TARGET_TURN_SPEED,
        "speed_tracking_weight": SPEED_TRACKING_WEIGHT,
        "turn_slow_angle_deg": TURN_SLOW_ANGLE_DEG,
        "turn_sharp_angle_deg": TURN_SHARP_ANGLE_DEG,
        "front_cone_half_angle_deg": FRONT_CONE_HALF_ANGLE_DEG,
        "obstacle_stop_distance": OBSTACLE_STOP_DISTANCE,
        "obstacle_slow_distance": OBSTACLE_SLOW_DISTANCE,
        "goal_stop_distance": GOAL_STOP_DISTANCE,
        "goal_slow_distance": GOAL_SLOW_DISTANCE,
        "idle_penalty_only_when_target_exceeds_idle_threshold": True,
        "terminal_rewards_override_dense_reward": True,
        "collision_has_priority_over_goal": True,
    }


PATHFINDING_MOVES = jnp.array([[0, 1], [0, -1], [-1, 0], [1, 0]], dtype=jnp.float32)
PATHFINDING_MOVES_DIAG = jnp.array(
    [[1, 1], [-1, -1], [-1, 1], [1, -1]], dtype=jnp.float32
)


class RobotaxiParams(BaseEnvParams):
    step_size: float
    wheelbase_length: float
    friction_coefficient: float  # linear velocity damping coefficient, 1/s
    mass: int
    max_acceleration: float  # максимальное ускорение, ограничено моторами
    max_steering_angle: float  # максимальный угол поворота руля, градусы


class RoboTaxiState(BaseEnvState):
    # это состояние такси и окружения, которое нужно для симуляции
    velocity: chex.Array  # TODO: здесь одно число или массив?


class RoboTaxiObservation(BaseEnvObservation):
    # сверху храним наблюдение такси, те его данные о мире вокруг и себе
    velocity: chex.Array  # TODO: здесь одно число или массив?
    target_speed: chex.Array

    def as_vector(self) -> chex.Array:
        return jnp.concatenate([
            jnp.ravel(self.distance_to_path),
            jnp.ravel(self.direction_of_path),
            jnp.ravel(self.collision_rays),
            jnp.ravel(self.velocity),
            jnp.ravel(self.target_speed),
        ])


class RoboTaxi(BaseEnv):

    def init_params(
        key: chex.PRNGKey,
        map_id: int,
        max_steps: int,
        path_length: int,
        discretization_scale: int = 1,
        perception_radius: float = 5,
        num_ray_sensors: float = NUM_RAY_SENSORS,
        fov: float = FOV,
        fps: float = 60,
        wheelbase_length: float = 1.0,
        friction_coefficient: float = 0.7,
        mass: int = 150,
        max_acceleration: float = 3.0,
        max_steering_angle: float = 40.0,
    ) -> Tuple[RobotaxiParams, RoboTaxiState]:
        task = BaseTaskSampler(map_id=map_id)

        env_params = RobotaxiParams(
            max_steps_in_episode=max_steps,
            fps=int(fps),
            step_size=1.0 / fps,
            map_height_width=(task.height, task.width),
            nav_grid_shape_dtype=(
                (task.height * discretization_scale, task.width * discretization_scale),
                jnp.float32,
            ),
            path_array_shape_dtype=((path_length, 2), jnp.float32),
            discretization_scale=discretization_scale,
            perception_radius=perception_radius,
            agent_FOV=fov,
            num_ray_sensors=num_ray_sensors,
            wheelbase_length=wheelbase_length,
            friction_coefficient=friction_coefficient,
            mass=mass,
            max_acceleration=max_acceleration,
            max_steering_angle=max_steering_angle,
        )

        shape, dtype = env_params.path_array_shape_dtype
        convert_to_world_f = partial(
            utils.convert_to_world_view, map_shape=(task.height, task.width)
        )
        convert_to_world_vmap = jax.vmap(convert_to_world_f, in_axes=(0))

        init_state = RoboTaxiState(
            time=jnp.asarray(0),
            goal_pos=convert_to_world_f(task.goal_pos).astype(jnp.float32),
            agent_pos=convert_to_world_f(task.agent_pos).astype(jnp.float32),
            agent_forward_dir=task.agent_forward_dir.astype(jnp.float32),
            # agent_goal_dir=task.agent_goal_dir,
            static_obstacles=convert_to_world_vmap(task.static_obstacles).astype(
                jnp.float32
            ),
            kinematic_obstacles=convert_to_world_vmap(task.kinematic_obstacles).astype(
                jnp.float32
            ),
            kinematic_obst_velocities=task.kinematic_obst_velocities.astype(
                jnp.float32
            ),
            path_array=jnp.zeros(shape=shape, dtype=dtype),
            velocity=jnp.array(0.0, dtype=jnp.float32),
        )

        obstacles = jnp.concatenate(
            [init_state.static_obstacles, init_state.kinematic_obstacles], axis=0
        )

        path_array = RoboTaxi._find_path(
            init_state.agent_pos, init_state.goal_pos, obstacles, env_params
        )

        init_state = equinox.tree_at(
            lambda state: state.path_array, init_state, path_array
        )

        return env_params, init_state

    @partial(jax.jit, static_argnames=("env_params",))
    def reset(
        key: chex.PRNGKey,
        env_params: RobotaxiParams,
        init_state: RoboTaxiState | None,
    ) -> Tuple[RoboTaxiObservation, RoboTaxiState]:
        if init_state is None:
            raise ValueError("RoboTaxi.reset requires an initial state")
        return RoboTaxi.get_observation(init_state, env_params), init_state

    @partial(jax.jit, static_argnames=("env_params",))
    def step(
        key: chex.PRNGKey,
        env_state: RoboTaxiState,
        action: chex.Scalar | chex.Array,
        env_params: RobotaxiParams,
    ) -> Tuple[
        RoboTaxiObservation, RoboTaxiState, chex.Scalar, chex.Array, Dict[Any, Any]
    ]:
        """Advance the world and return obs, state, reward, done, info."""
        # Update the vehicle and obstacles over the same time interval.
        new_state = RoboTaxi._simulate_dynamics(env_state, action, env_params)
        kinematic_obstacles = BaseEnv._move_kinematic_obstacles(
            env_state, env_params
        )
        new_time = env_state.time + 1

        # World coordinates are (x, y), whereas map dimensions are (h, w).
        height, width = env_params.map_height_width
        upper_bound = jnp.array([width - 1, height - 1])
        out_of_bounds = jnp.any(
            (new_state.agent_pos < 0.0) | (new_state.agent_pos > upper_bound)
        )
        new_agent_pos = jnp.clip(new_state.agent_pos, 0.0, upper_bound)
        new_state = equinox.tree_at(
            lambda state: (state.time, state.agent_pos, state.kinematic_obstacles),
            new_state,
            (new_time, new_agent_pos, kinematic_obstacles),
        )

        # Evaluate terminal events after moving both the vehicle and obstacles.
        obstacles = jnp.concatenate(
            [new_state.static_obstacles, new_state.kinematic_obstacles],
            axis=0,
        )
        goal_done = BaseEnv._check_goal(
            new_state.agent_pos, new_state.goal_pos
        )
        collision_done = BaseEnv._check_collisions(
            new_state.agent_pos, obstacles
        ) | out_of_bounds
        time_done = new_time >= env_params.max_steps_in_episode
        done = goal_done | collision_done | time_done

        # Replan once per simulated second, while the episode is active.
        should_replan = (jnp.mod(new_time, env_params.fps) == 0) & ~done
        path_array = jax.lax.cond(
            should_replan,
            lambda _: BaseEnv._find_path(
                new_state.agent_pos, new_state.goal_pos, obstacles, env_params
            ),
            lambda _: env_state.path_array,
            None,
        )
        new_state = equinox.tree_at(
            lambda state: state.path_array, new_state, path_array
        )

        obs = RoboTaxi.get_observation(new_state, env_params)
        old_distance = jnp.linalg.norm(env_state.agent_pos - env_state.goal_pos)
        new_distance = jnp.linalg.norm(new_state.agent_pos - new_state.goal_pos)
        progress = old_distance - new_distance
        progress_reward = PROGRESS_REWARD_WEIGHT * progress
        time_penalty = TIME_PENALTY_PER_SECOND * env_params.step_size
        is_idle = new_state.velocity < IDLE_SPEED_THRESHOLD
        idle_penalty_applied = is_idle & (obs.target_speed > IDLE_SPEED_THRESHOLD)
        idle_penalty = jnp.where(
            idle_penalty_applied, IDLE_PENALTY_PER_SECOND * env_params.step_size, 0.0
        )
        speed_error = new_state.velocity - obs.target_speed
        speed_penalty = SPEED_TRACKING_WEIGHT * jnp.square(speed_error) * env_params.step_size
        reward = progress_reward - time_penalty - idle_penalty - speed_penalty
        # Fixed terminal rewards override all dense terms; collision has priority.
        reward = jnp.where(
            collision_done, COLLISION_REWARD, jnp.where(goal_done, GOAL_REWARD, reward)
        )
        info = {
            "time": new_time,
            "distance_to_path": obs.distance_to_path,
            "direction_of_path": obs.direction_of_path,
            "velocity": new_state.velocity,
            "progress": progress,
            "progress_reward": progress_reward,
            "time_penalty": time_penalty,
            "idle_penalty": idle_penalty,
            "is_idle": is_idle,
            "target_speed": obs.target_speed,
            "speed_error": speed_error,
            "speed_penalty": speed_penalty,
            "idle_penalty_applied": idle_penalty_applied,
            "goal_done": goal_done,
            "collision_done": collision_done,
            "time_done": time_done,
            "out_of_bounds": out_of_bounds,
        }
        return obs, new_state, reward, done, info

    def get_observation(
        env_state: RoboTaxiState, env_params: RobotaxiParams
    ) -> RoboTaxiObservation:
        dists = jnp.linalg.norm(env_state.path_array - env_state.agent_pos, axis=1)
        closest_idx = jnp.argmin(dists)
        next_idx = jnp.minimum(closest_idx + 1, env_state.path_array.shape[0] - 1)
        anchor = env_state.path_array[closest_idx]
        path_vec = env_state.path_array[next_idx] - anchor
        path_dir = path_vec / (jnp.linalg.norm(path_vec) + 1e-8)
        rel_vec = env_state.agent_pos - anchor
        proj_point = anchor + jnp.dot(rel_vec, path_dir) * path_dir
        distance_to_path = jnp.linalg.norm(env_state.agent_pos - proj_point)

        obstacles = jnp.concatenate(
            [env_state.static_obstacles, env_state.kinematic_obstacles], axis=0
        )
        rays, ray_perceptions = BaseEnv._collision_ray_intersections(
            env_state.agent_pos, env_state.agent_forward_dir, obstacles, env_params
        )
        collision_rays = rays * ray_perceptions[:, None]
        target_speed = RoboTaxi._get_target_speed(env_state, env_params, collision_rays)
        return RoboTaxiObservation(
            distance_to_path=distance_to_path,
            direction_of_path=path_dir,
            collision_rays=collision_rays,
            velocity=env_state.velocity,
            target_speed=target_speed,
        )

    def _get_target_speed(env_state, env_params, collision_rays):
        """A reward/reference speed, not an override of the actor's action.

        Side/rear obstacles do not force a stop. Slow down for the direction
        toward a waypoint two entries ahead, obstacles within a forward cone,
        and the goal. The goal's zero-speed point lies inside its success radius.
        """
        closest = jnp.argmin(jnp.linalg.norm(
            env_state.path_array - env_state.agent_pos, axis=-1
        ))
        lookahead = jnp.minimum(closest + 2, env_state.path_array.shape[0] - 1)
        desired = env_state.path_array[lookahead] - env_state.agent_pos
        desired = jnp.where(
            jnp.linalg.norm(desired) > 1e-6, desired,
            env_state.goal_pos - env_state.agent_pos,
        )
        desired = desired / (jnp.linalg.norm(desired) + 1e-8)
        forward = env_state.agent_forward_dir
        forward = forward / (jnp.linalg.norm(forward) + 1e-8)
        angle = jnp.arccos(jnp.clip(jnp.dot(desired, forward), -1.0, 1.0))
        turn_fraction = jnp.clip(
            (angle - jnp.deg2rad(TURN_SLOW_ANGLE_DEG))
            / jnp.deg2rad(TURN_SHARP_ANGLE_DEG - TURN_SLOW_ANGLE_DEG), 0.0, 1.0
        )
        turn_limit = TARGET_CRUISE_SPEED - (
            TARGET_CRUISE_SPEED - TARGET_TURN_SPEED
        ) * turn_fraction

        angles = jnp.linspace(
            -env_params.agent_FOV, env_params.agent_FOV,
            env_params.num_ray_sensors, endpoint=False,
        )
        forward_rays = jnp.abs(angles) <= jnp.deg2rad(FRONT_CONE_HALF_ANGLE_DEG)
        distances = jnp.linalg.norm(collision_rays, axis=-1)
        front_clearance = jnp.min(jnp.where(
            forward_rays, distances, env_params.perception_radius
        ))
        obstacle_limit = TARGET_CRUISE_SPEED * jnp.clip(
            (front_clearance - OBSTACLE_STOP_DISTANCE)
            / (OBSTACLE_SLOW_DISTANCE - OBSTACLE_STOP_DISTANCE), 0.0, 1.0
        )
        goal_distance = jnp.linalg.norm(env_state.goal_pos - env_state.agent_pos)
        goal_limit = TARGET_CRUISE_SPEED * jnp.clip(
            (goal_distance - GOAL_STOP_DISTANCE)
            / (GOAL_SLOW_DISTANCE - GOAL_STOP_DISTANCE), 0.0, 1.0
        )
        return jnp.minimum(turn_limit, jnp.minimum(obstacle_limit, goal_limit))

    def _simulate_dynamics(
        env_state: RoboTaxiState,
        action: chex.Scalar | chex.Array,
        env_params: RobotaxiParams,
    ) -> RoboTaxiState:
        """Apply one explicit Euler step of the kinematic bicycle model.

        action has shape (2,): acceleration in world units/s^2 and steering
        angle in radians. max_steering_angle is configured in degrees.
        velocity is a scalar forward speed; reverse driving is disabled.
        This method only updates the vehicle; the caller updates the world.
        """
        dt = env_params.step_size
        acceleration, steering_angle = action

        acceleration = jnp.clip(
            acceleration,
            -env_params.max_acceleration,
            env_params.max_acceleration,
        )
        steering_limit = jnp.deg2rad(env_params.max_steering_angle)
        steering_angle = jnp.clip(
            steering_angle, -steering_limit, steering_limit
        )

        speed = env_state.velocity
        heading = jnp.arctan2(
            env_state.agent_forward_dir[1],
            env_state.agent_forward_dir[0],
        )

        new_speed = jnp.maximum(
            0.0,
            speed
            + (acceleration - env_params.friction_coefficient * speed) * dt,
        )
        new_heading = heading + (
            speed / env_params.wheelbase_length
        ) * jnp.tan(steering_angle) * dt
        new_position = env_state.agent_pos + speed * jnp.array(
            [jnp.cos(heading), jnp.sin(heading)]
        ) * dt
        new_forward_dir = jnp.array(
            [jnp.cos(new_heading), jnp.sin(new_heading)]
        )

        return equinox.tree_at(
            lambda state: (
                state.agent_pos,
                state.agent_forward_dir,
                state.velocity,
            ),
            env_state,
            (new_position, new_forward_dir, new_speed),
        )
