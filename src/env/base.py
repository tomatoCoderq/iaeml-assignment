import jax
from jax import numpy as jnp
import equinox
import chex

from functools import partial
from typing import Tuple, Dict, Any

import numpy as np

from env.tasks import BaseTaskSampler
from utils import utils

# NOTE:
# All colliders are modeled as cicles of radius 1!

np.set_printoptions(linewidth=300, threshold=np.inf)

RADIUS = 1.0  # radius of each circle
NUM_RAY_SENSORS = 16
FOV = jnp.pi  # used to get 360 degree vision
OBSTACLE_COST = 1000  # we are using int16, so choose this carefully
FPS = 60

PATHFINDING_MOVES = jnp.array([[0, 1], [0, -1], [-1, 0], [1, 0]], dtype=jnp.float32)
PATHFINDING_MOVES_DIAG = jnp.array(
    [[1, 1], [-1, -1], [-1, 1], [1, -1]], dtype=jnp.float32
)

# TODO: collision
# TODO: observation. why ray is flat
# TODO: termination
# TODO: reset, autoreset wrapper


# env params
class BaseEnvParams(equinox.Module):
    max_steps_in_episode: int
    fps: int
    step_size: int
    map_height_width: Tuple[int, int]
    # Discritized version of the map. used to find a path
    nav_grid_shape_dtype: Tuple[chex.Shape, chex.ArrayDType]
    path_array_shape_dtype: Tuple[chex.Shape, chex.ArrayDType]
    # Time discretization step size
    discretization_scale: int
    num_ray_sensors: float
    agent_FOV: float
    perception_radius: float


# env state
class BaseEnvState(equinox.Module):
    # Current step
    time: chex.Scalar
    # Positional vector of the goal
    goal_pos: chex.Array
    # Positional vector of agent
    agent_pos: chex.Array
    # Directional vector of agent
    agent_forward_dir: chex.Array
    # Defines along which direction to "dock" the agent
    # agent_goal_dir: chex.Array
    # Positional vectors of static objects
    static_obstacles: chex.Array
    # Positional vectors of kinematic objects
    kinematic_obstacles: chex.Array
    # Vector velocities of kinematic objects
    kinematic_obst_velocities: chex.Array
    # An array of statically defined size that defines waypoints to follow
    path_array: chex.Array


# env observation
class BaseEnvObservation(equinox.Module):
    distance_to_path: chex.Array
    direction_of_path: chex.Array

    # rays
    # goal_ray: chex.Array
    collision_rays: chex.Array

    def as_vector(self) -> chex.Array:
        """Concatenate all components into a single flat observation vector."""
        return jnp.concatenate(
            [
                self.distance_to_path,
                self.direction_of_path,
                # self.goal_ray,
                self.collision_rays,
            ],
            axis=-1,
        )


class BaseEnv:

    # abstract
    def init_params(
        key: chex.PRNGKey,
        map_id: int,
        max_steps: int,
        path_length: int,
        discretization_scale: int = 1,
        perception_radius: float = 5,
        num_ray_sensors: float = NUM_RAY_SENSORS,
        fov: float = FOV,
        fps: float = FPS,
    ) -> Tuple[BaseEnvParams, BaseEnvState]:

        if discretization_scale != 1:
            raise Exception("discretization_scale != 1 is not implemented!")

        task = BaseTaskSampler(map_id)
        # TODO: get map height and width
        env_params = BaseEnvParams(
            max_steps_in_episode=max_steps,
            fps=int(fps),
            step_size=int(1 / fps),
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
        )

        shape, dtype = env_params.path_array_shape_dtype

        convert_to_world_f = partial(
            utils.convert_to_world_view, map_shape=env_params.map_height_width
        )
        convert_to_world_vmap = jax.vmap(convert_to_world_f, in_axes=(0))

        init_state = BaseEnvState(
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
        )
        obstacles = jnp.concatenate(
            [init_state.static_obstacles, init_state.kinematic_obstacles], axis=0
        )
        path_array = BaseEnv._find_path(
            init_state.agent_pos, init_state.goal_pos, obstacles, env_params
        )
        # replace path_array
        init_state = equinox.tree_at(lambda t: t.path_array, init_state, path_array)

        return env_params, init_state

    # abstract
    @partial(jax.jit, static_argnames=("env_params",))
    def reset(
        key: chex.PRNGKey, env_params: BaseEnvParams, init_state: BaseEnvState | None
    ) -> Tuple[chex.Array, BaseEnvState]:

        if init_state is not None:
            state = init_state
        else:
            raise Exception("init_state is None is Not implemented!")
        obs = BaseEnv.get_observation(state, env_params)
        return obs, state

    # abstract
    @partial(jax.jit, static_argnames=("env_params",))
    def step(
        key: chex.PRNGKey,
        env_state: BaseEnvState,
        action: chex.Scalar | chex.Array,
        env_params: BaseEnvParams,
    ) -> Tuple[
        chex.Array, BaseEnvState, chex.Scalar | chex.Array, chex.Array, Dict[Any, Any]
    ]:
        # concatenate static and dynamic obstacles
        obstacles = jnp.concatenate(
            [env_state.static_obstacles, env_state.kinematic_obstacles], axis=0
        )

        # Check Done.
        goal_done = BaseEnv._check_goal(env_state.agent_pos, env_state.goal_pos)
        collision_done = BaseEnv._check_collisions(env_state.agent_pos, obstacles)
        time_done = env_state.time >= env_params.max_steps_in_episode

        done = jnp.logical_or(goal_done, jnp.logical_or(collision_done, time_done))

        # Calculate reward
        true_reward_f = lambda _: 1.0  # Define your own
        false_reward_f = lambda _: 0.0  # Define your own
        reward = jax.lax.cond(goal_done, true_reward_f, false_reward_f, None)
        # NOTE: what I have define above is called sparse reward. You dont have to use a sparse reward

        # Move agent
        new_agent_pos = env_state.agent_pos + action * env_params.step_size
        kinematic_obstacles = BaseEnv._move_kinematic_obstacles(env_state, env_params)

        # Increment time
        new_time = env_state.time + 1

        # path array. update every second.
        pred = jnp.mod(new_time, env_params.fps)
        path_array = jax.lax.cond(
            pred,
            lambda _: BaseEnv._find_path(
                new_agent_pos, env_state.goal_pos, obstacles, env_params
            ),
            lambda _: env_state.path_array,
            None,
        )

        # action without direction
        pred = jnp.linalg.norm(action) < 1e-8

        agent_forward_dir = jax.lax.cond(
            pred,
            lambda _: env_state.agent_forward_dir,
            lambda _: action / (jnp.linalg.norm(action) + 1e-6),
            None,
        )

        new_state = BaseEnvState(
            time=new_time,
            goal_pos=env_state.goal_pos,
            agent_pos=new_agent_pos,
            agent_forward_dir=agent_forward_dir,
            # agent_goal_dir=env_state.agent_goal_dir,
            static_obstacles=env_state.static_obstacles,
            kinematic_obstacles=kinematic_obstacles,
            kinematic_obst_velocities=env_state.kinematic_obst_velocities,
            path_array=path_array,
        )
        obs = BaseEnv.get_observation(new_state, env_params)
        info = {"time": new_time, "distance_to_path": obs.distance_to_path, "direction_of_path": obs.direction_of_path}
        # NOTE: info is used for evaluation

        return obs, new_state, reward, done, info

    def get_observation(
        env_state: BaseEnvState, env_params: BaseEnvParams
    ) -> BaseEnvObservation:
        # find the closest waypoints
        dists = jnp.linalg.norm(env_state.path_array - env_state.agent_pos, axis=1)
        closest_idx = jnp.argmin(dists)
        # construct a vector with the next after closest. it is safe since we always have excess number of waypoints
        path_vec = (
            env_state.path_array.at[closest_idx + 1].get()
            - env_state.path_array.at[closest_idx].get()
        )
        # normalize it
        path_dir = path_vec / (jnp.linalg.norm(path_vec) + 1e-8)

        # project agent's position to that vector to find distance
        anchor = env_state.path_array.at[closest_idx].get()
        rel_vec = env_state.agent_pos - anchor
        proj_len = jnp.dot(rel_vec, path_dir)
        proj_point = anchor + proj_len * path_dir
        distance_to_path = jnp.linalg.norm(env_state.agent_pos - proj_point)

        # perception:
        obstacles = jnp.concatenate(
            [env_state.static_obstacles, env_state.kinematic_obstacles], axis=0
        )
        rays, ray_perceptions = BaseEnv._collision_ray_intersections(
            env_state.agent_pos, env_state.agent_forward_dir, obstacles, env_params
        )
        # rays start at the center of the agent. we need to compensate for that
        ray_perceptions = ray_perceptions
        # rays were normalized. now we need to give them magnitude
        # jax.debug.print("rays before scale {x}", x=rays)
        rays = rays * ray_perceptions[:, None]

        return BaseEnvObservation(
            distance_to_path=distance_to_path,
            direction_of_path=path_dir,
            collision_rays=rays,
        )

    # Protected functions

    def _check_collisions(agent_pos, obstacles, circle_radius=RADIUS) -> chex.Array:
        distances = jnp.linalg.norm(agent_pos - obstacles, axis=1)
        
        is_colliding = distances < circle_radius - 1e-4 # give some room
        return jnp.any(is_colliding)

    def _check_goal(agent_pos, goal_pos, circle_radius=RADIUS) -> chex.Array:
        return jnp.linalg.norm(agent_pos - goal_pos) < circle_radius

    # @partial(jax.jit, static_argnames=("env_params",))
    # def _check_time(env_state: BaseEnvState, env_params: BaseEnvParams) -> chex.Array:
    #     return env_state.time >= env_params.max_steps_in_episode

    @partial(jax.jit, static_argnames=("env_params",))
    def _move_kinematic_obstacles(env_state: BaseEnvState, env_params: BaseEnvParams):
        positions = env_state.kinematic_obstacles
        velocities = env_state.kinematic_obst_velocities
        new_positions = positions + env_params.step_size * velocities

        new_positions = jax.vmap(lambda x, y: jnp.mod(x, y), in_axes=(0, None))(
            new_positions,
            jnp.flip(jnp.asarray(env_params.map_height_width, dtype=jnp.int32)),
        )

        return new_positions

    # ---------------------------
    # Perception
    # ---------------------------

    def _collision_ray_intersections(
        agent_pos, agent_forward_dir, obstacles, env_params: BaseEnvParams
    ):
        rays = BaseEnv._generate_rays(
            forward_vec=agent_forward_dir,
            n_rays=env_params.num_ray_sensors,
            fov=env_params.agent_FOV,
        )

        # predefined number of rays around the agent
        ray_perceptions = BaseEnv._ray_circle_intersections(
            agent_pos, rays, obstacles, RADIUS
        )

        # Apply perception radius limit
        ray_perceptions = jnp.where(
            ray_perceptions <= env_params.perception_radius,
            ray_perceptions,
            env_params.perception_radius,
        )

        return rays, ray_perceptions

    # def _goal_ray_intersections(env_state: BaseEnvState): ...

    @partial(jax.jit, static_argnames=("n_rays", "fov"))
    def _generate_rays(forward_vec: jnp.ndarray, n_rays=NUM_RAY_SENSORS, fov=FOV):
        """
        Generate n_rays directions around forward_ray using pure vector operations.
        forward_ray: (2,) normalized
        returns: (n_rays, 2)
        """
        # 1. Normalize forward vector
        forward_vec = forward_vec / (jnp.linalg.norm(forward_vec) + 1e-8)

        # 2. Angles for rays
        angles = jnp.linspace(-fov, fov, n_rays, endpoint=False)  # (n_rays,)

        # 3. Rays
        rays = jax.vmap(utils.rotate_2d, in_axes=(None, 0))(forward_vec, angles)
        # jax.debug.print("rays={x}", x=rays)

        return rays

    @partial(jax.jit, static_argnames=("circle_radius",))
    def _ray_circle_intersections(origins, dirs, centers, circle_radius=RADIUS):
        """
        origins: (N, 2) or (2,)
        dirs: (N, 2) - unit vectors
        centers: (M, 2)
        perception_radius: float
        circle_radius: float
        returns: (N,) closest intersection distance for each ray
                (inf if no intersection)
        """
        # If single origin
        if origins.ndim == 1:
            origins = jnp.broadcast_to(origins, dirs.shape)  # (N, 2)
        # Shape manipulation for broadcasting
        o = origins[:, None, :]  # (N, 1, 2)
        d = dirs[:, None, :]  # (N, 1, 2)
        c = centers[None, :, :]  # (1, M, 2)

        # Ray-circle intersection
        m = o - c  # (N, M, 2)
        b = jnp.sum(m * d, axis=-1)  # (N, M)
        c_val = jnp.sum(m * m, axis=-1) - circle_radius**2  # (N, M)
        disc = b**2 - c_val

        # Mask invalid (no intersection)
        valid = disc >= 0.0
        sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))

        t1 = -b - sqrt_disc
        t2 = -b + sqrt_disc

        # Take smallest positive intersection per ray–circle pair
        t_candidates = jnp.where(t1 > 0, t1, jnp.where(t2 > 0, t2, jnp.inf))

        # Mask invalid intersections
        t_candidates = jnp.where(valid, t_candidates, jnp.inf)

        # Closest intersection distance per ray
        t_min = jnp.min(t_candidates, axis=1)

        return t_min

    # ---------------------------
    # Path finding, planning
    # ---------------------------

    def _to_grid_pos(vec_map_coord: chex.Array, discretization_scale: int):
        pos = discretization_scale * (vec_map_coord)
        pos = jnp.round(pos).astype(dtype=jnp.int16)
        return pos

    @partial(jax.jit, static_argnames=("env_params",))
    def _construct_navigation_grid(goal_pos, obstacles, env_params: BaseEnvParams):
        """ """

        obstacle_poses = BaseEnv._to_grid_pos(
            obstacles, env_params.discretization_scale
        )
        goal_pos = BaseEnv._to_grid_pos(goal_pos, env_params.discretization_scale)
        goal_pos = jnp.atleast_2d(goal_pos)

        # to map coordinates
        to_map_view_vmap = jax.vmap(
            partial(utils.convert_to_map_view, map_shape=env_params.map_height_width)
        )
        obstacle_poses = to_map_view_vmap(obstacle_poses)
        goal_pos = to_map_view_vmap(goal_pos)

        # create empty navigation grid
        shape, dtype = env_params.nav_grid_shape_dtype
        navigation_grid = jnp.zeros(shape=shape, dtype=jnp.int16)
        V = jnp.zeros_like(navigation_grid, dtype=jnp.int16)

        navigation_grid = navigation_grid.at[
            obstacle_poses[:, 0], obstacle_poses[:, 1]
        ].set(1)
        V = V.at[goal_pos[:, 0], goal_pos[:, 1]].set(1)

        # jax.debug.print("V={V}", V=V)
        # jax.debug.print("mask={x}", x=navigation_grid)

        # put discritized circles (rounded up) on navivation grid around each onstacle pos
        # to have proper navigation grid, we need to set the radius as 2
        # since we are modeling the agent as a point.
        # TODO: make this static somehow
        radius = 2 * env_params.discretization_scale - 1  # Not of jax type, but static!
        # radius = 1 * env_params.discretization_scale  # Not of jax type, but static!

        def make_circle_mask(radius):
            side = 2 * radius + 1
            coords = jnp.arange(side) - radius
            yy, xx = jnp.meshgrid(coords, coords, indexing="ij")
            circle_mask = (xx**2 + yy**2) < (radius**2)
            return circle_mask

        # Generate kernel
        circle_mask = make_circle_mask(radius).astype(
            jnp.int16
        )  # since both V and nav_grid are int16
        ## Follow NCHW formal i.e. (batch, channels, height, width)
        # jax.debug.print("kernel={x}", x=circle_mask)
        kernel = circle_mask[None, None, :, :]
        navigation_grid = navigation_grid[None, None, :, :]
        V = V[None, None, :, :]

        inflated_nav_grid = jax.lax.conv_general_dilated(
            lhs=navigation_grid,
            rhs=kernel,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        inflated_V = jax.lax.conv_general_dilated(
            lhs=V,
            rhs=kernel,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )

        navigation_grid = jnp.reshape(
            jnp.where(inflated_nav_grid, 1, 0).astype(dtype=jnp.bool), shape=shape
        )
        V = jnp.reshape(
            jnp.where(inflated_V, 0, jnp.inf).astype(dtype=dtype), shape=shape
        )

        # jax.debug.print("V={V}", V=V)
        # jax.debug.print("mask={x}", x=navigation_grid)

        return V, navigation_grid

    @partial(jax.jit, static_argnames=("env_params",))
    def _find_path(agent_pos, goal_pos, obstacles, env_params: BaseEnvParams):
        # get navigation grid
        V, mask = BaseEnv._construct_navigation_grid(goal_pos, obstacles, env_params)
        # number of waypoints in the path
        N = max(env_params.path_array_shape_dtype[0])  # Static!

        # solve DP
        def DP(V, mask):
            roll_f = lambda nav, dir: jnp.roll(
                jnp.roll(nav, shift=dir[0], axis=0), shift=dir[1], axis=1
            )
            roll_vmap = jax.vmap(
                roll_f, in_axes=(None, 0), out_axes=-1
            )  # outputs is batched along axis -1 (the last)

            def body_f(i, carry):
                V = carry
                rolled_Vs = roll_vmap(V, PATHFINDING_MOVES)
                rolled_Vs_diag = roll_vmap(V, PATHFINDING_MOVES_DIAG)
                rolled_masks = roll_vmap(mask, PATHFINDING_MOVES)
                rolled_masks_diag = roll_vmap(mask, PATHFINDING_MOVES_DIAG)

                costs = jnp.where(rolled_masks, OBSTACLE_COST, 1)
                costs_diag = jnp.where(rolled_masks_diag, OBSTACLE_COST, jnp.sqrt(2))

                rolled_Vs = jnp.concatenate([rolled_Vs, rolled_Vs_diag], axis=-1)
                costs = jnp.concatenate([costs, costs_diag], axis=-1)

                V = (
                    jnp.minimum(V, jnp.min(rolled_Vs + costs, axis=-1))
                    + mask * OBSTACLE_COST / 10
                )
                return V

            V = jax.lax.fori_loop(0, N, body_f, init_val=(V))

            return V

        # trace the path
        def trace_path(V: chex.Array, start_pos: chex.Array):
            # Add (0,0) move to stay in place!
            moves = jnp.concatenate(
                [PATHFINDING_MOVES, PATHFINDING_MOVES_DIAG, jnp.zeros(shape=(1, 2))],
                axis=0,
            ).astype(jnp.int16)

            start_pos = start_pos.astype(jnp.int16)

            # Pad V with infty not to check for out of border
            # V = jnp.pad(V, pad_width=1, constant_values=jnp.inf)
            # start_pos = start_pos + 1

            def body_f(carry, x):
                pos = carry

                # check all neighbors wrt PATHFINDING_MOVES
                def f(dir, pos):
                    new_pos = pos + dir
                    return new_pos, V.at[new_pos[0], new_pos[1]].get()

                # O-th axis is the batch dim
                poses, get_neighbors = jax.vmap(f, in_axes=(0, None))(moves, pos)

                min_pos_idx = jnp.argmin(get_neighbors, axis=0)
                # return new pos
                new_pos = poses.at[min_pos_idx].get()
                return new_pos, (new_pos)

            _, ys = jax.lax.scan(body_f, start_pos, length=N)

            return ys

        V = DP(V, mask)

        # rotate position to (h, w), discretize and cast to int
        start_pos = BaseEnv._to_grid_pos(
            utils.convert_to_map_view(agent_pos, env_params.map_height_width),
            env_params.discretization_scale,
        )
        path_array = trace_path(V, start_pos)

        # print(V.shape, path_array.shape)
        # jax.debug.print("start_pos={x}", x=start_pos)
        # jax.debug.print("V={V}", V=V.at[start_pos[0], start_pos[1]].get())
        # jax.debug.print("V={V}", V=jnp.round(V).astype(jnp.int16))
        # jax.debug.print("path={x}", x=path_array)

        # rotate path coordinates to (x, y)
        path_array = jax.vmap(utils.convert_to_world_view, in_axes=(0, None))(
            path_array, env_params.map_height_width
        )

        return path_array.astype(dtype=env_params.path_array_shape_dtype[1])
