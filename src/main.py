import jax

from env.base import BaseEnv
from env.robotaxi import RoboTaxi
from utils.renderer import PygameFrontend
from utils.autoreset import AutoResetWrapper


key = jax.random.PRNGKey(0)
# env_params, init_state = BaseEnv.init_params(
#     key=key,
#     map_id=0,
#     max_steps=1000,
#     discretization_scale=1,
#     path_length=100,
#     fps=60,
#     perception_radius=5.0,
#     num_ray_sensors=32,
# )

env_params, init_state = RoboTaxi.init_params(
    key=key,
    map_id=0,
    max_steps=1000,
    discretization_scale=1,
    path_length=100,
    fps=60,
    perception_radius=5.0,
    num_ray_sensors=32,
)

env = AutoResetWrapper(RoboTaxi, env_params, init_state)

# test parallel envs
# TBD

# test renderer
frontend = PygameFrontend(
    env, env_params, init_state, eval_mode=False, control_mode="vehicle"
)
frontend.run()
