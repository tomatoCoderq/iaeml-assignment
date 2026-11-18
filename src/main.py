import jax

from env.route_rewards import RouteRewardEnv as RoboTaxi
from utils.autoreset import AutoResetWrapper
from utils.renderer import PygameFrontend


def main():
    """Open the simulator for manual acceleration and steering control."""
    key = jax.random.PRNGKey(0)
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
    frontend = PygameFrontend(env, env_params, init_state, control_mode="vehicle")
    frontend.run()


if __name__ == "__main__":
    main()
