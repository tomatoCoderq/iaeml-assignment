import chex
import equinox
import jax

from env.base import BaseEnv, BaseEnvParams, BaseEnvState


class AutoResetWrapper(equinox.Module):
    env: BaseEnv = equinox.field(static=True)
    env_params: BaseEnvParams = equinox.field(static=True)
    init_state: BaseEnvState

    @equinox.filter_jit
    def reset(
        self,
        key: chex.PRNGKey,
        *args,
    ) -> tuple[chex.Array, BaseEnvState]:
        # Initial state arrays remain dynamic PyTree leaves for JAX.
        obs, state = self.env.reset(key, self.env_params, self.init_state)
        return obs, state

    @equinox.filter_jit
    def step(
        self,
        key: chex.PRNGKey,
        state: BaseEnvState,
        action: chex.Array,
        *args,
    ) -> tuple[chex.Array, BaseEnvState, chex.Scalar, chex.Array, dict]:
        """
        Returns: obs, state, reward, done, info
        But auto-resets if done=True
        """
        obs, new_state, reward, done, info = self.env.step(key, state, action, self.env_params)

        def reset_state(_):
            if "goal_done" in info:
                jax.debug.print(
                    "Autoreset: step={step}, goal={goal}, collision={collision}, "
                    "timeout={timeout}, out_of_bounds={bounds}",
                    step=info["time"],
                    goal=info["goal_done"],
                    collision=info["collision_done"],
                    timeout=info["time_done"],
                    bounds=info.get("out_of_bounds", False),
                )
            else:
                jax.debug.print(
                    "Autoreset: step={step}, reward={reward}",
                    step=new_state.time,
                    reward=reward,
                )
            new_obs, new_st = self.reset(key, self.env_params, self.init_state)
            # Preserve the terminal reward/info while returning the reset observation.
            return new_obs, new_st, reward, info

        # If done, reset
        obs, new_state, reward, info = jax.lax.cond(
            done,
            reset_state,
            lambda _: (obs, new_state, reward, info),
            operand=None,
        )

        return obs, new_state, reward, done, info
