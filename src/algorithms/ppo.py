"""Train/evaluate RoboTaxi PPO: PYTHONPATH=src python -m algorithms.ppo."""
import argparse
import json
import time
from pathlib import Path

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx, serialization
from tqdm import tqdm

from env.robotaxi import RoboTaxi, get_reward_config

# Local CPU defaults. Large accelerator runs can override these sizes.
TIME_STEPS = 100_000
NUM_ENVS = 16
BATCH_SIZE = 256
UNROLL_LENGTH = 128
NUM_EPOCHS = 4
DISCOUNT = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01  # actor loss = PPO surrogate loss - 0.01 * H(tanh(Normal))
MAX_GRAD_NORM = 1.0
CKPT_DIR = "checkpoints"
SEED = 67

DEFAULT_ENV_KWARGS = dict(
    map_id=0, max_steps=1000, path_length=100, discretization_scale=1,
    perception_radius=5.0, num_ray_sensors=32, fov=float(np.pi), fps=60,
    wheelbase_length=1.0, friction_coefficient=0.7, mass=150,
    max_acceleration=3.0, max_steering_angle=40.0,
)
DEFAULT_OBS_DIM = 5 + 2 * DEFAULT_ENV_KWARGS["num_ray_sensors"]  # includes target speed


class Actor(nnx.Module):
    def __init__(self, obs_dim, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(obs_dim, 256, rngs=rngs)
        self.dense2 = nnx.Linear(256, 128, rngs=rngs)
        self.dense3 = nnx.Linear(
            128, 4, kernel_init=nnx.initializers.orthogonal(0.01), rngs=rngs
        )

    def __call__(self, x):
        x = jax.nn.relu(self.dense1(x))
        x = jax.nn.relu(self.dense2(x))
        mean, log_std = jnp.split(self.dense3(x), 2, axis=-1)
        return mean, jnp.exp(jnp.clip(log_std, -5.0, 2.0))

    @nnx.jit
    def sample_action(self, x, rng):
        mean, std = self(x)
        raw = mean + std * jax.random.normal(rng, mean.shape, dtype=mean.dtype)
        return jnp.tanh(raw), raw

    def log_prob(self, raw_action, loc, scale):
        normal = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale)
        _, log_det = distrax.Block(distrax.Tanh(), ndims=1).forward_and_log_det(raw_action)
        return (normal.log_prob(raw_action) - log_det)[..., None]

    def entropy(self, loc, scale, key):
        """One reparameterized Monte Carlo sample per observation; shape (..., 1).

        H(tanh(U)) = H(U) + E[log|det J_tanh(U)|]. This regularizes the
        normalized bounded actions, not the unbounded Gaussian alone.
        """
        raw = loc + scale * jax.random.normal(key, loc.shape, dtype=loc.dtype)
        normal = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale)
        _, log_det = distrax.Block(distrax.Tanh(), ndims=1).forward_and_log_det(raw)
        return (normal.entropy() + log_det)[..., None]


class Critic(nnx.Module):
    def __init__(self, obs_dim, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(obs_dim, 256, rngs=rngs)
        self.dense2 = nnx.Linear(256, 128, rngs=rngs)
        self.dense3 = nnx.Linear(128, 1, rngs=rngs)

    def __call__(self, x, rngs=None):
        x = jax.nn.relu(self.dense1(x))
        x = jax.nn.relu(self.dense2(x))
        return self.dense3(x)


def scale_action(action, params):
    """Normalized [-1, 1] -> acceleration and steering in radians."""
    limits = jnp.array([params.max_acceleration, jnp.deg2rad(params.max_steering_angle)])
    return jnp.clip(action, -1.0, 1.0) * limits


@jax.jit
def compute_advantage_and_target(
    values, next_values, rewards, dones, terminated,
    discount=DISCOUNT, gae_lambda=GAE_LAMBDA,
):
    """All inputs have shape (T, B). Bootstrap timeouts, stop GAE at resets."""
    deltas = rewards + discount * (~terminated) * next_values - values

    def backward(next_advantage, data):
        delta, done = data
        advantage = delta + discount * gae_lambda * (~done) * next_advantage
        return advantage, advantage

    _, advantages = jax.lax.scan(
        backward, jnp.zeros_like(values[0]), (deltas, dones), reverse=True
    )
    return advantages, advantages + values


def actor_loss_fn(actor, batch, advantages, key):
    mu, std = actor(batch["obs"])
    log_probs = actor.log_prob(batch["raw_actions"], mu, std)
    ratio = jnp.exp(log_probs - batch["old_log_probs"])
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages
    policy_loss = -jnp.minimum(surr1, surr2).mean()
    entropy = actor.entropy(mu, std, key).mean()
    return policy_loss - ENTROPY_COEF * entropy


def critic_loss_fn(critic, batch):
    return jnp.square(critic(batch["obs"]) - batch["target_values"]).mean()


@nnx.jit
def update_minibatch(actor, critic, actor_optimizer, critic_optimizer, batch, key):
    p_loss, p_grad = nnx.value_and_grad(actor_loss_fn)(actor, batch, batch["advantages"], key)
    v_loss, v_grad = nnx.value_and_grad(critic_loss_fn)(critic, batch)
    actor_optimizer.update(actor, p_grad)
    critic_optimizer.update(critic, v_grad)
    return p_loss, v_loss


def make_rollout(params, initial, num_envs, length):
    """Build a compiled rollout while keeping terminal observations for GAE."""
    reset_obs, _ = RoboTaxi.reset(jax.random.PRNGKey(0), params, initial)
    reset_obs = reset_obs.as_vector()
    batched_initial = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (num_envs,) + x.shape), initial
    )
    initial_obs = jnp.broadcast_to(reset_obs, (num_envs, reset_obs.size))

    @nnx.jit
    def collect(actor, critic, carry):
        def step(carry, _):
            states, obs, key, episode_returns, episode_lengths = carry
            key, action_key, step_key = jax.random.split(key, 3)
            mu, std = actor(obs)
            raw_actions = mu + std * jax.random.normal(action_key, mu.shape, dtype=mu.dtype)
            actions = scale_action(jnp.tanh(raw_actions), params)
            old_log_probs = actor.log_prob(raw_actions, mu, std)
            values = critic(obs)[..., 0]
            keys = jax.random.split(step_key, num_envs)
            # Preserve conditional DP replanning: vmap(cond) would execute both
            # branches, running the planner for every environment every frame.
            obs_obj, next_states, rewards, dones, info = jax.lax.map(
                lambda args: RoboTaxi.step(*args, params), (keys, states, actions)
            )
            terminal_obs = jax.vmap(lambda item: item.as_vector())(obs_obj)
            next_values = critic(terminal_obs)[..., 0]
            total_returns = episode_returns + rewards
            total_lengths = episode_lengths + 1

            def reset_leaf(new, init):
                mask = dones.reshape((num_envs,) + (1,) * (new.ndim - 1))
                return jnp.where(mask, init, new)

            next_states = jax.tree.map(reset_leaf, next_states, batched_initial)
            next_obs = jnp.where(dones[:, None], reset_obs, terminal_obs)
            transition = dict(
                obs=obs, raw_actions=raw_actions, old_log_probs=old_log_probs,
                values=values, next_values=next_values, rewards=rewards, dones=dones,
                terminated=info["goal_done"] | info["collision_done"],
                success=info["goal_done"] & ~info["collision_done"],
                is_idle=info["is_idle"],
                velocity=info["velocity"], target_speed=info["target_speed"],
                episode_returns=jnp.where(dones, total_returns, 0.0),
                episode_lengths=jnp.where(dones, total_lengths, 0),
            )
            carry = (
                next_states, next_obs, key,
                jnp.where(dones, 0.0, total_returns),
                jnp.where(dones, 0, total_lengths),
            )
            return carry, transition

        return jax.lax.scan(step, carry, None, length=length)

    return collect, batched_initial, initial_obs


def make_evaluator(params, initial, deterministic):
    @nnx.jit
    def run_episode(actor, key):
        obs, state = RoboTaxi.reset(key, params, initial)

        def condition(carry):
            return ~carry[5] & (carry[4] < params.max_steps_in_episode)

        def step(carry):
            state, obs, key, total_reward, length, _, _, _, _ = carry
            key, action_key, step_key = jax.random.split(key, 3)
            mu, std = actor(obs)
            raw = mu if deterministic else mu + std * jax.random.normal(
                action_key, mu.shape, dtype=mu.dtype
            )
            action = scale_action(jnp.tanh(raw), params)
            obs_obj, state, reward, done, info = RoboTaxi.step(step_key, state, action, params)
            return (
                state, obs_obj.as_vector(), key, total_reward + reward, length + 1, done,
                info["goal_done"] & ~info["collision_done"],
                info["collision_done"], info["time_done"],
            )

        carry = jax.lax.while_loop(condition, step, (
            state, obs.as_vector(), key, jnp.asarray(0.0), jnp.asarray(0),
            jnp.asarray(False), jnp.asarray(False), jnp.asarray(False), jnp.asarray(False),
        ))
        return carry[3], carry[4], carry[6], carry[7], carry[8]

    return run_episode


class PPO:
    def __init__(
        self, obs_dim=DEFAULT_OBS_DIM, critic_nn=None, actor_nn=None,
        actor_optimizer=None, critic_optimizer=None, key=None,
    ):
        self.obs_dim = obs_dim
        self.key = jax.random.PRNGKey(SEED) if key is None else key
        self.key, actor_key, critic_key = jax.random.split(self.key, 3)
        self.actor_nn = actor_nn if actor_nn is not None else Actor(obs_dim, nnx.Rngs(actor_key))
        self.critic_nn = critic_nn if critic_nn is not None else Critic(obs_dim, nnx.Rngs(critic_key))
        actor_tx = optax.adam(3e-4) if actor_optimizer is None else actor_optimizer
        critic_tx = optax.adam(1e-3) if critic_optimizer is None else critic_optimizer
        # Adam states must persist across minibatches.
        self.actor_optimizer = nnx.Optimizer(
            self.actor_nn, optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), actor_tx),
            wrt=nnx.Param,
        )
        self.critic_optimizer = nnx.Optimizer(
            self.critic_nn, optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), critic_tx),
            wrt=nnx.Param,
        )
        self.total_steps = 0
        self.env_kwargs = dict(DEFAULT_ENV_KWARGS)
        self._env_cache = {}
        self._eval_cache = {}

    def _environment(self, overrides=None):
        config = dict(self.env_kwargs)
        config.update(overrides or {})
        if config["discretization_scale"] != 1:
            raise ValueError("The base planner supports discretization_scale=1 only")
        if config["fps"] < 1 or config["max_steps"] < 1 or config["path_length"] < 2:
            raise ValueError("fps/max_steps must be positive; path_length must be >= 2")
        cache_key = tuple(sorted(config.items()))
        if cache_key not in self._env_cache:
            params, initial = RoboTaxi.init_params(jax.random.PRNGKey(SEED), **config)
            obs, _ = RoboTaxi.reset(jax.random.PRNGKey(SEED), params, initial)
            if obs.as_vector().size != self.obs_dim:
                raise ValueError(
                    f"Network expects {self.obs_dim} features; environment returns {obs.as_vector().size}"
                )
            self._env_cache[cache_key] = (params, initial)
        return config, cache_key, self._env_cache[cache_key]

    def compute_actor_loss(self, actor_nn, batch_data, normalized_advantages, key=None):
        if key is None:
            self.key, key = jax.random.split(self.key)
        return actor_loss_fn(actor_nn, batch_data, normalized_advantages, key)

    def compute_critic_loss(self, critic_nn, batch_data):
        return critic_loss_fn(critic_nn, batch_data)

    def train_step(self, batch_data, key=None):
        """Update actor/critic. Advantages must already be normalized."""
        if key is None:
            # Fresh entropy samples on every minibatch, separate from rollout noise.
            self.key, key = jax.random.split(self.key)
        return update_minibatch(
            self.actor_nn, self.critic_nn,
            self.actor_optimizer, self.critic_optimizer, batch_data, key,
        )

    def _calculate_advantage(self, rollout):
        return compute_advantage_and_target(
            rollout["values"], rollout["next_values"], rollout["rewards"],
            rollout["dones"], rollout["terminated"],
        )

    def train(
        self, total_steps=TIME_STEPS, num_envs=NUM_ENVS, unroll_length=UNROLL_LENGTH,
        batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, env_kwargs=None,
        log_dir="runs/ppo", checkpoint_path=f"{CKPT_DIR}/ppo.msgpack",
        eval_every=10, eval_episodes=3, progress=True,
    ):
        """Train for an additional step budget, rounded up to whole env batches."""
        if min(total_steps, num_envs, unroll_length, batch_size, num_epochs) < 1:
            raise ValueError("Training sizes and epoch count must be positive")
        if eval_every < 0 or eval_episodes < 1:
            raise ValueError("eval_every must be >= 0 and eval_episodes must be positive")
        config, _, (params, initial) = self._environment(env_kwargs)
        self.env_kwargs = config
        log_path = Path(log_dir) / "train.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _, states, obs = make_rollout(params, initial, num_envs, 1)
        carry = (states, obs, self.key, jnp.zeros(num_envs),
                 jnp.zeros(num_envs, dtype=jnp.int32))
        rollout_fns, history = {}, []
        collected, update = 0, 0
        best_score = (-float("inf"), -float("inf"))
        started = time.perf_counter()
        with log_path.open("a", encoding="utf-8") as log, tqdm(
            total=total_steps, disable=not progress, desc="PPO", unit="step"
        ) as bar:
            log.write(json.dumps(dict(event="config", env=config, num_envs=num_envs,
                                      unroll_length=unroll_length, num_epochs=num_epochs,
                                      reward_config=get_reward_config(),
                                      entropy_coef=ENTROPY_COEF, discount=DISCOUNT,
                                      gae_lambda=GAE_LAMBDA)) + "\n")
            while collected < total_steps:
                steps = min(unroll_length, (total_steps - collected + num_envs - 1) // num_envs)
                if steps not in rollout_fns:
                    rollout_fns[steps] = make_rollout(params, initial, num_envs, steps)[0]
                carry, rollout = rollout_fns[steps](self.actor_nn, self.critic_nn, carry)
                advantages, targets = self._calculate_advantage(rollout)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                size = steps * num_envs
                batch = dict(
                    obs=rollout["obs"].reshape(size, self.obs_dim),
                    raw_actions=rollout["raw_actions"].reshape(size, 2),
                    old_log_probs=rollout["old_log_probs"].reshape(size, 1),
                    advantages=advantages.reshape(size, 1),
                    target_values=targets.reshape(size, 1),
                )
                batch = jax.tree.map(jax.lax.stop_gradient, batch)
                self.key, losses = carry[2], []
                for _ in range(num_epochs):
                    self.key, shuffle_key = jax.random.split(self.key)
                    indices = jax.random.permutation(shuffle_key, size)
                    for start in range(0, size, batch_size):
                        selected = indices[start:start + batch_size]
                        losses.append(self.train_step(jax.tree.map(lambda x: x[selected], batch)))
                carry = (carry[0], carry[1], self.key, carry[3], carry[4])
                p_loss, v_loss = np.asarray(jax.device_get(jnp.mean(jnp.array(losses), axis=0)))
                if not np.isfinite([p_loss, v_loss]).all():
                    raise FloatingPointError("Non-finite PPO loss: inspect observations, rewards and std")
                collected += size
                self.total_steps += size
                update += 1
                completed = np.asarray(jax.device_get(rollout["dones"]))
                count = int(completed.sum())
                returns = np.asarray(jax.device_get(rollout["episode_returns"]))[completed]
                record = dict(
                    event="update", update=update, total_steps=self.total_steps,
                    actor_loss=float(p_loss), critic_loss=float(v_loss),
                    mean_step_reward=float(rollout["rewards"].mean()),
                    idle_fraction=float(rollout["is_idle"].mean()),
                    mean_velocity=float(rollout["velocity"].mean()),
                    mean_target_speed=float(rollout["target_speed"].mean()),
                    completed_episodes=count,
                    mean_episode_return=float(returns.mean()) if count else None,
                    steps_per_second=collected / (time.perf_counter() - started),
                )
                if eval_every and update % eval_every == 0:
                    result = self.evaluate(num_episodes=eval_episodes)
                    record["evaluation"] = result
                    score = (result["success_rate"], result["mean_return"])
                    if checkpoint_path and score > best_score:
                        path = Path(checkpoint_path)
                        self.save_checkpoint(path.with_name(path.stem + ".best" + path.suffix))
                        best_score = score
                history.append(record)
                log.write(json.dumps(record) + "\n")
                log.flush()
                bar.update(size)
                bar.set_postfix(actor=f"{p_loss:.3f}", critic=f"{v_loss:.3f}", episodes=count)
            if checkpoint_path:
                self.save_checkpoint(checkpoint_path)
        return history

    def evaluate(
        self, num_episodes=5, env_kwargs=None, deterministic=True,
        render=False, seed=SEED + 1,
    ):
        """Measure current policy; do not change weights or the training RNG."""
        if num_episodes < 1:
            raise ValueError("num_episodes must be positive")
        _, cache_key, (params, initial) = self._environment(env_kwargs)
        key, results = jax.random.PRNGKey(seed), []
        frontend, interrupted = None, False
        if render:
            import pygame
            from utils.autoreset import AutoResetWrapper
            from utils.renderer import PygameFrontend
            frontend = PygameFrontend(
                AutoResetWrapper(RoboTaxi, params, initial), params, initial,
                eval_mode=True, control_mode="vehicle",
            )
        try:
            if not render:
                eval_key = (cache_key, deterministic)
                if eval_key not in self._eval_cache:
                    self._eval_cache[eval_key] = make_evaluator(params, initial, deterministic)
                for _ in range(num_episodes):
                    key, episode_key = jax.random.split(key)
                    result = self._eval_cache[eval_key](self.actor_nn, episode_key)
                    results.append(tuple(np.asarray(x).item() for x in jax.device_get(result)))
            else:
                for _ in range(num_episodes):
                    key, reset_key = jax.random.split(key)
                    obs, state = RoboTaxi.reset(reset_key, params, initial)
                    total_reward, length, done = 0.0, 0, False
                    while not done:
                        if any(event.type == pygame.QUIT for event in pygame.event.get()):
                            interrupted = True
                            break
                        key, action_key, step_key = jax.random.split(key, 3)
                        mu, std = self.actor_nn(obs.as_vector())
                        raw = mu if deterministic else mu + std * jax.random.normal(action_key, mu.shape)
                        obs, state, reward, done, info = RoboTaxi.step(
                            step_key, state, scale_action(jnp.tanh(raw), params), params
                        )
                        total_reward += float(reward)
                        length += 1
                        done = bool(done)
                        frontend.obs, frontend.state, frontend.info = obs, state, info
                        frontend.draw()
                        frontend.clock.tick(params.fps)
                    if interrupted:
                        break
                    results.append((total_reward, length,
                                    bool(info["goal_done"] & ~info["collision_done"]),
                                    bool(info["collision_done"]), bool(info["time_done"])))
        finally:
            if frontend is not None:
                pygame.quit()
        mean = lambda index: float(np.mean([x[index] for x in results])) if results else 0.0
        return dict(
            episodes=len(results), mean_return=mean(0), mean_length=mean(1),
            success_rate=mean(2), collision_rate=mean(3), timeout_rate=mean(4),
            interrupted=interrupted,
        )

    def save_checkpoint(self, path):
        """Save weights/settings for evaluation, not an exact optimizer resume."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            version=1, obs_dim=self.obs_dim, total_steps=self.total_steps,
            env_kwargs=self.env_kwargs,
            reward_config=get_reward_config(),
            entropy_coef=ENTROPY_COEF, discount=DISCOUNT, gae_lambda=GAE_LAMBDA,
            actor=nnx.state(self.actor_nn, nnx.Param).to_pure_dict(),
            critic=nnx.state(self.critic_nn, nnx.Param).to_pure_dict(),
        )
        path.write_bytes(serialization.msgpack_serialize(jax.device_get(payload)))
        return path

    @classmethod
    def load_checkpoint(cls, path):
        """Restore weights; a new training run starts with fresh Adam state."""
        payload = serialization.msgpack_restore(Path(path).read_bytes())
        if payload["version"] != 1:
            raise ValueError("Unsupported checkpoint version")
        agent = cls(obs_dim=payload["obs_dim"])
        for model, name in ((agent.actor_nn, "actor"), (agent.critic_nn, "critic")):
            state = nnx.state(model, nnx.Param)
            state.replace_by_pure_dict(jax.tree.map(jnp.asarray, payload[name]))
            nnx.update(model, state)
        agent.env_kwargs = dict(payload["env_kwargs"])
        agent.total_steps = int(payload["total_steps"])
        return agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--total-steps", type=int, default=TIME_STEPS)
    train.add_argument("--num-envs", type=int, default=NUM_ENVS)
    train.add_argument("--unroll-length", type=int, default=UNROLL_LENGTH)
    train.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    train.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    train.add_argument("--log-dir", default="runs/ppo")
    train.add_argument("--eval-every", type=int, default=10)
    train.add_argument("--no-progress", action="store_true")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--episodes", type=int, default=5)
    evaluate.add_argument("--render", action="store_true")
    evaluate.add_argument("--stochastic", action="store_true")
    for command in (train, evaluate):
        command.add_argument("--checkpoint", default=f"{CKPT_DIR}/ppo.msgpack")
        command.add_argument("--map-id", type=int, default=None)
        command.add_argument("--max-steps", type=int, default=None)
        command.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    env_kwargs = {}
    if args.map_id is not None:
        env_kwargs["map_id"] = args.map_id
    if args.max_steps is not None:
        env_kwargs["max_steps"] = args.max_steps
    if args.command == "train":
        agent = PPO(key=jax.random.PRNGKey(args.seed))
        history = agent.train(
            total_steps=args.total_steps, num_envs=args.num_envs,
            unroll_length=args.unroll_length, batch_size=args.batch_size,
            num_epochs=args.epochs, env_kwargs=env_kwargs, log_dir=args.log_dir,
            checkpoint_path=args.checkpoint, eval_every=args.eval_every,
            progress=not args.no_progress,
        )
        print(json.dumps(history[-1], indent=2))
    else:
        agent = PPO.load_checkpoint(args.checkpoint)
        print(json.dumps(agent.evaluate(
            num_episodes=args.episodes, env_kwargs=env_kwargs,
            deterministic=not args.stochastic, render=args.render, seed=args.seed,
        ), indent=2))


if __name__ == "__main__":
    main()
