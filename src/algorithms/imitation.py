"""Pure Pursuit demonstrations, behavioral cloning, DAgger and PPO fine-tuning."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import equinox
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx, serialization

from algorithms.ppo import DEFAULT_ENV_KWARGS, DISCOUNT, PPO, scale_action
from env.route_rewards import RouteRewardObservation, RouteRewardState, route_distance
from env.tracking import TRACKING_OBS_DIM, TrackingEnv, normalized_teacher


class CollectionState(NamedTuple):
    """Named JAX loop carry for one teacher/student episode."""

    state: RouteRewardState
    observation: RouteRewardObservation
    key: jax.Array
    done: jax.Array
    length: jax.Array
    total_reward: jax.Array
    observations: jax.Array
    teacher_actions: jax.Array
    rewards: jax.Array
    success: jax.Array
    collision: jax.Array


def varied_initial(initial, params, key):
    """Perturb pose, speed and obstacle phase while retaining the same destination."""
    position_key, heading_key, phase_key, velocity_key = jax.random.split(key, 4)
    position = initial.agent_pos + jax.random.uniform(position_key, (2,), minval=-0.12, maxval=0.12)
    position = position.at[1].set(jnp.maximum(position[1], 1.0))
    angle = jax.random.uniform(heading_key, (), minval=-0.12, maxval=0.12) + jnp.pi / 2
    phase = jax.random.uniform(phase_key, (), minval=0.0, maxval=10.0)
    obstacles = jnp.mod(
        initial.kinematic_obstacles + phase * initial.kinematic_obst_velocities,
        jnp.asarray(params.map_height_width[::-1]),
    )
    return equinox.tree_at(
        lambda state: (
            state.agent_pos,
            state.agent_forward_dir,
            state.velocity,
            state.kinematic_obstacles,
            state.best_route_distance,
        ),
        initial,
        (
            position,
            jnp.array([jnp.cos(angle), jnp.sin(angle)]),
            jax.random.uniform(velocity_key, (), maxval=0.15),
            obstacles,
            route_distance(position, params),
        ),
    )


def make_collector(params, initial):
    """Collect state-action labels under a teacher/student action mixture."""

    @nnx.jit
    def collect(actor, key, beta, randomize):
        key, start_key = jax.random.split(key)
        state = jax.lax.cond(
            randomize, lambda _: varied_initial(initial, params, start_key), lambda _: initial, None
        )
        obs, state = TrackingEnv.reset(key, params, state)
        observations = jnp.zeros((params.max_steps_in_episode, TRACKING_OBS_DIM))
        targets = jnp.zeros((params.max_steps_in_episode, 2))
        rewards = jnp.zeros(params.max_steps_in_episode)

        def condition(carry):
            return ~carry.done & (carry.length < params.max_steps_in_episode)

        def step(carry):
            vector = carry.observation.as_vector()
            target = normalized_teacher(vector, params)
            mean, _ = actor(vector)
            action = beta * target + (1 - beta) * jnp.tanh(mean)
            key, step_key = jax.random.split(carry.key)
            next_obs, next_state, reward, done, info = TrackingEnv.step(
                step_key, carry.state, scale_action(action, params), params
            )
            return CollectionState(
                state=next_state,
                observation=next_obs,
                key=key,
                done=done,
                length=carry.length + 1,
                total_reward=carry.total_reward + reward,
                observations=carry.observations.at[carry.length].set(vector),
                teacher_actions=carry.teacher_actions.at[carry.length].set(target),
                rewards=carry.rewards.at[carry.length].set(reward),
                success=info["goal_done"] & ~info["collision_done"],
                collision=info["collision_done"],
            )

        result = jax.lax.while_loop(
            condition,
            step,
            CollectionState(
                state=state,
                observation=obs,
                key=key,
                done=jnp.asarray(False),
                length=jnp.asarray(0),
                total_reward=jnp.asarray(0.0),
                observations=observations,
                teacher_actions=targets,
                rewards=rewards,
                success=jnp.asarray(False),
                collision=jnp.asarray(False),
            ),
        )
        return (
            result.observations,
            result.teacher_actions,
            result.rewards,
            result.length,
            result.total_reward,
            result.success,
            result.collision,
        )

    return collect


@nnx.jit
def supervised_update(actor, optimizer, observations, teacher_actions):
    """Fit normalized actions, emphasizing turns and braking examples."""

    def loss(model):
        mean, _ = model(observations)
        weights = 1 + 4 * (jnp.abs(teacher_actions[:, 1:2]) > 0.15)
        weights = weights + 2 * (teacher_actions[:, :1] < 0)
        return jnp.mean(weights * jnp.square(jnp.tanh(mean) - teacher_actions))

    value, gradients = nnx.value_and_grad(loss)(actor)
    optimizer.update(actor, gradients)
    return value


@nnx.jit
def value_update(critic, optimizer, observations, returns):
    """Fit the critic to discounted demonstration/DAgger returns."""

    def loss(model):
        return jnp.mean(jnp.square(model(observations)[:, 0] - returns))

    value, gradients = nnx.value_and_grad(loss)(critic)
    optimizer.update(critic, gradients)
    return value


def fit_actor(model, observations, teacher_actions, steps, key, learning_rate=0.0003):
    """Perform one BC/DAgger stage with a fresh supervised optimizer."""
    optimizer = nnx.Optimizer(model.actor_nn, optax.adam(learning_rate), wrt=nnx.Param)
    observations, teacher_actions = jnp.asarray(observations), jnp.asarray(teacher_actions)
    for _ in range(steps):
        key, draw = jax.random.split(key)
        indices = jax.random.randint(draw, (512,), 0, len(observations))
        loss = supervised_update(
            model.actor_nn, optimizer, observations[indices], teacher_actions[indices]
        )
    return (key, float(loss))


def fit_critic(model, observations, returns, key, steps=2000):
    """Initialize the value estimate before on-policy PPO fine-tuning."""
    optimizer = nnx.Optimizer(model.critic_nn, optax.adam(0.001), wrt=nnx.Param)
    observations, returns = jnp.asarray(observations), jnp.asarray(returns)
    for _ in range(steps):
        key, draw = jax.random.split(key)
        indices = jax.random.randint(draw, (512,), 0, len(observations))
        value_update(model.critic_nn, optimizer, observations[indices], returns[indices])
    return key


def set_exploration(actor, std=0.03):
    """Initialize constant Gaussian noise without changing the mean-action head."""
    actor.dense3.kernel[...] = actor.dense3.kernel[...].at[:, 2:].set(0)
    actor.dense3.bias[...] = actor.dense3.bias[...].at[2:].set(jnp.log(std))


def gather(model, collector, key, episodes, beta, randomize, only_success=False):
    """Aggregate variable-length episodes; record failures even when filtering data."""
    observation_batches, action_batches, returns, metrics = ([], [], [], [])
    for _ in range(episodes):
        key, episode_key = jax.random.split(key)
        observations, teacher_actions, rewards, length, total, success, collision = collector(
            model.actor_nn, episode_key, jnp.asarray(beta), jnp.asarray(randomize)
        )
        length = int(length)
        metrics.append(
            dict(
                success=bool(success),
                collision=bool(collision),
                length=length,
                return_value=float(total),
            )
        )
        if only_success and (not bool(success)):
            continue
        rewards = np.asarray(rewards)[:length]
        episode_returns = np.zeros(length, dtype=np.float32)
        discounted = 0.0
        for i in range(length - 1, -1, -1):
            discounted = rewards[i] + DISCOUNT * discounted
            episode_returns[i] = discounted
        observation_batches.append(np.asarray(observations)[:length])
        action_batches.append(np.asarray(teacher_actions)[:length])
        returns.append(episode_returns)
    data = (
        (
            np.concatenate(observation_batches),
            np.concatenate(action_batches),
            np.concatenate(returns),
        )
        if observation_batches
        else None
    )
    return (key, data, metrics)


def save_training_state(model, weights_path, actor_lr=3e-5, critic_lr=1e-3):
    weights_path = Path(weights_path)
    payload = dict(
        actor_optimizer=serialization.to_state_dict(
            nnx.state(model.actor_optimizer).to_pure_dict()
        ),
        critic_optimizer=serialization.to_state_dict(
            nnx.state(model.critic_optimizer).to_pure_dict()
        ),
        rng=np.asarray(model.key),
        total_steps=model.total_steps,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        imitation_weight=model.imitation_weight,
        weights_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        active_rollouts_saved=False,
    )
    weights_path.with_suffix(".training.msgpack").write_bytes(
        serialization.msgpack_serialize(jax.device_get(payload))
    )


def restore_training_state(model, weights_path):
    """Restore Adam and RNG alongside matching weights; new rollouts start fresh."""
    weights_path = Path(weights_path)
    payload = serialization.msgpack_restore(
        weights_path.with_suffix(".training.msgpack").read_bytes()
    )
    if payload["weights_sha256"] != hashlib.sha256(weights_path.read_bytes()).hexdigest():
        raise ValueError("Weights and optimizer checkpoint do not match")
    if payload["total_steps"] != model.total_steps:
        raise ValueError("Checkpoint step counters do not match")
    for attribute, rate in (
        ("actor_optimizer", payload["actor_lr"]),
        ("critic_optimizer", payload["critic_lr"]),
    ):
        network = model.actor_nn if attribute == "actor_optimizer" else model.critic_nn
        optimizer = nnx.Optimizer(
            network, optax.chain(optax.clip_by_global_norm(1.0), optax.adam(rate)), wrt=nnx.Param
        )
        state = nnx.state(optimizer)
        values = serialization.from_state_dict(state.to_pure_dict(), payload[attribute])
        state.replace_by_pure_dict(values)
        nnx.update(optimizer, state)
        setattr(model, attribute, optimizer)
    model.key = jnp.asarray(payload["rng"])
    model.imitation_weight = payload["imitation_weight"]
    return model


def save_metrics(path, records):
    path.write_text(json.dumps(records, indent=2) + "\n")


def save_best_policy(evaluations, output):
    """Select by success, collisions, duration and return; preserve a matching sidecar."""
    winner = max(
        evaluations,
        key=lambda item: (
            item["metrics"]["success_rate"],
            -item["metrics"]["collision_rate"],
            -item["metrics"]["mean_length"],
            item["metrics"]["mean_return"],
        ),
    )
    best = PPO.load_checkpoint(winner["path"])
    best.imitation_weight = 0.5
    best.save_checkpoint(output / "best_policy.msgpack")
    winner_state = Path(winner["path"]).with_suffix(".training.msgpack")
    if winner_state.exists():
        payload = serialization.msgpack_restore(winner_state.read_bytes())
        payload["weights_sha256"] = hashlib.sha256(
            (output / "best_policy.msgpack").read_bytes()
        ).hexdigest()
        (output / "best_policy.training.msgpack").write_bytes(
            serialization.msgpack_serialize(payload)
        )
    return winner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/imitation_20261118")
    parser.add_argument("--ppo-steps", type=int, default=200000)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env_kwargs = dict(
        DEFAULT_ENV_KWARGS,
        max_steps=3000,
        subgoal_fractions=(),
        observe_navigation_target=True,
        tracking_margin=1.25,
        lookahead_distance=1.3,
        cruise_speed=0.8,
        safety_horizon=2.0,
        prediction_clearance=1.3,
    )
    model = PPO(
        obs_dim=TRACKING_OBS_DIM,
        env_class=TrackingEnv,
        teacher_fn=normalized_teacher,
        imitation_weight=0.5,
        key=jax.random.PRNGKey(67),
    )
    model.env_kwargs = env_kwargs
    _, _, (params, initial) = model._environment()
    collector = make_collector(params, initial)
    key = jax.random.PRNGKey(1234)
    # Collect successful teacher demonstrations, then clone their actions.
    key, data, teacher_metrics = gather(model, collector, key, 32, 1.0, True, only_success=True)
    save_metrics(output / "teacher.json", teacher_metrics)
    if data is None:
        raise RuntimeError("No successful teacher trajectories")
    observations, teacher_actions, returns = data
    np.savez_compressed(
        output / "demonstrations.npz",
        observations=observations,
        actions=teacher_actions,
        returns=returns,
    )
    print(
        json.dumps(
            dict(
                stage="teacher",
                episodes=len(teacher_metrics),
                success_rate=np.mean([r["success"] for r in teacher_metrics]),
                samples=len(observations),
            )
        ),
        flush=True,
    )
    key, loss = fit_actor(model, observations, teacher_actions, 5000, key)
    set_exploration(model.actor_nn)
    model.save_checkpoint(output / "bc.msgpack")
    bc = model.evaluate(num_episodes=5, seed=123)
    save_metrics(output / "bc_evaluation.json", bc)
    print(json.dumps(dict(stage="bc", loss=loss, evaluation=bc)), flush=True)
    evaluations = [dict(stage="bc", metrics=bc, path=str(output / "bc.msgpack"))]
    # Visit student-induced states and aggregate corrective teacher labels.
    for iteration, beta in enumerate((0.5, 0.2, 0.0)):
        key, new_data, metrics = gather(model, collector, key, 12, beta, True)
        save_metrics(output / f"dagger_{iteration + 1}_rollouts.json", metrics)
        new_observations, new_actions, new_returns = new_data
        observations, teacher_actions, returns = (
            np.concatenate((observations, new_observations)),
            np.concatenate((teacher_actions, new_actions)),
            np.concatenate((returns, new_returns)),
        )
        key, loss = fit_actor(model, observations, teacher_actions, 2500, key, learning_rate=0.0001)
        set_exploration(model.actor_nn)
        path = output / f"dagger_{iteration + 1}.msgpack"
        model.save_checkpoint(path)
        result = model.evaluate(num_episodes=5, seed=123)
        save_metrics(output / f"dagger_{iteration + 1}_evaluation.json", result)
        evaluations.append(dict(stage=f"dagger_{iteration + 1}", metrics=result, path=str(path)))
        print(
            json.dumps(dict(stage=f"dagger_{iteration + 1}", loss=loss, evaluation=result)),
            flush=True,
        )
    # Warm up the value network, then fine-tune with PPO and teacher regularization.
    key = fit_critic(model, observations, returns, key)
    model.actor_optimizer = nnx.Optimizer(
        model.actor_nn,
        optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-05)),
        wrt=nnx.Param,
    )
    model.critic_optimizer = nnx.Optimizer(
        model.critic_nn,
        optax.chain(optax.clip_by_global_norm(1.0), optax.adam(0.001)),
        wrt=nnx.Param,
    )
    model.imitation_weight = 0.5
    original_save = model.save_checkpoint

    def save_with_optimizer(path):
        result = original_save(path)
        save_training_state(model, path)
        return result

    model.save_checkpoint = save_with_optimizer
    history = model.train(
        total_steps=args.ppo_steps,
        num_envs=16,
        num_epochs=2,
        env_kwargs=env_kwargs,
        log_dir=output / "ppo",
        checkpoint_path=output / "ppo.msgpack",
        eval_every=5,
        progress=False,
    )
    save_training_state(model, output / "ppo.msgpack")
    print(json.dumps(dict(stage="ppo", last_update=history[-1])), flush=True)
    for path in (output / "ppo.msgpack", output / "ppo.best.msgpack"):
        restored = PPO.load_checkpoint(path)
        result = restored.evaluate(num_episodes=20, seed=123)
        stochastic = restored.evaluate(num_episodes=20, deterministic=False, seed=123)
        evaluations.append(
            dict(stage=path.stem, metrics=result, stochastic=stochastic, path=str(path))
        )
    winner = save_best_policy(evaluations, output)
    save_metrics(output / "evaluations.json", evaluations)
    print(json.dumps(dict(stage="best", result=winner)), flush=True)


if __name__ == "__main__":
    main()
