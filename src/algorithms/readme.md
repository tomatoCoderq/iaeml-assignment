# PPO for RoboTaxi

Run commands from the repository root with the Python environment containing
the dependencies in `requirements.txt` (the local environment is `jaxing`).

```bash
conda activate jaxing
PYTHONPATH=src python -m algorithms.ppo train --total-steps 100000 --num-envs 16
PYTHONPATH=src python -m algorithms.ppo evaluate --checkpoint checkpoints/ppo.msgpack --episodes 5
PYTHONPATH=src python -m algorithms.ppo evaluate --checkpoint checkpoints/ppo.msgpack --render
```

For a small smoke run:

```bash
PYTHONPATH=src python -m algorithms.ppo train --total-steps 256 --num-envs 2 --unroll-length 32 --batch-size 64 --epochs 1 --max-steps 64
```

`train()` collects observations from `RoboTaxi.reset/step`, converts them with
`as_vector()`, samples normalized acceleration/steering, scales them to the
vehicle's physical limits, and collects rollouts. GAE bootstraps from the final
observation on timeouts, while stopping its recursion at every episode boundary.
The actor and critic use the same observation; no privileged state is supplied.

Rollouts use a compiled `lax.scan` over time and `lax.map` over environments.
`lax.map` preserves the conditional path-planner calls; batching those conditions
with `vmap` would evaluate both branches. Network inference is batched and PPO
minibatch updates are compiled. Adam moments persist across updates, and gradients
are clipped. Actor loss includes `-0.01 * H(tanh(Normal))`, with fresh
Monte Carlo entropy samples per minibatch. Running observation statistics are omitted.

Reward v3 uses `0.2 * (old_distance - new_distance) - 0.01 * dt`, minus
`0.2 * (speed - target_speed)**2 * dt`. Target speed is up to `1.0` on a clear
straight section, reduced toward `0.3` for turns, and reduced further near forward
obstacles and the goal. A stop is requested when a forward ray's clearance is
at most `0.25`; side/rear rays do not force stops. An extra `-3.0 * dt` applies
when speed is below `0.05` and target speed is above `0.05`. This idle penalty is
waived for requested stops. Goal/collision rewards remain `+100`/`-100` and override
all dense terms. Progress shaping is symmetric in both directions.

Target speed is appended to the observation, so 32 rays now produce **69 features**.
The default actor/critic input dimensions have been updated. Old 68-feature
checkpoints cannot be evaluated with this observation contract; start a new run:

```bash
PYTHONPATH=src python -m algorithms.ppo train --total-steps 5000000 --num-envs 16 --log-dir runs/ppo_v3 --checkpoint checkpoints/ppo_v3.msgpack
PYTHONPATH=src python -m algorithms.ppo evaluate --checkpoint checkpoints/ppo_v3.msgpack --render
```

Training logs include `mean_velocity` and `mean_target_speed`; renderer info
includes `target_speed`, `speed_error`, `speed_penalty` and `idle_penalty_applied`.
Exact coefficients, the baseline experiment, and its observed failure are recorded
in [the reward history](../../docs/reward_history.md). New config log records and
checkpoints also store the reward version and entropy coefficient.

Metrics are appended to `runs/ppo/train.jsonl`. The final weights and environment
settings are saved to `checkpoints/ppo.msgpack`; periodic evaluation also saves
`ppo.best.msgpack`. Checkpoints reproduce evaluation; they do not contain Adam
moments or the active rollouts, so loading starts any new training with fresh
optimizer state. Evaluation reports return, episode length, success, collision
and timeout rates. It uses `tanh(mean)` by default; use `--stochastic` to sample.

All resets currently use the map's fixed initial state. Repeating deterministic
evaluation on one map does not test generalization. Use `--map-id 1` or `2` to
evaluate other provided maps. A working training loop is not evidence that the
policy has learned to reach the goal; assess that through evaluation/video.

Tests (including rendering into SDL's in-memory display):

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
