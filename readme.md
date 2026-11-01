# IAEML - Assignment

This repository was specifically designed to practice and implement the concepts covered in IAEML course at Innopolis University. [Vehicular automation](https://en.wikipedia.org/wiki/Vehicular_automation) is a notable challenging domain where these concepts can be applied.

Specifically, in this repository we have a simple 2D environment with an agent, destination and obstacles to avoid. The obstacles can be either static or moving. For simplicity, all objects are modeled as circles with radius $r=1$.

`tasks.py` comes in handy when we want to train something like meta-learning.

`base.py` defines an environment which models core elements required in self-driving cars:

- Perception is modeled as $M$ evenly fanned rays casted from the center of the agent.
- Navigation is solved using a dynamic programming algorithm fully executed on a device.

The environment has the following interface

- `init_params()` prepares `env_params` and `init_state`.
- `reset()` can be used to set `env_state` back to `env_state`.
- `step()` executes one step of the simulation.
- `get_observation()` returns the "point-of-view" of the agent.

Notably, `BaseEnv` is just a struct with functions that operates on `BaseEnvState` and takes `BaseEnvParams` as static arguments. Static arguments are "baked" into a function using `jax.jit()` decorator.

One limitation at the moment is that sampling a new task will always result in recompiling the environment functions since it changes the content of `env_params` and shapes in `env_state`. The latter can be solved by forcing all maps to be of the same shape.

`base.py` does not implement a model of vehicle dynamics.

You can play it

```bash
python src/main.py
```

There are some issues with JAX on OSX. This solves it

```bash
JAX_PLATFORM_NAME=cpu python src/main.py
```

...

---

Tested with python 3.11, Ubuntu 24.04 adm64 on Oct 20, 2025.
