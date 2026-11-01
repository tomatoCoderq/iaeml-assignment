import chex
from jax import numpy as jnp
from dataclasses import dataclass, field


# Define any 10x20 map
MAPS = [
    [
        "####################",
        "####################",
        "#.........v........#",
        "#.........v........#",
        "#...##hhh.v........#",
        "#...##.............#",
        "#...##.............#",
        "#...##.............#",
        "#.S.##..........G..#",
        "####################",
    ],
    [
        "####################",
        "####################",
        "#.........#........#",
        "#.........#........#",
        "#...##hhh.#........#",
        "#...##.............#",
        "#...##......v......#",
        "#...##......v......#",
        "#.S.##......v...G..#",
        "####################",
    ],
    [
        "####################",
        "...................#",
        "#.........#........#",
        "#.........#........#",
        "#hhhh#....#.....S..#",
        "#....#.........#####",
        "#....#......v..#####",
        "#....#......v..#####",
        "#.G..#......v..#####",
        "####################",
    ],
]


@dataclass
class BaseTaskSampler:
    map_id: int
    speed_h: int = 1  # horizontal speed for 'h'
    speed_v: int = 1  # vertical speed for 'v'

    # Derived fields initialized later
    map_str: list[str] = field(init=False)
    height: int = field(init=False)
    width: int = field(init=False)

    agent_pos: chex.Array = field(init=False)
    goal_pos: chex.Array = field(init=False)
    agent_forward_dir: chex.Array = field(init=False, default_factory=lambda: jnp.array([0.0, 1.0]))
    agent_goal_dir: chex.Array = field(init=False)

    static_obstacles: chex.Array = field(init=False)
    kinematic_obstacles: chex.Array = field(init=False)
    kinematic_obst_velocities: chex.Array = field(init=False)

    def __post_init__(self):
        self.map_str = MAPS[self.map_id]
        self.height = len(self.map_str)
        self.width = len(self.map_str[0])
        self._parse_map()

    def _parse_map(self):
        static_obs = []
        kinematic_obs = []
        kinematic_vel = []

        agent_pos = None
        goal_pos = None

        for y, row in enumerate(self.map_str):
            for x, cell in enumerate(row):
                if cell == "#":
                    static_obs.append((y, x))
                elif cell == "h":
                    kinematic_obs.append((y, x))
                    kinematic_vel.append([self.speed_h, 0])  # horizontal
                elif cell == "v":
                    kinematic_obs.append((y, x))
                    kinematic_vel.append([0, self.speed_v])  # vertical
                elif cell == "S":
                    agent_pos = jnp.array([y, x])
                elif cell == "G":
                    goal_pos = jnp.array([y, x])

        self.static_obstacles = jnp.array(static_obs)
        self.kinematic_obstacles = jnp.array(kinematic_obs)
        self.kinematic_obst_velocities = jnp.array(kinematic_vel, dtype=jnp.int32)
        self.agent_pos = agent_pos
        self.goal_pos = goal_pos
        # normalized direction to goal
        self.agent_goal_dir = (goal_pos - agent_pos) / jnp.linalg.norm(
            goal_pos - agent_pos
        )
