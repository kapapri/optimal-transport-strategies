from gymnasium.envs.registration import register

register(
    id="gym_environments/Lattice-v0",
    entry_point="gym_environments.envs:LatticeEnv",
)

register(
    id="gym_environments/GridWorld-v3",
    entry_point="gym_environments.envs:GridWorldEnv",
)

register(
    id="gym_environments/FrozenLake-v8",
    entry_point="gym_environments.envs:FrozenLakeEnv",
)