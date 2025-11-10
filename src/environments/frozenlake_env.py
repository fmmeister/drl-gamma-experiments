import gymnasium as gym

def make_frozenlake_env(
    is_slippery: bool = True,
    map_name: str = "4x4",
    reward_schedule: tuple = (1, 0, 0)
):
    """
    Create the FrozenLake environment with configurable parameters.

    Parameters
    ----------
    is_slippery : bool, optional
        Whether the lake is slippery.
    map_name : str, optional
        Map size name, default is "4x4".
    reward_schedule : tuple, optional
        Custom reward configuration as (goal, hole, step).

    Returns
    -------
    gymnasium.Env
        Configured FrozenLake-v1 environment.
    """
    env = gym.make(
        "FrozenLake-v1",
        map_name=map_name,
        is_slippery=is_slippery,
        reward_schedule=reward_schedule,
    )
    return env

