import gymnasium as gym

def make_frozenlake_env(is_slippery: bool = True):
    """
    Create the FrozenLake environment with specified parameters.

    Parameters
    ----------
    is_slippery : bool, optional
        Whether the frozen lake is slippery, by default True

    Returns
    -------
    gymnasium.Env
        An instance of the FrozenLake-v1 environment
    """
    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=is_slippery,
        reward_schedule=(1, 0, 0)
    )
    return env
