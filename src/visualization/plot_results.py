import matplotlib.pyplot as plt
import os

def plot_rewards(rewards, gamma, save_path=None):
    """
    Plot episode rewards over time.

    Parameters
    ----------
    rewards : list of float
        Rewards per episode.
    gamma : float
        Discount factor used during training.
    save_path : str, optional
        Path to save the plot, by default None
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label=f'Gamma = {gamma}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Reward Progression for Gamma={gamma}')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_gamma_comparison(all_rewards, save_path=None):
    """
    Plot average reward per gamma for comparison.

    Parameters
    ----------
    all_rewards : dict
        Keys are gamma values, values are lists of rewards.
    save_path : str, optional
        Path to save the plot, by default None
    """
    plt.figure(figsize=(10, 5))
    for gamma, rewards in all_rewards.items():
        smoothed = moving_average(rewards, window=10)
        plt.plot(smoothed, label=f'Gamma = {gamma}')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward (window=10)')
    plt.title('Reward Comparison for Different Discount Factors')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()


def moving_average(data, window=10):
    """
    Compute the moving average of a 1D list or array.

    Parameters
    ----------
    data : list or np.ndarray
        Input data.
    window : int, optional
        Window size, by default 10

    Returns
    -------
    np.ndarray
        Smoothed data.
    """
    import numpy as np
    return np.convolve(data, np.ones(window) / window, mode='valid')
