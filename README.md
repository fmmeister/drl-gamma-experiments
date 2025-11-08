# Influence of Discount Factor on Deep Q-Learning in FrozenLake

## Overview

This project investigates how the discount factor (γ) influences agent behavior in the FrozenLake-v1 environment using Deep Q-Learning.  
Lower γ values encourage "short-sighted" behavior focusing on immediate rewards, whereas higher γ promotes "far-sighted" planning.

## Structure

- `src/environments/`: FrozenLake environment wrapper.
- `src/agents/`: Deep Q-Learning agent implementation.
- `src/visualization/`: Plotting utilities.
- `experiments/`: Scripts to run experiments for different γ.
- `results/`: Logs and plots saved here.

## Usage

1. Install dependencies:

```bash
pip install gymnasium torch matplotlib numpy
