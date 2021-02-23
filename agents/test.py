import numpy as np


def print_metrics(rewards_list, keys_list, keys_reward_list, timesteps_list):
    print(f"Reward\n\t- mean: {np.mean(rewards_list)}"
        + f"\t- min: {np.min(rewards_list)}"
        + f"\t- max: {np.max(rewards_list)}"
    )
    print(f"Mean keys: {np.mean(keys_list)}")
    print(f"Max keys: {np.max(keys_list)}"))
    print(f"Mean keys reward: {np.mean(keys_reward_list)}")
    print(f"Mean reward for creatures: {np.mean(rewards_list) - np.mean(keys_reward_list)}")
    print(f"Mean timesteps: {np.mean(timesteps_list)}")
