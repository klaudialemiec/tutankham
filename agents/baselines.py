from agents.test import print_metrics
from processors.baselines_atari import make_tutankham_env, make_tutankham_env_test
from stable_baselines import A2C, PPO2
from stable_baselines.common.policies import CnnPolicy

AGENTS = {"a2c": A2C, "ppo": PPO2}


def create_agent(agent_name, num_env=16, logdir="board/"):
    env = make_tutankham_env(num_env=num_env)
    agent = AGENTS[agent_name]
    return agent(CnnPolicy, env, verbose=1, tensorboard_log=logdir + agent_name)


def train_test_baseline(
    agent_name,
    checkpoint,
    epochs_train,
    epochs_test,
    test_visualize=False,
    num_env=16,
    logdir="board/",
    episode_len_test=1000,
):
    agent = create_agent(agent_name, num_env, logdir)
    train_baseline(agent, epochs_train, checkpoint=checkpoint)
    test_baseline(agent, epochs_test, episode_len_test, visualize=test_visualize)


def train_baseline(agent, total_timesteps, checkpoint):
    agent.learn(total_timesteps=total_timesteps)
    agent.save(checkpoint)


def test_baseline(agent, num_episodes, episode_len, visualize):
    env = make_tutankham_env_test()
    state = env.reset()
    rewards_list = []
    keys_list = []
    keys_reward_list = []
    timesteps_list = []

    for epiosode in range(num_episodes):
        episode_reward = 0
        is_done = False
        timestep = 0
        keys = 0
        keys_reward = 0

        while not is_done:
            if visualize:
                env.render()

            action, _ = agent.predict(state)
            state, rewards, dones, info = env.step(action)
            episode_reward += rewards[0]

            if rewards[0] >= 10:
                keys += 1
                keys_reward += rewards[0]

            if dones[0] == True or timestep == episode_len:
                print(
                    "Episode {epiosode} finished after {timestep} timesteps with reward {episode_reward}.\nReward from keys: {keys_reward}; keys: {keys}"
                )
                is_done = True
                rewards_list.append(episode_reward)
                keys_list.append(keys)
                timesteps_list.append(timestep)
                keys_reward_list.append(keys_reward)
            timestep += 1

    rewards_list = np.array(rewards_list)
    keys_list = np.array(keys_list)
    timesteps_list = np.array(timesteps_list)

    print_metrics(rewards_list, keys_list, keys_reward_list, timesteps_list)