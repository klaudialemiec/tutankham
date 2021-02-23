import gym
from processors.baselines_atari import make_tutankham_env, make_tutankham_env_test
from stable_baselines import A2C, PPO2
from agents.baselines import train_test_baseline, test_baseline
from agents.dqn import train_test_dqn

DQN_AGENT = "dqn"
A2C_AGENT = "a2c"
PPO_AGENT = "ppo"


def train_agent(agent_name, checkpoint, epochs_train, epochs_test, **kwargs):
    if agent_name == DQN_AGENT:
        train_test_dqn(checkpoint, epochs_train, epochs_test, **kwargs)
    else:
        train_and_test(agent_name, checkpoint, epochs_train, epochs_test)


def load_agent(agent_name, checkpoint, epochs_test, episode_len, visualize):
    if agent_name == DQN_AGENT:
        agent = create_dqn_agent()
        optimizer = Adam(lr=0.00025)
        agent.compile(optimizer)
        agent.load_weights(checkpoint)
        test_dqn(agent, epochs_test, episode_len, visualize)

    elif agent_name == A2C_AGENT:
        agent = A2C.load(checkpoint)
        baselines_test(agent, epochs_test, episode_len, visualize)

    elif agent_name == PPO_AGENT:
        agent = PPO2.load(checkpoint)
        baselines_test(agent, epochs_test, episode_len, visualize)


if __name__ == "__main__":
    load_agent(DQN, "checkpoints/dqn/5m_2exp", 5, 1000, False)
