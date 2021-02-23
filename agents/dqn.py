from rl.agents.dqn import DQNAgent
import gym
from processors.keras_atari import AtariProcessor
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from networks.keras_network import Network
from agents.dqn_agent import create_agent
from callbacks.keras_callbacks import SubTensorBoard, TestCallback
from rl.callbacks import ModelIntervalCheckpoint
import numpy as np
from keras.optimizers import Adam
from agents.test import print_metrics


def create_agent(
    network,
    processor,
    nb_actions,
    policy,
    memory,
    batch_size=32,
    nb_steps_warmup=32,
    gamma=0.99,
    target_model_update=10000,
    train_interval=4,
    delta_clip=1.0,
    enable_double_dqn=False,
    enable_dueling_network=False,
):

    return DQNAgent(
        model=network,
        processor=processor,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory,
        batch_size=batch_size,
        enable_double_dqn=enable_double_dqn,
        enable_dueling_network=enable_dueling_network,
        nb_steps_warmup=nb_steps_warmup,
        gamma=gamma,
        target_model_update=target_model_update,
        train_interval=train_interval,
        delta_clip=delta_clip,
    )


def create_dqn_agent(
    memory_capacity=500000,
    exploration_max=1.0,
    exploration_min=0.1,
    exploration_test=0,
    exploration_steps=1e6,
    frame_shape=(84, 84),
    window_length=4,
):
    env = gym.make("Tutankham-v4")
    nb_actions = env.action_space.n
    processor = AtariProcessor(frame_shape)
    memory = SequentialMemory(limit=memory_capacity, window_length=window_length)
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=exploration_max,
        value_min=exploration_min,
        value_test=exploration_test,
        nb_steps=exploration_steps,
    )
    network = Network(frame_shape, window_length, nb_actions)
    model = network.create_model()

    return create_agent(model, processor, nb_actions, policy, memory)


def train_dqn(
    agent, optimizer, train_episodes, episode_len, logdir, checkpoint, verbose_flag=1
):
    tb_callback = [SubTensorBoard(logdir=logdir)]
    tb_callback += [ModelIntervalCheckpoint(checkpoint, 10000)]
    agent.compile(optimizer)
    env = gym.make("Tutankham-v4")
    agent.fit(
        env,
        visualize=False,
        nb_steps=train_episodes,
        verbose=verbose_flag,
        nb_max_episode_steps=episode_len,
        callbacks=tb_callback,
    )


def test_dqn(agent, num_episodes, episode_len, visualize):
    env = gym.make("Tutankham-v4")
    test_callback = TestCallback()
    agent.test(
        env,
        callbacks=[test_callback],
        nb_episodes=num_episodes,
        visualize=visualize,
        nb_max_episode_steps=episode_len,
        verbose=0,
    )

    rewards_list = np.array(test_callback.rewards_list)
    keys_list = np.array(test_callback.keys_list)
    keys_reward_list = np.array(test_callback.keys_reward_list)
    timesteps_list = np.array(test_callback.timesteps_list)

    print_metrics(rewards_list, keys_list, keys_reward_list, timesteps_list)


def train_test_dqn(
    exploration_steps,
    epochs_train,
    checkpoint,
    epochs_test,
    episode_len_train=1000,
    episode_len_test=1000,
    test_visualize=False,
    memory_capacity=500000,
    exploration_max=1.0,
    exploration_min=0.1,
    exploration_test=0,
    logdir="board/DQN",
):

    agent = create_dqn_agent(
        memory_capacity=memory_capacity,
        exploration_max=exploration_max,
        exploration_min=exploration_min,
        exploration_test=exploration_test,
        exploration_steps=exploration_steps,
    )

    optimizer = Adam(lr=0.00025)
    train_dqn(agent, optimizer, epochs_train, episode_len_train, logdir, checkpoint)
    test_dqn(agent, epochs_test, episode_len_test, test_visualize)
