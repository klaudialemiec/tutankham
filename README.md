# Research of Tutankham game

The repository is an effect of analysis on deep reinforcement learning algorithms:
- Deep Q-Network, 
- Advantage Actor-Critic (A2C),
- Proximal Policy Optimization (PPO)
based on Tutankham game. 

Tutankham is old Atari game labirynth-type. 
The goal of project was to learn agent go throug the maze and collcect highly valued rewards. 
The current algorithms have a trouble to achive this goal due to simple reward source. 
By killing reviving monsters agents are able to obtain high amount of points in a short period of time, so agents have no motivation to try find traverse the labirynth.

The project introduces **conditional reward factor**, which adjust reward signal due to source of the award.

## Results
![PPO agent (without reward modification)](./evideo/ppo_before.gif)
![PPO agent (with reward modification)](./video/ppo_after.gif)