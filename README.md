# Research of Tutankham game

The repository is an effect of analysis on deep reinforcement learning algorithms:
- Deep Q-Network, 
- Advantage Actor-Critic (A2C),
- Proximal Policy Optimization (PPO)
based on Tutankham game. 

Tutankham is old Atari game labirynth-type. 
The goal of project was to learn agent go throug the maze and collcect highly valued rewards (**treasures**). 
The current algorithms have a trouble to achive this goal due to simple reward source. 
By killing reviving monsters agents are able to obtain high amount of points in a short period of time, so agents have no motivation to try find traverse the labirynth.

The project introduces **conditional reward factor**, which adjust reward signal due to source of the award.

## Results
The videos show how conditional reward impacts on maze traverse:
![PPO agent (without reward modification)](https://github.com/klaudialemiec/tutankham/blob/master/video/ppo-before.gif?raw=true)
![PPO CR agent (with reward modification)](https://github.com/klaudialemiec/tutankham/blob/master/video/ppo-after.gif?raw=true)

### Comaprison of tester and achived models results:

| Agent | Scores | Treasuer no. |
| --- | --- |
| Human | **648** | **20** |
| Random | 14 | 0 |
| DQN | 88 | 3 |
| A2C | 187 | 2 |
| PPO | 196 | 3 |
| DQN CR | 121 | 3 |
| A2C CR | 156 | 3 |
| PPO CR | 148 | 3 |
