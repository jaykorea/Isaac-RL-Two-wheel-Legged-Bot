# "Balancing a Two-Wheeled Legged Robot using Isaac Gym with RL(Reinforcement Learning)"
## Abstract
This project presents an approach to balance a two-wheeled legged robot using reinforcement learning (RL) with Nvidia's Isaac Gym. We detail the design of the reward and reset functions, which are critical for successful learning, and present experimental results to demonstrate the effectiveness of our approach.

## Demo video
### Rev 02 - stand up & balancing
https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/991b99c9-48e9-40f5-8cbe-47b1534bf3d7
### Rev 02 - stand up & drive
https://github.com/jaykorea/Isaac-gym-Two-wheel-Legged-Bot/assets/95605860/b9f55d9f-e6c5-48fc-b0ce-113310b360bf
### Stationary
https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/7a667e89-0f2b-4cd6-8566-e1fbaf1d21b8
### Plane forward
https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/6a15511e-4872-4440-8ddf-89077e9e4513
### Terrain - stair up
https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/f4b122b7-3ec5-4e63-95a3-1e1fc1f1b2d7
### Terrain - stair down
https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/3aaf25ed-b8df-44f7-b84c-325c7b0c6570
### Terrain - rocky & slope
https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/6db75aa1-4438-44c2-b878-d37c775f41fa
### Terrain
https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/a83e79f0-fbac-4c58-94eb-305d6e184868
## Introduction
Balancing robots with non-standard configurations, such as those with both wheels and legs, poses unique challenges that traditional control methods struggle to address. Reinforcement learning offers an alternative method that can adapt to complex dynamics and environments. In this work, I use Nvidia's Isaac Gym, a toolkit for RL in robotic simulation, to train a two-wheeled legged robot to maintain balance.

## Robot config
<p align="center">
  <img src="https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/e5b2e72f-7782-4d77-b834-cee85585e36b" width="50%" height="50%">
</p>

## Reward viz
* You can simply set the 'enableRewardVis' param to True in postech.yaml configuration file
<br/><br/>
<p align="center">
  <img src="https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/4c09772e-eba6-4137-b672-1c7a431408f1" width="30%" height="30%" align="center">
</p>

## Methodology
### Reward function
see here
```
https://literate-cyclone-7bd.notion.site/Reward-function-Eng-6ba1948f61694c019dc61ee347e0451f
```
<p align="center">
  <img src="https://github.com/jaykorea/Isaac-gym-Two-wheel-Legged-Bot/assets/95605860/bd2fe9d6-05d2-41f0-8d08-623b6e95e592" width="30% height="30%">
</p>

### Robot Configuration
TBD

## Conclusion
I presented an RL-based approach for balancing a two-wheeled legged robot using Isaac Gym. The reward and reset functions were designed to address the unique challenges posed by this robot configuration. Experimental results demonstrate the effectiveness of our approach in various scenarios.
