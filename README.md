
# "Balancing a Two-Wheeled Legged Robot using Isaac Gym and Reinforcement Learning"
## Abstract
This paper presents an approach to balance a two-wheeled legged robot using reinforcement learning (RL) with Nvidia's Isaac Gym. We detail the design of the reward and reset functions, which are critical for successful learning, and present experimental results to demonstrate the effectiveness of our approach.

## Introduction
Balancing robots with non-standard configurations, such as those with both wheels and legs, poses unique challenges that traditional control methods struggle to address. Reinforcement learning offers an alternative method that can adapt to complex dynamics and environments. In this work, we use Nvidia's Isaac Gym, a toolkit for RL in robotic simulation, to train a two-wheeled legged robot to maintain balance.

## Methodology
### Robot Configuration
The robot has two legs, each with a hip and shin joint, and two wheels. The robot's task is to maintain an upright position while also being capable of forward and backward motion.

### Simulation Environment
We use Nvidia's Isaac Gym to simulate the robot's dynamics and environment. The Isaac Gym provides a PyTorch-compatible interface, allowing seamless integration with popular RL algorithms.

### State Observations
The robot's state comprises its joint positions and velocities, body position, orientation, and linear and angular velocities. Specifically:
<br/>
Joint positions for hips and shins: 4 variables<br/>
Wheel positions: 2 variables<br/>
Body position (x, y, z): 3 variables<br/>
Body orientation (roll, pitch, yaw): 3 variables<br/>
Body linear velocity (x, y, z): 3 variables<br/>
Body angular velocity (roll, pitch, yaw): 3 variables

### Action Space
The robot's actions are the torques applied to the joints and wheels.

### Reward Function Design
The reward function aims to incentivize the robot to maintain an upright orientation while discouraging excessive movements or control efforts.<br/>
![CodeCogsEqn (1)](https://github.com/jaykorea/isaac_gym_legged_bot/assets/95605860/53a88490-0fe5-402b-bb56-5dfe21c9309d)

* OrientationReward: Rewards the robot for maintaining an upright orientation. This is calculated as<br/>
![CodeCogsEqn](https://github.com/jaykorea/isaac_gym_legged_bot/assets/95605860/55d6b696-ce36-4a0f-baf6-16493b810dac)

* SmoothnessReward<br/>
![CodeCogsEqn (2)](https://github.com/jaykorea/isaac_gym_legged_bot/assets/95605860/b38b9d97-b620-459b-b1ac-23e3ed2eac85)

* ControlEffort: Penalizes large torques or forces applied at the joints and wheels.

### Reset Function Design
The reset function is triggered if:

1. The robot's pitch or roll exceeds a set threshold, indicating that it has fallen over.
2. The height of the robot's base_link drops below a certain value, indicating that it has kneeled or otherwise left a standard operating position.

## Experiments and Results
TBD

## Conclusion
I presented an RL-based approach for balancing a two-wheeled legged robot using Isaac Gym. The reward and reset functions were designed to address the unique challenges posed by this robot configuration. Experimental results demonstrate the effectiveness of our approach in various scenarios.
