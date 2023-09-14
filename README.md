
# "Balancing a Two-Wheeled Legged Robot using Isaac Gym and Reinforcement Learning"
## Abstract
This project presents an approach to balance a two-wheeled legged robot using reinforcement learning (RL) with Nvidia's Isaac Gym. We detail the design of the reward and reset functions, which are critical for successful learning, and present experimental results to demonstrate the effectiveness of our approach.
# Demo video
<img src = "https://github.com/jaykorea/isaac_gym_legged_bot/assets/95605860/e26a70b3-2308-4bcc-b491-1f2229d04f47" width="95%" height="95%">

## Introduction
Balancing robots with non-standard configurations, such as those with both wheels and legs, poses unique challenges that traditional control methods struggle to address. Reinforcement learning offers an alternative method that can adapt to complex dynamics and environments. In this work, I use Nvidia's Isaac Gym, a toolkit for RL in robotic simulation, to train a two-wheeled legged robot to maintain balance.

## Methodology
### Robot Configuration
The robot has two legs, each with a hip and shin joint, and two wheels. The robot's task is to maintain an upright position while also being capable of forward and backward motion.<br/>
<img src = "https://github.com/jaykorea/isaac_gym_legged_bot/assets/95605860/91801f7a-984f-4b02-a988-0eb07372dacb" width="70%" height="70%">

### Simulation Environment
I use Nvidia's Isaac Gym to simulate the robot's dynamics and environment. The Isaac Gym provides a PyTorch-compatible interface, allowing seamless integration with popular RL algorithms.

### State Observations
The robot's state comprises its joint positions and velocities, body position, orientation, and linear and angular velocities. Specifically:
<br/>
- Joint positions for hips and shins: 4 variables<br/>
- Wheel positions: 2 variables<br/>
- Body position (x, y, z): 3 variables<br/>
- Body orientation (roll, pitch, yaw): 3 variables<br/>
- Body linear velocity (x, y, z): 3 variables<br/>
- Body angular velocity (roll, pitch, yaw): 3 variables

### Action Space
The robot's actions are the torques applied to the joints and wheels.

### Reward Function Design
The reward function is designed to balance multiple objectives, encouraging the robot to maintain an upright posture, move with smooth motions, exert efficient control efforts, and achieve forward velocity, among other criteria. The function is given by:
</br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![total_reward](https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/ac49dce9-ef91-4093-92ae-8c3a8ed5e82f)

* stability reward: Penalizes rapid changes in roll and pitch angles to improve stability. This is calculated as
<br/> ![kstability](https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/eb109f54-13c9-4a51-8f1e-ff8f52d68b91)

* gravity reward: Gravity-based orientation penalty. This is calculated as
</br> ![kgravity](https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/67e30506-7b58-4f90-b85a-ed45541d4a52)

* orientation reward: Encourages the robot to maintain an upright orientation. This is cacluated as
<br/> ![korientation](https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/25a540d5-89bd-458d-a870-29187091f4f8)

* smoothness reward: Penalizes rapid changes in angular velocity to encourage smooth movements. This is cacluated as
<br/> ![ksmooth](https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/eda6fecb-62b6-49dd-bea6-c59435abc837)

* effort reward: Penalizes high control effort. This is cacluated as
<br/> ![keffort](https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/b4c549a5-a412-471c-8741-88bfc01768ed)

* contact reward: Penalties for any contact with the base or knees. This is cacluated as
<br/> ![kbase](https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/580883e0-132d-46b1-9970-201fb2e6cf32)
<br/> ![kknee](https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/4d6eb96c-4bc9-46a9-b543-f42ce2a624ca)



### Reset Function Design
The reset function is triggered if:
1. The robot's pitch or roll exceeds a set threshold, indicating that it has fallen over.
2. The height of the robot's base_link drops below a certain value, indicating that it has kneeled or otherwise left a standard operating position.

* Orientation-based reset</br>
The orientation-based reset occurs if the absolute value of the pitch or the roll exceeds a predefined threshold.
<div align="center">
<img src = "https://github.com/jaykorea/isaac_gym_legged_bot/assets/95605860/a57596b0-aae6-40c4-9e25-fe1c08aa94e2" width="80%" height="80%">
</div>

* Height-based Reset</br>
The height-based reset condition is triggered if the height of the robot's base link falls below a certain threshold.
<div align="center">
<img src = "https://github.com/jaykorea/isaac_gym_legged_bot/assets/95605860/c867b068-7a31-4faa-8c5e-3ce56c3fdcf4" width="60%" height="60%">
</div>

* Final reset condition</br>
The final reset condition is a logical OR between these two conditions, as well as a condition that checks if the episode length has been exceeded
<div align="center">
<img src = "https://github.com/jaykorea/isaac_gym_legged_bot/assets/95605860/92d75890-c4dc-45ff-80b9-eb20846c0db0" width="90%" height="90%">
</div>

- pitch and roll are the current pitch and roll angles of the robot, respectively.
- pitcht_hreshold and roll_threshold are the respective thresholds for pitch and roll.
- baselink_height is the height of the robot's base link.
- height_threshold is the height threshold.
- progress_buf is the current episode length.
- maxepisodelength is the maximum allowed episode length.
If any of theses conditions are met, the robot's environment will be reset.

## Conclusion
I presented an RL-based approach for balancing a two-wheeled legged robot using Isaac Gym. The reward and reset functions were designed to address the unique challenges posed by this robot configuration. Experimental results demonstrate the effectiveness of our approach in various scenarios.
