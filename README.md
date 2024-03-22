# "Balancing a Two-Wheeled Legged Robot using Isaac Gym and Reinforcement Learning"
## Abstract
This project presents an approach to balance a two-wheeled legged robot using reinforcement learning (RL) with Nvidia's Isaac Gym. We detail the design of the reward and reset functions, which are critical for successful learning, and present experimental results to demonstrate the effectiveness of our approach.
## Demo video
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

## Methodology
### Robot Configuration
The robot has two legs, each with a hip and shin joint, and two wheels. The robot's task is to maintain an upright position while also being capable of forward and backward motion.<br/>
<22img src = "https://github.com/jaykorea/isaac_gym_legged_bot/assets/95605860/91801f7a-984f-4b02-a988-0eb07372dacb" width="70%" height="70%">

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

* forward velocity reward: Encourages forward movement. This is cacluated as
<br/> ![kforwardvelocity](https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/1799551e-35c8-41e9-8d6e-63f78fdb3a1d)

* hip alignment reward: Penalizes misalignment of the hip joints. This is cacluated as
<br/>![khip](https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/c4bd342c-b87d-4930-9c11-e74fc57e329c)

### Reset Function Design
The reset function is triggered if:
1. The robot's knees come into contact with the ground or an obstacle, indicating it has fallen or kneeled.
2. The robot's base comes into contact with the ground or an obstacle, indicating it has fallen or tilted too much.
3. The magnitude of the projected gravity in the z-axis exceeds a threshold, indicating that the robot is likely falling or has fallen over.
4. The current episode length exceeds a predefined maximum length.

* Knee Contact Reset
This reset condition is based on the sensor forces at the robot's knees. If these forces exceed a certain threshold, it indicates that the knees have made contact with the ground or an obstacle.
<br/> ![kknee](https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/1ba67e7f-6913-446d-bb92-cac13d2b7fd6)

* Base Contact Reset
This reset condition is based on the sensor forces at the robot's base. If these forces exceed a certain threshold, it indicates that the base has made contact with the ground or an obstacle.
<br/> ![base_contact](https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/74199bcb-1785-4c7b-a230-2f97dc6b61d7)

* Gravity-based Reset
This condition triggers a reset if the magnitude of the projected gravity in the z-axis exceeds a threshold.
<br/> ![gravity_reset](https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/f78ad393-be32-4aae-8edd-4ac7c2baec46)

* Episode Length-based Reset
This condition is triggered if the current episode length has reached or exceeded the maximum allowable episode length.
<br/> ![episode_reset](https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/63987917-7ecd-445a-90e3-c7f991430c52)

* Final Reset Condition
The final reset condition is a logical OR operation combining the Knee Contact Reset, Base Contact Reset, Gravity-based Reset, and Episode Length-based Reset.
<br/> ![reset](https://github.com/jaykorea/Isaac-gym-Legged-Bot/assets/95605860/d63ca05e-20d8-4fba-a8c0-381cd625f151)


- 'knee_contact' and 'base_contact' are the contact information from the sensor forces at the robot's knees and base, respectively.
- 'projected_gravity' is the projected gravity value in the z-axis.
- 'progress_buf' is the current episode length.
- 'max_episode_length' is the maximum allowed episode length.

If any of theses conditions are met, the robot's environment will be reset.

## Conclusion
I presented an RL-based approach for balancing a two-wheeled legged robot using Isaac Gym. The reward and reset functions were designed to address the unique challenges posed by this robot configuration. Experimental results demonstrate the effectiveness of our approach in various scenarios.
