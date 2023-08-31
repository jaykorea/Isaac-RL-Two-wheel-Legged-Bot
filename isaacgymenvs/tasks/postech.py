import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi, torch_utils
from .base.vec_task import VecTask

from isaacgymenvs.utils.torch_jit_utils import *

class Postech(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.reset_dist = self.cfg["env"]["resetDist"]

        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.max_episode_length = 200

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]

        # Observations:
        # 0:6 - joint positions
        # 6:12 - joint velocities
        # 12:15 - body position (x, y, z)
        # 15:18 - body orientation (roll, pitch, yaw)
        # 18:21 - body linear velocity (x, y, z)
        # 21:24 - body angular velocity (roll, pitch, yaw)
        num_obs = 24

        # Actions:
        # 0:6 - joint action targets
        num_acts = 6

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_acts

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)

        self.all_actor_indices = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)

        # Actors x Info
        self.root_orientations = self.root_states[:, 3:7]
        print("get_euler_xyz(self.root_states[:, 3:7])", get_euler_xyz(self.root_states[:, 3:7]))
        print("self.root_states[:, 3:7]", self.root_states[:, 3:7])
        self.root_pitch = get_euler_xyz(self.root_states[:, 3:7])[1]
        self.root_yaw = get_euler_xyz(self.root_states[:, 3:7])[2]
        self.root_roll = get_euler_xyz(self.root_states[:, 3:7])[0]

        self.root_x_d = quat_rotate_inverse(self.root_orientations, self.root_states[:, 7:10])[:, :2]
        self.root_angular_vels = self.root_states[:, 10:13] #pitch_d, yaw_d

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 2] = 0.75

        self.initial_dof_states = self.dof_state.clone()

        self.commands = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)  # Adjusted for 6 joints
        self.commands_y = self.commands.view(self.num_envs, 6)[..., 1]  # Adjusted for 6 joints
        self.commands_x = self.commands.view(self.num_envs, 6)[..., 0]  # Adjusted for 6 joints
        self.commands_yaw = self.commands.view(self.num_envs, 6)[..., 2]  # Adjusted for 6 joints

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)


    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0) if self.up_axis == 'z' else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/new_robot_design.urdf"  # Update this to point to your new robot design

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        postech_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(postech_asset)

        pose = gymapi.Transform()
        if self.up_axis == 'z':
            pose.p.z = 0.75 # ground to base_link height
            # asset is rotated z-up by default, no additional rotations needed
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        else:
            pose.p.y = 2.0
            pose.r = gymapi.Quat(-np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)

        self.postech_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            postech_handle = self.gym.create_actor(env_ptr, postech_asset, pose, "postech", i, 1, 0)

            dof_props = self.gym.get_actor_dof_properties(env_ptr, postech_handle)
            for j in range(6):  # Set the driveMode for all 6 joints
                dof_props['driveMode'][j] = gymapi.DOF_MODE_EFFORT
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, postech_handle, dof_props)

            self.envs.append(env_ptr)
            self.postech_handles.append(postech_handle)

    def compute_reward(self):
        # Retrieve environment observations from buffer

        # For joint positions (assuming the order is hip, leg, wheel - for the urdf it is leg - shin - wheel)
        hip_pos = self.obs_buf[:, [0, 3]]
        leg_pos = self.obs_buf[:, [1, 4]]
        wheel_pos = self.obs_buf[:, [2, 5]]

        # For joint velocities (assuming the order is hip, leg, wheel - for the urdf it is leg - shin - wheel)
        hip_vel = self.obs_buf[:, [6, 9]]
        leg_vel = self.obs_buf[:, [7, 10]]
        wheel_vel = self.obs_buf[:, [8, 11]]

        # For body states
        body_pos = self.obs_buf[:, 12:15]
        body_orientation = self.obs_buf[:, 15:18]
        body_linear_vel = self.obs_buf[:, 18:21]
        body_angular_vel = self.obs_buf[:, 21:24]

        base_link_height = self.root_states[:, 2]

        # Compute reward and reset conditions
        self.rew_buf[:], self.reset_buf[:] = compute_postech_reward(
            hip_pos, leg_pos, wheel_pos, hip_vel, leg_vel, wheel_vel,
            body_pos, body_orientation, body_linear_vel, body_angular_vel,
            self.commands, self.reset_buf, self.progress_buf, self.max_episode_length, base_link_height
        )

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        # Debugging: print joint positions and velocities
        #print("Debug - dof_pos: ", self.dof_pos[1])
        #print("Debug - dof_vel: ", self.dof_vel[1])

        # Joint states
        self.obs_buf[env_ids, 0:6] = self.dof_pos[env_ids]  # Joint positions
        self.obs_buf[env_ids, 6:12] = self.dof_vel[env_ids]  # Joint velocities

        # Body states
        self.obs_buf[env_ids, 12:15] = self.root_states[env_ids, 0:3]  # Body position (x, y, z)
        # Body orientation (roll, pitch, yaw) - assuming get_euler_xyz gives (roll, pitch, yaw)
        self.obs_buf[env_ids, 15:18] = torch.stack([get_euler_xyz(self.root_states[env_ids, 3:7])[0], 
                                                    get_euler_xyz(self.root_states[env_ids, 3:7])[1], 
                                                    get_euler_xyz(self.root_states[env_ids, 3:7])[2]], dim=1)
        # Body linear velocity (x, y, z)
        self.obs_buf[env_ids, 18:21] = quat_rotate_inverse(self.root_states[env_ids, 3:7], self.root_states[env_ids, 7:10])[:, :3]
        # Body angular velocity (roll, pitch, yaw)
        self.obs_buf[env_ids, 21:24] = self.root_states[env_ids, 10:13]
        
        return self.obs_buf
        
    def reset_idx(self, env_ids):
        print("################ RESET")

        num_resets = len(env_ids)
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Resetting to initial root states
        self.root_states[env_ids] = self.initial_root_states[env_ids]

        # Introduce small perturbations in roll, pitch, and yaw (comment these lines if you want no randomness)
        roll = torch_rand_float(-0.05, 0.05, (num_resets, 1), self.device).flatten()
        pitch = torch_rand_float(-0.05, 0.05, (num_resets, 1), self.device).flatten()
        yaw = torch_rand_float(-0.05, 0.05, (num_resets, 1), self.device).flatten()

        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)

        # Update the simulation environment with the new initial states
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.initial_dof_states),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # Resetting buffers
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # Randomizing initial commands (comment these lines if you want no randomness)
        self.commands_x[env_ids] = torch_rand_float(-0.2, 0.2, (len(env_ids), 1), device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(-0.2, 0.2, (len(env_ids), 1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(-0.2, 0.2, (len(env_ids), 1), device=self.device).squeeze()

        # Debugging: Log initial states
        #print("Initial Root States after Reset:", self.root_states[env_ids])
        #print("Initial DOF States after Reset:", self.initial_dof_states[env_ids])

    def check_base_link_collisions_and_reset(self):
        # Assuming Isaac Gym provides a function to get active collisions for each environment
        collisions = self.gym.get_active_collisions(self.sim)

        # Assuming the ground has a unique ID, say 'ground_id'
        ground_id = get_ground_id()  # You'll need to define this based on Isaac Gym's specifics

        # Assuming each robot's base_link has a unique identifier or name, say 'base_link'
        base_link_name = "base_link"

        # Check if robot's base_link is in collision with the ground
        for env_id in range(self.num_envs):
            robot_id = self.postech_handles[env_id]  # Assuming this gives the robot's unique ID in that environment

            # Fetch the unique ID for the robot's base_link in this environment
            base_link_id = self.gym.get_actor_rigid_body_id(robot_id, base_link_name)  # This function might be specific to Isaac Gym

            # Check for collision between robot's base_link and ground
            if (base_link_id, ground_id) in collisions[env_id]:  # Assuming collisions[env_id] gives a list of active collision pairs for that environment
                self.reset_idx([env_id])  # Resetting that specific environment
    
    def pre_physics_step(self, actions):
        
        #actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float) * self.max_push_effort
        #[::self.num_dof] = actions.to(self.device).squeeze() * self.max_push_effort
        actions = actions.to(self.device)

        torques = gymtorch.unwrap_tensor(actions* self.max_push_effort)
        self.gym.set_dof_actuation_force_tensor(self.sim, torques)

    def post_physics_step(self):
        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.compute_observations()
        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################

def compute_postech_reward(hip_pos, leg_pos, wheel_pos, hip_vel, leg_vel, wheel_vel,
                        body_pos, body_orientation, body_linear_vel, body_angular_vel,
                        commands, reset_buf, progress_buf, max_episode_length, base_link_height):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor) -> Tuple[Tensor, Tensor]

    # Constants
    k_orientation = 100.0  # Emphasizes upright orientation
    k_smooth = 1.0         # Emphasizes smooth movements
    k_effort = 0.01        # Penalizes high control efforts
    k_velocity = 1.0       # Penalizes movement
    
    # Orientation reward: Prioritizing upright stance
    reward_orientation = k_orientation * (1.0 - torch.abs(body_orientation[:, 1])) * (1.0 - torch.abs(body_orientation[:, 0]))
    
    # Wheel velocity reward: Encourage the robot to use its wheels
    reward_wheel_vel = torch.norm(wheel_vel, dim=1)
    
    # Smooth movements penalty: Encourages smooth control actions
    reward_smooth = -k_smooth * (torch.abs(body_angular_vel).sum(dim=1))
    
    # Control effort penalty: Encourages efficient actions
    reward_effort = -k_effort * torch.norm(commands, dim=1) ** 2
    
    # Velocity penalty: Discourages unnecessary movement
    reward_velocity = -k_velocity * torch.norm(body_linear_vel, dim=1)
    
    # Total reward
    reward = reward_orientation + reward_wheel_vel + reward_smooth + reward_effort + reward_velocity

    # Define thresholds for reset conditions
    pitch_threshold = np.pi / 6.0  # 30 degrees, you can adjust this
    roll_threshold = np.pi / 6.0   # 30 degrees, you can adjust this
    height_threshold = 0.5         # Assuming robot starts at height > 0.5, adjust as needed

    # Check the reset conditions based on body orientation
    reset_orientation = (torch.abs(body_orientation[:, 1]) > pitch_threshold) | (torch.abs(body_orientation[:, 0]) > roll_threshold)

    # Debugging print statements
    if torch.any(reset_orientation):
        #print("Reset due to orientation. Pitch: {}, Roll: {}".format(body_orientation[:, 1][reset_orientation], body_orientation[:, 0][reset_orientation]))
        #print("Reset due to orientation")
        pass
    # Check the reset conditions based on base link height
    reset_height = base_link_height < height_threshold

    # Debugging print statements
    if torch.any(reset_height):
        #print("Reset due to height. Height: {}".format(base_link_height[reset_height]))
        #print("Reset due to height")
        pass

    # Combine all reset conditions
    reset = reset_orientation | reset_height
    reset = torch.where(reset, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return reward, reset