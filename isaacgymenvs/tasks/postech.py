import numpy as np
import os, time
import torch

from isaacgym import gymutil, gymtorch, gymapi, torch_utils
from .base.vec_task import VecTask

from isaacgymenvs.utils.torch_jit_utils import *

class Postech(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.custom_origins = False
        self.reset_dist = self.cfg["env"]["resetDist"]
        self.height_samples = None
        self.custom_origins = False
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.height_meas_scale = self.cfg["env"]["learn"]["heightMeasurementScale"]

        self.max_push_effort = self.cfg["env"]["maxEffort"]
        #self.max_episode_length = self.cfg["env"]["learn"]["episodeLength_m"]

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]

        self.init_done = False
        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang
        
        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        # other
        self.decimation = self.cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self.cfg["sim"]["dt"]
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s/ self.dt + 0.5)
        print("max_episode_length: ", self.max_episode_length)
        self.push_interval = int(self.cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        self.allow_knee_contacts = self.cfg["env"]["learn"]["allowKneeContacts"]
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]
        self.curriculum = self.cfg["env"]["terrain"]["curriculum"]

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
        
        #command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]
        # Joint command ranges
        self.hip_l_range = self.cfg["env"]["randomJointVelocityRanges"]["hip_l_range"]
        self.knee_l_range = self.cfg["env"]["randomJointVelocityRanges"]["knee_l_range"]
        self.wheel_l_range = self.cfg["env"]["randomJointVelocityRanges"]["wheel_l_range"]
        self.hip_r_range = self.cfg["env"]["randomJointVelocityRanges"]["hip_r_range"]
        self.knee_r_range = self.cfg["env"]["randomJointVelocityRanges"]["knee_r_range"]
        self.wheel_r_range = self.cfg["env"]["randomJointVelocityRanges"]["wheel_r_range"]

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        if self.graphics_device_id != -1:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        
        # get gym GPU state tensors
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.all_actor_indices = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.feet_air_time = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.height_points = self.init_height_points()
        self.measured_heights = None

        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) 
        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]
       
        # joint positions offsets
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        # for i in range(self.num_actions):
        #     name = self.dof_names[i]
        #     angle = self.named_default_joint_angles[name]
        #     self.default_dof_pos[:, i] = angle

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.init_done = True

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.up_axis = self.cfg["sim"]["up_axis"]
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        terrain_type = self.cfg["env"]["terrain"]["terrainType"] 
        if terrain_type=='plane':
            self._create_ground_plane()
        elif terrain_type=='trimesh':
            self._create_trimesh()
            self.custom_origins = True
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        plane_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        plane_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        plane_params.restitution = self.cfg["env"]["terrain"]["restitution"]
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        self.terrain = Terrain(self.cfg["env"]["terrain"], num_robots=self.num_envs)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.border_size 
        tm_params.transform.p.y = -self.terrain.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        tm_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        tm_params.restitution = self.cfg["env"]["terrain"]["restitution"]

        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/postech/urdf/postech.urdf" # Update this to point to your new robot design
        asset_path = os.path.join(asset_root, asset_file)
        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.collapse_fixed_joints = self.cfg["env"]["asset"]["collapseFixedJoints"]
        asset_options.replace_cylinder_with_capsule = False
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = self.cfg["env"]["asset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        postech_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(postech_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(postech_asset)

        # prepare friction randomization
        rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(postech_asset)
        friction_range = self.cfg["env"]["learn"]["frictionRange"]
        num_buckets = 100
        friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device=self.device)

        self.base_init_state = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(postech_asset)
        self.dof_names = self.gym.get_asset_dof_names(postech_asset)
        foot_name = self.cfg["env"]["asset"]["footName"]
        knee_name = self.cfg["env"]["asset"]["kneeName"]
        base_name = self.cfg["env"]["asset"]["baseName"]
        feet_names = [s for s in body_names if foot_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if knee_name in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        base_names = [s for s in body_names if base_name in s]
        self.base_indices = torch.zeros(len(base_names), dtype=torch.long, device=self.device, requires_grad=False)

        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(postech_asset)

        # env origins
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        if not self.curriculum: self.cfg["env"]["terrain"]["maxInitMapLevel"] = self.cfg["env"]["terrain"]["numLevels"] - 1
        self.terrain_levels = torch.randint(0, self.cfg["env"]["terrain"]["maxInitMapLevel"]+1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.randint(0, self.cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device)
        if self.custom_origins:
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            spacing = 0.

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        self.postech_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            if self.custom_origins:
                self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
                pos = self.env_origins[i].clone()
                pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
                start_pose.p = gymapi.Vec3(*pos)

            for s in range(len(rigid_shape_prop)):
                rigid_shape_prop[s].friction = friction_buckets[i % num_buckets]
            self.gym.set_asset_rigid_shape_properties(postech_asset, rigid_shape_prop)
            postech_handle = self.gym.create_actor(env_handle, postech_asset, start_pose, "postech", i, 0, 0)
            
            for j in range(6):  # Set the driveMode for all 6 joints
                dof_props['driveMode'][j] = gymapi.DOF_MODE_EFFORT
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
            self.gym.set_actor_dof_properties(env_handle, postech_handle, dof_props)
            self.envs.append(env_handle)
            self.postech_handles.append(postech_handle)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.postech_handles[0], feet_names[i])
            print("feet_names: ", feet_names)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.postech_handles[0], knee_names[i])
            print("knee_names: ", knee_names)

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.postech_handles[0], "base_link")

    def compute_observations(self, env_ids=None):
        self.measured_heights = self.get_heights()
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.height_meas_scale

        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        # Debugging: print joint positions and velocities
        #print("Debug - dof_pos: ", self.dof_pos[1])
        #print("Debug - dof_vel: ", self.dof_vel[1])

        #####################################################
        #           dof_name[0] :  hip_l_joint              #
        #           dof_name[1] :  knee_l_joint             #
        #           dof_name[2] :  wheel_l_joint            #
        #           dof_name[3] :  hip_r_joint              #
        #           dof_name[4] :  knee_r_joint             #
        #           dof_name[5] :  wheel_r_joint            #
        #####################################################
        
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
        self.obs_buf[env_ids, 18:21] = quat_rotate_inverse(self.root_states[env_ids, 3:7], self.root_states[env_ids, 7:10])
        # Body angular velocity (roll, pitch, yaw)
        #self.obs_buf[env_ids, 21:24] = self.root_states[env_ids, 10:13]
        self.obs_buf[env_ids, 21:24] = quat_rotate_inverse(self.root_states[env_ids, 3:7], self.root_states[env_ids, 10:13])
        
        return self.obs_buf
        
    def reset_idx(self, env_ids):
        # print("################ RESET")
        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        num_resets = len(env_ids)
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        if self.custom_origins:
            self.update_terrain_level(env_ids)
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state
            
        # Resetting to initial root states
        #self.root_states[env_ids] = self.initial_root_states[env_ids]

        # Introduce small perturbations in roll, pitch, and yaw (comment these lines if you want no randomness)
        roll = torch_rand_float(-0.05, 0.05, (num_resets, 1), self.device).flatten()
        pitch = torch_rand_float(-0.05, 0.05, (num_resets, 1), self.device).flatten()
        yaw = torch_rand_float(-0.05, 0.05, (num_resets, 1), self.device).flatten()

        #self.root_states[env_ids, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)

        # Update the simulation environment with the new initial states
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        # Randomizing initial commands (comment these lines if you want no randomness)              
        self.commands[env_ids, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 2] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.25).unsqueeze(1) # set small commands to zero


        # Resetting buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def push_robots(self):
        # Only apply forces in the x-direction (forward/backward)
        self.root_states[:, 7] = torch_rand_float(-1., 1., (self.num_envs, 1), device=self.device).squeeze()
        
        # Set the y-direction velocity to 0
        #self.root_states[:, 8] = torch.zeros(self.num_envs, device=self.device)

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def update_terrain_level(self, env_ids):
        if not self.init_done or not self.curriculum:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        self.terrain_levels[env_ids] -= 1 * (distance < torch.norm(self.commands[env_ids, :2])*self.max_episode_length_s*0.25)
        self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def pre_physics_step(self, actions):

        # Map the commands to wheel torques
        forward_command = self.commands[:, 0]
        yaw_command = self.commands[:, 2]
        
        # Constant torque to move both wheels forward
        forward_torque = torch.tensor(10.0)  # Set to a constant value or calculate dynamically
    

        left_wheel_torque = 1.5 * torch.abs(forward_command - yaw_command)
        right_wheel_torque = 1.5 * torch.abs(forward_command + yaw_command)

        # Update the actions tensor with the new wheel torques
        # According to the dof_name order, the wheel joints are at indices 2 and 5

        # Convert to the device and scale by max effort
        actions = actions.to(self.device)
        #actions[:, 2] *= 20.0
        #actions[:, 5] *= 20.0
        
        torques = gymtorch.unwrap_tensor(actions * self.max_push_effort)

        # Set the actuation
        self.gym.set_dof_actuation_force_tensor(self.sim, torques)
        self.gym.refresh_dof_state_tensor(self.sim)

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
        self.progress_buf += 1
        self.randomize_buf += 1
        self.common_step_counter += 1
        
        if self.common_step_counter % self.push_interval == 0:
            if self.cfg["env"]["learn"]["pushRobots"] == True:
                self.push_robots()
                #pass

        # prepare quantities
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 2] - heading), -1., 1.)
        
        self.compute_reward()
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
            
        self.compute_observations()
        
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            # draw height lines
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
            for i in range(self.num_envs):
                base_pos = (self.root_states[i, :3]).cpu().numpy()
                heights = self.measured_heights[i].cpu().numpy()
                height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
                for j in range(heights.shape[0]):
                    x = height_points[j, 0] + base_pos[0]
                    y = height_points[j, 1] + base_pos[1]
                    z = heights[j]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
        
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

    def init_height_points(self):
        # 1mx1.6m rectangle (without center line)
        y = 0.1 * torch.tensor([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], device=self.device, requires_grad=False) # 10-50cm on each side
        x = 0.1 * torch.tensor([-8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8], device=self.device, requires_grad=False) # 20-80cm on each side
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def get_heights(self, env_ids=None):
        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg["env"]["terrain"]["terrainType"] == 'none':
            raise NameError("Can't measure height with terrain type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)
 
        points += self.terrain.border_size
        points = (points/self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]

        heights2 = self.height_samples[px+1, py+1]
        heights = torch.min(heights1, heights2)

        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale

    def compute_reward(self):
        # Retrieve environment observations from buffer

        # For joint positions (assuming the order is hip, knee, wheel)
        hip_pos = self.obs_buf[:, [0, 3]]
        knee_pos = self.obs_buf[:, [1, 4]]
        wheel_pos = self.obs_buf[:, [2, 5]]

        # For joint velocities (assuming the order is hip, knee, wheel)
        hip_vel = self.obs_buf[:, [6, 9]]
        knee_vel = self.obs_buf[:, [7, 10]]
        wheel_vel = self.obs_buf[:, [8, 11]]

        # For body states
        body_pos = self.obs_buf[:, 12:15]
        body_orientation = self.obs_buf[:, 15:18]
        body_linear_vel = self.obs_buf[:, 18:21]
        body_angular_vel = self.obs_buf[:, 21:24]

        base_link_height = self.root_states[:, 2]
    
        if not self.allow_knee_contacts:
            knee_contact = torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 2.
            base_contact = torch.norm(self.contact_forces[:, self.base_indices, :], dim=2) > 2.
            #print("contact_forces: ", self.contact_forces)

        # Compute reward and reset conditions
        self.rew_buf[:], self.reset_buf[:] = self.compute_postech_reward(
            hip_pos, knee_pos, wheel_pos, hip_vel, knee_vel, wheel_vel,
            body_pos, body_orientation, body_linear_vel, body_angular_vel,
            self.commands, self.reset_buf, self.progress_buf, self.max_episode_length, base_link_height, knee_contact, base_contact, self.projected_gravity
        )

    def compute_postech_reward(self, hip_pos, knee_pos, wheel_pos, hip_vel, knee_vel, wheel_vel,
                                 body_pos, body_orientation, body_linear_vel, body_angular_vel,
                                 commands, reset_buf, progress_buf, max_episode_length, base_link_height, knee_contact, base_contact, projected_gravity):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]

        k_orientation = 1.0  # Emphasizes upright orientation
        k_stability = -1.5  # Weight for stability
        k_gravity = 1.0
        k_smooth = -1.0         # Emphasizes smooth movements
        k_effort = -0.5        # Penalizes high control efforts
        k_air = 0.0
        k_contact = -0.1  # Weight for contact
        k_forward_velocity = 1.95  # Adjust this weight based on your requirements      
        k_hip_align = -0.5  # Weight for knee alignment
        
        # New Reward Term for Minimizing Rate of Change in Roll and Pitch
        reward_stability = k_stability * torch.norm(body_angular_vel[:, :2], dim=1)  # Assuming the first two components correspond to roll and pitch rates
   
        # Orientation reward: Prioritizing upright stance
        reward_orientation = k_orientation * (1.0 - torch.abs(body_orientation[:, 1])) * (1.0 - torch.abs(body_orientation[:, 0]))

        # orientation penalty
        rew_gravity = k_gravity * torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

        # Smooth movements penalty: Encourages smooth control actions
        reward_smooth = k_smooth * (torch.abs(body_angular_vel).sum(dim=1))

        # Control effort penalty: Encourages efficient actions
        reward_effort = k_effort * torch.norm(commands, dim=1) ** 2
        
        # Penalizes hip joint misalignment
        reward_hip_alignment = k_hip_align * torch.abs(hip_pos[:, 0] - hip_pos[:, 1])

        # Base Contact Penalty
        reward_base_contact = k_contact * torch.any(base_contact, dim=1).float()
        
        # Base Contact Penalty
        reward_knee_contact = k_contact * torch.any(knee_contact, dim=1).float()
        
        # Forward Velocity Tracking
        reward_forward_velocity = k_forward_velocity * body_linear_vel[:, 0]

        # air time reward
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        first_contact = (self.feet_air_time > 0.) * contact
        self.feet_air_time += self.dt
        reward_air_time = k_air * torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        reward_air_time *= torch.norm(self.commands[:, :1], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact

        # Total reward
        reward = reward_stability + rew_gravity + reward_orientation + reward_smooth + reward_effort + reward_air_time + reward_base_contact + reward_knee_contact + reward_forward_velocity + reward_hip_alignment
        
        reset_gravity = torch.any(self.projected_gravity[:, 2] > 7.0)
        # Combine all reset conditions
        reset = torch.any(knee_contact, dim=1) | torch.any(base_contact, dim=1) | reset_gravity
        reset = torch.where(reset, torch.ones_like(reset_buf), reset_buf)
        reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

        return reward, reset

# terrain generator
from isaacgym.terrain_utils import *
class Terrain:
    def __init__(self, cfg, num_robots) -> None:

        self.type = cfg["terrainType"]
        if self.type in ["none", 'plane']:
            return
        self.horizontal_scale = 0.1
        self.vertical_scale = 0.005
        self.border_size = 1
        self.num_per_env = 2
        self.env_length = cfg["mapLength"]
        self.env_width = cfg["mapWidth"]
        self.proportions = [np.sum(cfg["terrainProportions"][:i+1]) for i in range(len(cfg["terrainProportions"]))]
        self.difficulty_height = cfg["difficulty_height"]

        self.env_rows = cfg["numLevels"]
        self.env_cols = cfg["numTerrains"]
        self.num_maps = self.env_rows * self.env_cols
        self.num_per_env = int(num_robots / self.num_maps)
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.border = int(self.border_size/self.horizontal_scale)
        self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg["curriculum"]:
            self.curiculum(num_robots, num_terrains=self.env_cols, num_levels=self.env_rows)
        else:
            self.randomized_terrain()   
        self.heightsamples = self.height_field_raw
        self.vertices, self.triangles = convert_heightfield_to_trimesh(self.height_field_raw, self.horizontal_scale, self.vertical_scale, cfg["slopeTreshold"])
    
    def randomized_terrain(self):
        for k in range(self.num_maps):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

            # Heightfield coordinate system from now on
            start_x = self.border + i * self.length_per_env_pixels
            end_x = self.border + (i + 1) * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            end_y = self.border + (j + 1) * self.width_per_env_pixels

            terrain = SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)
            choice = np.random.uniform(0, 1)
            if choice < 0.1:
                if np.random.choice([0, 1]):
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
                    random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.05, downsampled_scale=0.2)
                else:
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
            elif choice < 0.6:
                # step_height = np.random.choice([-0.18, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.18])
                step_height = np.random.choice([-0.15, 0.15])
                pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
            elif choice < 1.:
                discrete_obstacles_terrain(terrain, 0.15, 1., 2., 40, platform_size=3.)

            self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length/2. - 1) / self.horizontal_scale)
            x2 = int((self.env_length/2. + 1) / self.horizontal_scale)
            y1 = int((self.env_width/2. - 1) / self.horizontal_scale)
            y2 = int((self.env_width/2. + 1) / self.horizontal_scale)
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*self.vertical_scale
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def curiculum(self, num_robots, num_terrains, num_levels):
        num_robots_per_map = int(num_robots / num_terrains)
        left_over = num_robots % num_terrains
        idx = 0
        for j in range(num_terrains):
            for i in range(num_levels):
                terrain = SubTerrain("terrain",
                                    width=self.width_per_env_pixels,
                                    length=self.width_per_env_pixels,
                                    vertical_scale=self.vertical_scale,
                                    horizontal_scale=self.horizontal_scale)
                difficulty = i / (num_levels * self.difficulty_height)
                choice = j / num_terrains

                slope = difficulty * 0.4
                step_height = 0.05 + 0.175 * difficulty
                discrete_obstacles_height = 0.025 + difficulty * 0.15
                stepping_stones_size = 2 - 1.8 * difficulty
                if choice < self.proportions[0]:
                    if choice < 0.05:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                elif choice < self.proportions[1]:
                    if choice < 0.15:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                    random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.025, downsampled_scale=0.2)
                elif choice < self.proportions[3]:
                    if choice<self.proportions[2]:
                        step_height *= -1
                    pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
                elif choice < self.proportions[4]:
                    discrete_obstacles_terrain(terrain, discrete_obstacles_height, 1., 2., 40, platform_size=3.)
                else:
                    stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=0.1, max_height=0., platform_size=3.)

                # Heightfield coordinate system
                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = self.border + (j + 1) * self.width_per_env_pixels
                self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

                robots_in_map = num_robots_per_map
                if j < left_over:
                    robots_in_map +=1

                env_origin_x = (i + 0.5) * self.env_length
                env_origin_y = (j + 0.5) * self.env_width
                x1 = int((self.env_length/2. - 1) / self.horizontal_scale)
                x2 = int((self.env_length/2. + 1) / self.horizontal_scale)
                y1 = int((self.env_width/2. - 1) / self.horizontal_scale)
                y2 = int((self.env_width/2. + 1) / self.horizontal_scale)
                env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*self.vertical_scale
                self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles