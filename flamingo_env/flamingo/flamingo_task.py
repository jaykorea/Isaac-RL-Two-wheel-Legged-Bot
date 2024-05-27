import numpy as np
import cv2
from typing import Tuple, Dict
from isaacgym import gymutil, gymtorch, gymapi, torch_utils
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import *
import os, time

class Flamingo(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.reset_dist = self.cfg["env"]["resetDist"]
        self.height_samples = None
        self.custom_origins = False
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.reward_vis = self.cfg["env"]["enableRewardVis"]
        self.test_mode = self.cfg["env"]["test_mode"]
        self.scale_effort_joints = self.cfg["env"]["scaleEffortJoints"]
        self.scale_effort_wheels = self.cfg["env"]["scaleEffortWheels"]
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]

        self.init_done = False

        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.height_meas_scale = self.cfg["env"]["learn"]["heightMeasurementScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        #* Load Reward scale factor params -JH
        self.reward_scale = {}
        self.reward_scale["alive_time"] = self.cfg["env"]["learn"]["reward_gain"]["alive_time"]
        self.reward_scale["orientation"] = self.cfg["env"]["learn"]["reward_gain"]["orientation"]
        self.reward_scale["lin_vel_xy"] = self.cfg["env"]["learn"]["reward_gain"]["lin_vel_xy"]
        self.reward_scale["ang_vel_z"] = self.cfg["env"]["learn"]["reward_gain"]["ang_vel_z"]
        self.reward_scale["wheel_vel_x"] = self.cfg["env"]["learn"]["reward_gain"]["wheel_vel_x"]
        self.reward_scale["wheel_vel_z"] = self.cfg["env"]["learn"]["reward_gain"]["wheel_vel_z"]
        self.reward_scale["lin_vel_z"] = self.cfg["env"]["learn"]["reward_gain"]["lin_vel_z"]
        self.reward_scale["ang_vel_xy"] = self.cfg["env"]["learn"]["reward_gain"]["ang_vel_xy"]
        self.reward_scale["gravity"] = self.cfg["env"]["learn"]["reward_gain"]["gravity"]
        self.reward_scale["contact"] = self.cfg["env"]["learn"]["reward_gain"]["contact"]
        self.reward_scale["height"] = self.cfg["env"]["learn"]["reward_gain"]["height"]
        self.reward_scale["target_height"] = self.cfg["env"]["learn"]["reward_gain"]["target_height"]
        self.reward_scale["hip_align"] = self.cfg["env"]["learn"]["reward_gain"]["hip_align"]
        self.reward_scale["des_hip"] = self.cfg["env"]["learn"]["reward_gain"]["des_hip"]
        self.reward_scale["shoulder_align"] = self.cfg["env"]["learn"]["reward_gain"]["shoulder_align"]
        self.reward_scale["leg_align"] = self.cfg["env"]["learn"]["reward_gain"]["leg_align"]
        self.reward_scale["position"] = self.cfg["env"]["learn"]["reward_gain"]["position"]
        self.reward_scale["heading"] = self.cfg["env"]["learn"]["reward_gain"]["heading"]
        self.reward_scale["torque"] = self.cfg["env"]["learn"]["reward_gain"]["torque"]
        self.reward_scale["joint_acc"] = self.cfg["env"]["learn"]["reward_gain"]["joint_acc"]
        self.reward_scale["act_rate"] = self.cfg["env"]["learn"]["reward_gain"]["act_rate"]
        self.reward_scale["shoulder_pos"] = self.cfg["env"]["learn"]["reward_gain"]["shoulder_pos"]
        self.reward_scale["leg_pos"] = self.cfg["env"]["learn"]["reward_gain"]["leg_pos"]

        # base init state
        self.setRandomJointOffset = self.cfg["env"]["setRandomJointOffset"]
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang
        
        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        # target joint positions
        self.named_target_joint_angles = self.cfg["env"]["defaultJointAngles"]

        # other
        self.decimation = self.cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self.cfg["sim"]["dt"]
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        print("max_episode_length: ", self.max_episode_length)
        self.push_interval = int(self.cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        print("push_interval: ", self.push_interval)
        self.allow_legs_contacts = self.cfg["env"]["learn"]["allowLegsContacts"]
        self.always_positive_reward = self.cfg["env"]["learn"]["alwaysPositiveReward"]
        self.usePd = self.cfg["env"]["control"]["use_pd"]
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]
        self.curriculum = self.cfg["env"]["terrain"]["curriculum"]
        self.num_terrain_levels = self.cfg["env"]["terrain"]["numLevels"]
        self.num_terrain_types = self.cfg["env"]["terrain"]["numTerrains"]

        # Observations:
        num_obs = self.cfg["env"]["numObservations"]
        # Actions:
        num_acts = self.cfg["env"]["numActions"]
        
        #command ranges
        self.sendRandomVelocity = self.cfg["env"]["sendRandomVelocity"]
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]
        # Joint command ranges
        # self.left_hip_range = self.cfg["env"]["randomJointVelocityRanges"]["left_hip_range"]
        # self.left_shoulder_range = self.cfg["env"]["randomJointVelocityRanges"]["left_shoulder_range"]
        # self.left_thigh_range = self.cfg["env"]["randomJointVelocityRanges"]["left_thigh_range"]
        # self.left_wheel_range = self.cfg["env"]["randomJointVelocityRanges"]["left_wheel_range"]
        # self.right_hip_range = self.cfg["env"]["randomJointVelocityRanges"]["right_hip_range"]
        # self.right_shoulder_range = self.cfg["env"]["randomJointVelocityRanges"]["right_shoulder_range"]
        # self.right_thigh_range = self.cfg["env"]["randomJointVelocityRanges"]["right_thigh_range"]
        # self.right_wheel_range = self.cfg["env"]["randomJointVelocityRanges"]["right_wheel_range"]

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
        
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        
        self.all_actor_indices = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.commands = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], device=self.device, requires_grad=False,)
        self.noisy_target_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.feet_air_time = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
    
        self.height_points = self.init_height_points()
        self.measured_heights = None
        self.approx_measured_heights = None
        # self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) 
        # self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        # self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        # self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]
       
        # joint positions offsets
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_actions): # self.num_actions
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        # joint positions offsets
        self.target_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_actions): # self.num_actions
            name = self.dof_names[i]
            angle = self.named_target_joint_angles[name]
            self.target_dof_pos[:, i] = angle

        self.privileged_obs_buf = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_obs_buf = torch.zeros_like(self.obs_buf, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_obs_buf = torch.zeros_like(self.last_obs_buf, dtype=torch.float, device=self.device, requires_grad=False)

        # reward episode sums
        self.torch_zeros = lambda : torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {
            "alive_time" : self.torch_zeros(),
            "orientation": self.torch_zeros(),
            "lin_vel_xy": self.torch_zeros(),
            "ang_vel_z": self.torch_zeros(),
            "wheel_vel_x": self.torch_zeros(),
            "wheel_vel_z": self.torch_zeros(),
            "lin_vel_z": self.torch_zeros(),
            "ang_vel_xy": self.torch_zeros(),
            "gravity": self.torch_zeros(),
            "base_contact": self.torch_zeros(),
            "legs_contact": self.torch_zeros(),
            "height": self.torch_zeros(),
            "hip_alignment": self.torch_zeros(),
            "hip_desired": self.torch_zeros(),
            "shoulder_alignment": self.torch_zeros(),
            "leg_alignment": self.torch_zeros(),
            "position": self.torch_zeros(),
            "heading": self.torch_zeros(),
            "torque": self.torch_zeros(),
            "joint_acc": self.torch_zeros(),
            "action_rate": self.torch_zeros(),
            "shoulder_pos": self.torch_zeros(),
            "leg_pos": self.torch_zeros(),
            # "reward_orientation_": self.torch_zeros(),
            # "reward_hip_": self.torch_zeros(),
            # "reward_shoulder_": self.torch_zeros(),
            # "reward_leg_": self.torch_zeros(),
            # "reward_action_" : self.torch_zeros(),
            # "reward_torque_" : self.torch_zeros(),
        }
        # self.reset_episode_sums(self.torch_zeros)
        self.total_reward_sums = {key: 0 for key in self.episode_sums.keys()}

        self.init_done = True

    def create_sim(self):
        self.up_axis_idx = self.cfg["sim"]["up_axis_idx"] # index of up axis: Y=1, Z=2
        self.up_axis = self.cfg["sim"]["up_axis"]
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        terrain_type = self.cfg["env"]["terrain"]["terrainType"] 
        if terrain_type=='plane':
            self._create_ground_plane()
        elif terrain_type=='trimesh':
            self._create_trimesh()
            self.custom_origins = True
        elif terrain_type=='postech_terrain':
            self._create_postech_terrain_curriculum()
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

    def _create_postech_terrain_curriculum(self):
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
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets")
        asset_file = "urdf/flamingo/urdf/flamingo.urdf"
        # if "asset" in self.cfg["env"]:
        #     asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
        #     asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_options = gymapi.AssetOptions()
        #asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 100000
        asset_options.replace_cylinder_with_capsule = False
        asset_options.slices_per_cylinder = 120 # previous: 60
        asset_options.vhacd_params.max_convex_hulls = 10  # using STL's max hull is 10
        asset_options.vhacd_params.max_num_vertices_per_ch = 64 # max: 6916
        asset_options.convex_decomposition_from_submeshes = True
        asset_options.override_inertia = False
        asset_options.override_com = False # 또하나의 발견이다잉~

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
        #print("asset num_dof : ", self.num_dof) num_dof 는 8개다잉~

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
        base_name = self.cfg["env"]["asset"]["baseName"]
        hip_name = self.cfg["env"]["asset"]["hipName"]
        thigh_name = self.cfg["env"]["asset"]["thighName"]
        shoulder_name = self.cfg["env"]["asset"]["shoulderName"]
        foot_name = self.cfg["env"]["asset"]["footName"]
        print("Dof names: ", self.dof_names)

        feet_names = [s for s in body_names if foot_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        shoulders_names = [s for s in body_names if shoulder_name in s]
        self.shoulders_indices = torch.zeros(len(shoulders_names), dtype=torch.long, device=self.device, requires_grad=False)
        thighs_names = [s for s in body_names if thigh_name in s]
        self.thighs_indices = torch.zeros(len(thighs_names), dtype=torch.long, device=self.device, requires_grad=False)
        hips_names = [s for s in body_names if hip_name in s]
        self.hips_indices = torch.zeros(len(hips_names), dtype=torch.long, device=self.device, requires_grad=False)
        base_names = [s for s in body_names if base_name in s]
        self.base_indices = torch.zeros(len(base_names), dtype=torch.long, device=self.device, requires_grad=False)
        # caster_names = [s for s in body_names if "caster_bottom" in s]
        # self.caster_indices = torch.zeros(len(caster_names), dtype=torch.long, device=self.device, requires_grad=False)
        #self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(postech_asset)

        # env origins
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        if not self.curriculum: self.cfg["env"]["terrain"]["maxInitMapLevel"] = self.cfg["env"]["terrain"]["numLevels"]

        if not self.curriculum:
            self.terrain_levels = torch.randint(0, self.cfg["env"]["terrain"]["maxInitMapLevel"]+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.randint(0, self.cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device)
        else:
            #* Sequentially genenrate robtos (map env to num_terrain , e.g. [0 ,1 ,2 ... , self.num_envs] map to [0, 0, ... ,2,2,2 ... , self.num_terrain_types...]  -JH
            # group_size_per_terrain_type = num_envs // self.num_terrain_types
            # self.terrain_levels = (torch.arange(self.num_envs, device=self.device) % (self.num_terrain_levels)).clamp(max=self.num_terrain_levels - 1)
            #self.terrain_types = (torch.arange(num_envs, device=self.device) // group_size_per_terrain_type).clamp(max=self.num_terrain_types - 1)
            #* Sequentially genenrate robtos (map env to num_terrain , e.g. [0 ,1 ,2 ... , self.num_envs] map to [0, 1 ,2, ... , self.num_terrain_types]  -JH
            self.terrain_levels = torch.randint(0, self.cfg["env"]["terrain"]["maxInitMapLevel"]+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.arange(self.num_envs, device=self.device) % self.num_terrain_types

                                        #* This looks like this
            #*************************************************************************************#
            #*       self.terrain type tensor                                                    *#
            #*       ([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,   *#
            #*         18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,   *#
            #*         16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,   *#
            #*         14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,   *#
            #*         12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,   *#
            #*         10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,   *#
            #*          8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,   *#
            #*          6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,   *#
            #*          4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,   *#
            #*          2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,   *#
            #*          0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,   *#
            #*         18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,   *#
            #*         16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,   *#
            #*         14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,   *#
            #*         12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,   *#
            #*         10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,   *#
            #*          8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,   *#
            #*          6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,   *#
            #*          4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,   *#
            #*          2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,   *#
            #*          0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,   *#
            #*         18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,   *#
            #*         16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,   *#
            #*         14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,   *#
            #*         12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,   *#
            #*         10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,   *#
            #*          8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,   *#
            #*          6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,   *#
            #*          4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,   *#
            #*          2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,   *#
            #*          0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,   *#
            #*         18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,   *#
            #*         16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,   *#
            #*         14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,   *#
            #*         12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,   *#
            #*         10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,   *#
            #*          8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,   *#
            #*          6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,   *#
            #*          4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,   *#
            #*          2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,   *#
            #*          0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,   *#
            #*         18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,   *#
            #*         16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,   *#
            #*         14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,   *#
            #*         12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,   *#
            #*         10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,   *#
            #*          8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,   *#
            #*          6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,   *#
            #*          4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,   *#
            #*          2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,   *#
            #*          0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,   *#
            #*         18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,   *#
            #*         16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,   *#
            #*         14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,   *#
            #*         12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,   *#
            #*         10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,   *#
            #*          8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3],          *#
            #*************************************************************************************#
 
        if self.custom_origins:
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            spacing = 0.

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        self.postech_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        
        num_per_terrain_type = self.num_envs // self.cfg["env"]["terrain"]["numTerrains"]
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
            postech_handle = self.gym.create_actor(env_handle, postech_asset, start_pose, "flamingo", i, 0, 0) #! i, 0, 0
            
            #* Set joint property with respect to each motor joints -JH
            for j in range(self.cfg["env"]["numActions"]):  # Set the driveMode for all 8 joints
                dof_props['driveMode'][j] = gymapi.DOF_MODE_EFFORT

            dof_props['stiffness'][:][3] = 0.0
            dof_props['stiffness'][:][7] = 0.0
            dof_props['damping'][:][3] = 0.0
            dof_props['damping'][:][7] = 0.0

            # print("dof_props : ", dof_props)
            # print("---------------")
            # raise

            #* TODO force=stiffness*(position－targetposition)+damping*(velocity－targetvelocity)
            self.gym.set_actor_dof_properties(env_handle, postech_handle, dof_props)
            self.envs.append(env_handle)
            self.postech_handles.append(postech_handle)

        print("********************Link names*********************************")
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.postech_handles[0], feet_names[i])
            print("feet_names: ", feet_names)
        for i in range(len(shoulders_names)):
            self.shoulders_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.postech_handles[0], shoulders_names[i])
            print("shoulders_names: ", shoulders_names)
        for i in range(len(thighs_names)):
            self.thighs_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.postech_handles[0], thighs_names[i])
            print("thighs_names: ", thighs_names)
        for i in range(len(hips_names)):
            self.hips_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.postech_handles[0], hips_names[i])
            print("hips_names: ", hips_names)
        for i in range(len(base_names)):
            self.base_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.postech_handles[0], base_names[i])
            print("base_names: ", base_names) 
        # for i in range(len(caster_names)):
        #     self.caster_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.postech_handles[0], caster_names[i])
        #    print("caster_names: ", caster_names)            
        print("********************Link names*********************************")
        print("")

        print("feet_indices: ", self.feet_indices)
        print("shoulder_indices: ", self.shoulders_indices)
        print("thighs_indices: ", self.thighs_indices)
        print("hip_indices: ", self.hips_indices)
        print("base_indices: ", self.base_indices)
        #print("caster_indices: ", self.caster_indices)
        print("********************Indices names*********************************")
        print("")

        for j in range(self.num_dof):
            if dof_props['lower'][j] > dof_props['upper'][j]:
                self.dof_limits_lower.append(dof_props['upper'][j])
                self.dof_limits_upper.append(dof_props['lower'][j])
            else:
                self.dof_limits_lower.append(dof_props['lower'][j])
                self.dof_limits_upper.append(dof_props['upper'][j])
        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        print("dof_limits_lower: ", self.dof_limits_lower)
        print("dof_limits_upper: ", self.dof_limits_upper)
        print("********************limits ranges*********************************")
        print("")

        #self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.postech_handles[0], "base_link")

    def compute_observations(self, env_ids=None):
        self.measured_heights = self.get_heights()
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.height_meas_scale
        self.approx_measured_heights = torch.mean(self.measured_heights, dim=1)
            
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)

        #* Joint pos states -JH
        self.obs_buf[env_ids, 0] = self.dof_pos[env_ids, 0]  # hip_l_joint
        self.obs_buf[env_ids, 1] = self.dof_pos[env_ids, 1]  # shoulder_l_joint
        self.obs_buf[env_ids, 2] = self.dof_pos[env_ids, 2]  # thigh_l_joint
        self.obs_buf[env_ids, 3] = self.dof_pos[env_ids, 3]  # wheel_l_joint
        self.obs_buf[env_ids, 4] = self.dof_pos[env_ids, 4]  # hip_r_joint
        self.obs_buf[env_ids, 5] = self.dof_pos[env_ids, 5]  # shoulder_r_joint
        self.obs_buf[env_ids, 6] = self.dof_pos[env_ids, 6]  # thigh_r_joint
        self.obs_buf[env_ids, 7] = self.dof_pos[env_ids, 7]  # wheel_r_joint

        #* Joint velocities -JH
        self.obs_buf[env_ids, 8] = self.dof_vel[env_ids, 0]  # hip_l_joint velocity
        self.obs_buf[env_ids, 9] = self.dof_vel[env_ids, 1]  # shoulder_l_joint velocity
        self.obs_buf[env_ids, 10] = self.dof_vel[env_ids, 2]  # thigh_l_joint velocity
        self.obs_buf[env_ids, 11] = self.dof_vel[env_ids, 3]  # wheel_l_joint velocity
        self.obs_buf[env_ids, 12] = self.dof_vel[env_ids, 4]  # hip_r_joint velocity
        self.obs_buf[env_ids, 13] = self.dof_vel[env_ids, 5]  # shoulder_r_joint velocity
        self.obs_buf[env_ids, 14] = self.dof_vel[env_ids, 6]  # thigh_r_joint velocity
        self.obs_buf[env_ids, 15] = self.dof_vel[env_ids, 7]  # wheel_r_joint velocity

        #* Body states -JH
        self.privileged_obs_buf[env_ids, 0:3] = self.root_states[env_ids, 0:3]  # Body position (x, y, z)
        #* Body orientation (roll, pitch, yaw) - assuming get_euler_xyz gives (roll, pitch, yaw) -JH
        self.obs_buf[env_ids, 16:19] = torch.stack([
            get_euler_xyz(self.root_states[env_ids, 3:7])[0],
            get_euler_xyz(self.root_states[env_ids, 3:7])[1],
            get_euler_xyz(self.root_states[env_ids, 3:7])[2]], dim=1)
        #* Body linear velocity (x, y, z) -JH
        self.obs_buf[env_ids, 19:22] = quat_rotate_inverse(self.root_states[env_ids, 3:7], self.root_states[env_ids, 7:10])
        #* Body angular velocity (roll, pitch, yaw) -JH
        self.obs_buf[env_ids, 22:25] = quat_rotate_inverse(self.root_states[env_ids, 3:7], self.root_states[env_ids, 10:13])

        #* Previous actions -JH
        self.obs_buf[env_ids, 25:33] = self.actions[env_ids]

        #* Previous Observations Stack -JH
        self.obs_buf[env_ids, 33:66] = self.last_obs_buf[env_ids, 0:33]
        self.obs_buf[env_ids, 66:99] = self.last_last_obs_buf[env_ids, 0:33]

        #* Input Commands (x, y, yaw) -JH
        self.obs_buf[env_ids, 99:102] = self.commands[env_ids, :3]
        #self.obs_buf[env_ids, 111:112] = 1 if torch.any(self.commands[env_ids, :3] != 0) else 0

        # #* Body states -JH
        # self.obs_buf[env_ids, 16:19] = self.root_states[env_ids, 0:3]  # Body position (x, y, z)
        # #* Body orientation (roll, pitch, yaw) - assuming get_euler_xyz gives (roll, pitch, yaw) -JH
        # self.obs_buf[env_ids, 19:22] = torch.stack([
        #     get_euler_xyz(self.root_states[env_ids, 3:7])[0],
        #     get_euler_xyz(self.root_states[env_ids, 3:7])[1],
        #     get_euler_xyz(self.root_states[env_ids, 3:7])[2]], dim=1)
        # #* Body linear velocity (x, y, z) -JH
        # self.obs_buf[env_ids, 22:25] = quat_rotate_inverse(self.root_states[env_ids, 3:7], self.root_states[env_ids, 7:10])
        # #* Body angular velocity (roll, pitch, yaw) -JH
        # self.obs_buf[env_ids, 25:28] = quat_rotate_inverse(self.root_states[env_ids, 3:7], self.root_states[env_ids, 10:13])

        # #* Previous actions -JH
        # self.obs_buf[env_ids, 28:36] = self.actions[env_ids]

        # #* Previous Observations Stack -JH
        # self.obs_buf[env_ids, 36:72] = self.last_obs_buf[env_ids, 0:36]
        # self.obs_buf[env_ids, 72:108] = self.last_last_obs_buf[env_ids, 0:36]

        # #* Input Commands (x, y, yaw) -JH
        # self.obs_buf[env_ids, 108:111] = self.commands[env_ids, :3]
        # #self.obs_buf[env_ids, 111:112] = 1 if torch.any(self.commands[env_ids, :3] != 0) else 0
        
        #self.obs_buf[:, :] = 0.1
        return self.obs_buf

    def reset_idx(self, env_ids):
        #* Simply randomize the position offset regardless of joint limits -JH
        #positions_offset = torch_rand_float(0.0, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        if env_ids is None:
            self.actions = torch.zeros_like(self.actions)
        else:
            self.actions[env_ids] = 0

        # Define noise standard deviation
        noise_std = 0.001  # You can adjust this value as needed
        # Generate Gaussian noise tensor for specific environments
        target_height_noise = torch.randn((len(env_ids),), device=self.device) * noise_std
        # Adjust target height with noise for specific environments
        self.noisy_target_height[env_ids] = self.reward_scale["target_height"] + target_height_noise

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        #* This part is for 'PLANE' -JH
        if not self.custom_origins:
            if not self.setRandomJointOffset:
                self.dof_pos[env_ids] = self.default_dof_pos[env_ids]
                #* Initialise robot on initial state(static) when reset -JH
                self.root_states[env_ids] = self.base_init_state
            else:
                joint_position_offset = self.get_joint_offset(env_ids, self.dof_limits_lower, self.dof_limits_upper, 256, 6, 6, 256, 6, 6, False)
                #* Randomize the initial joint offset when reset -JH
                self.dof_pos[env_ids,0] = tensor_clamp(self.default_dof_pos[env_ids,0] + joint_position_offset[:, 0], self.dof_limits_lower[0], self.dof_limits_upper[0])    #* hip_l_joint
                self.dof_pos[env_ids,1] = tensor_clamp(self.default_dof_pos[env_ids,1] + joint_position_offset[:, 1], self.dof_limits_lower[1], self.dof_limits_upper[1])    #* shoulder_l_joint
                self.dof_pos[env_ids,2] = tensor_clamp(self.default_dof_pos[env_ids,2] + joint_position_offset[:, 2], self.dof_limits_lower[2], self.dof_limits_upper[2])    #* thigh_l_joint
                self.dof_pos[env_ids,3] = self.default_dof_pos[env_ids,3] #* wheel_l_joint
                self.dof_pos[env_ids,4] = tensor_clamp(self.default_dof_pos[env_ids,4] + joint_position_offset[:, 3], self.dof_limits_lower[4], self.dof_limits_upper[4])    #* hip_r_joint
                self.dof_pos[env_ids,5] = tensor_clamp(self.default_dof_pos[env_ids,5] + joint_position_offset[:, 4], self.dof_limits_lower[5], self.dof_limits_upper[5])    #* shoulder_r_joint
                self.dof_pos[env_ids,6] = tensor_clamp(self.default_dof_pos[env_ids,6] + joint_position_offset[:, 5], self.dof_limits_lower[6], self.dof_limits_upper[6])    #* thigh_r_joint
                self.dof_pos[env_ids,7] = self.default_dof_pos[env_ids,7] #* wheel_r_joint
                #* Calculate base_link height(on Z-axis) based on dof_pos and env_ids whe6 reset -JH
                base_height_tensor = self.get_base_height(self.dof_pos, env_ids)
                self.root_states[env_ids] = self.base_init_state
                self.root_states[env_ids, 2] = base_height_tensor
        #* This part is for 'TRIMESH' -JH        
        else:
            self.update_terrain_level(env_ids)
            if not self.setRandomJointOffset:
                self.dof_pos[env_ids] = self.default_dof_pos[env_ids]
                self.root_states[env_ids] = self.base_init_state
            else:
                #* put env_ids to a group of terrain type w.r.t. self.num_terrain_types(0~19)
                terrain_groups = [[] for _ in range(self.num_terrain_types)]
                for i in range(self.num_terrain_types):
                    env_indices = torch.where(self.terrain_types == i)[0]  # Indices where terrain type is i
                    comparison_matrix = env_ids.unsqueeze(1) == env_indices.unsqueeze(0)
                    match_found = comparison_matrix.any(dim=1)
                    matched_ids = env_ids[match_found]
                    terrain_groups[i].extend(matched_ids.tolist())
                for i, group in enumerate(terrain_groups):
                    if group:
                        # Convert non-empty groups to tensors
                        tensor = torch.tensor(group, dtype=torch.int64, device=self.device)
                    else:
                        # Create an empty tensor for empty groups
                        tensor = torch.tensor([], dtype=torch.int64, device=self.device)
                    globals()[f'terrain_groups{i}'] = tensor

                #* Debug terrain_groups{i} variable - JH
                # for i in range(self.num_terrain_types):
                #     print(f"Terrain groups{i}:{globals()[f'terrain_groups{i}']}")

                #* Randomize the initial joint offset when reset -JH
                # self.dof_pos[env_ids,0] = self.default_dof_pos[env_ids,0] #* caster_top_joint
                # self.dof_pos[env_ids,1] = self.default_dof_pos[env_ids,1] #* caster_bottom_joint
                # self.dof_pos[env_ids,2] = tensor_clamp(self.default_dof_pos[env_ids,0] + joint_position_offset[:, 0], self.dof_limits_lower[0], self.dof_limits_upper[0])    #* hip_l_joint
                # self.dof_pos[env_ids,3] = tensor_clamp(self.default_dof_pos[env_ids,1] + joint_position_offset[:, 1], self.dof_limits_lower[1], self.dof_limits_upper[1])    #* shoulder_l_joint
                # self.dof_pos[env_ids,4] = tensor_clamp(self.default_dof_pos[env_ids,3] + joint_position_offset[:, 2], self.dof_limits_lower[2], self.dof_limits_upper[2])    #* thigh_l_joint
                # self.dof_pos[env_ids,5] = self.default_dof_pos[env_ids,3] #* wheel_l_joint
                # self.dof_pos[env_ids,6] = tensor_clamp(self.default_dof_pos[env_ids,4] + joint_position_offset[:, 3], self.dof_limits_lower[4], self.dof_limits_upper[4])    #* hip_r_joint
                # self.dof_pos[env_ids,7] = tensor_clamp(self.default_dof_pos[env_ids,5] + joint_position_offset[:, 4], self.dof_limits_lower[5], self.dof_limits_upper[5])    #* shoulder_r_joint
                # self.dof_pos[env_ids,8] = tensor_clamp(self.default_dof_pos[env_ids,6] + joint_position_offset[:, 5], self.dof_limits_lower[6], self.dof_limits_upper[6])    #* thigh_r_joint
                # self.dof_pos[env_ids,9] = self.default_dof_pos[env_ids,7] #* wheel_r_joint

                #* Randomize the initial joint offset w.r.t. specific terrain group when reset -JH
                if not torch.numel(terrain_groups0) == 0:
                    joint_position_offset = self.get_joint_offset(terrain_groups0, self.dof_limits_lower, self.dof_limits_upper, 128, 6, 6, 128, 6, 6, False)
                    self.dof_pos[terrain_groups0,0] = tensor_clamp(self.default_dof_pos[terrain_groups0,0] + joint_position_offset[:, 0], self.dof_limits_lower[0], self.dof_limits_upper[0])    #* hip_l_joint
                    self.dof_pos[terrain_groups0,1] = tensor_clamp(self.default_dof_pos[terrain_groups0,1] + joint_position_offset[:, 1], self.dof_limits_lower[1], self.dof_limits_upper[1])    #* shoulder_l_joint
                    self.dof_pos[terrain_groups0,2] = tensor_clamp(self.default_dof_pos[terrain_groups0,2] + joint_position_offset[:, 2], self.dof_limits_lower[2], self.dof_limits_upper[2])    #* thigh_l_joint
                    self.dof_pos[terrain_groups0,3] = self.default_dof_pos[terrain_groups0,3] #* wheel_l_joint
                    self.dof_pos[terrain_groups0,4] = tensor_clamp(self.default_dof_pos[terrain_groups0,4] + joint_position_offset[:, 3], self.dof_limits_lower[4], self.dof_limits_upper[4])    #* hip_r_joint
                    self.dof_pos[terrain_groups0,5] = tensor_clamp(self.default_dof_pos[terrain_groups0,5] + joint_position_offset[:, 4], self.dof_limits_lower[5], self.dof_limits_upper[5])    #* shoulder_r_joint
                    self.dof_pos[terrain_groups0,6] = tensor_clamp(self.default_dof_pos[terrain_groups0,6] + joint_position_offset[:, 5], self.dof_limits_lower[6], self.dof_limits_upper[6])    #* thigh_r_joint
                    self.dof_pos[terrain_groups0,7] = self.default_dof_pos[terrain_groups0,7] #* wheel_r_joint
                if not torch.numel(terrain_groups1) == 0:
                    joint_position_offset = self.get_joint_offset(terrain_groups1, self.dof_limits_lower, self.dof_limits_upper, 128, 6, 6, 128, 6, 6, False)
                    self.dof_pos[terrain_groups1,0] = tensor_clamp(self.default_dof_pos[terrain_groups1,0] + joint_position_offset[:, 0], self.dof_limits_lower[0], self.dof_limits_upper[0])    #* hip_l_joint
                    self.dof_pos[terrain_groups1,1] = tensor_clamp(self.default_dof_pos[terrain_groups1,1] + joint_position_offset[:, 1], self.dof_limits_lower[1], self.dof_limits_upper[1])    #* shoulder_l_joint
                    self.dof_pos[terrain_groups1,2] = tensor_clamp(self.default_dof_pos[terrain_groups1,2] + joint_position_offset[:, 2], self.dof_limits_lower[2], self.dof_limits_upper[2])    #* thigh_l_joint
                    self.dof_pos[terrain_groups1,3] = self.default_dof_pos[terrain_groups1,3] #* wheel_l_joint
                    self.dof_pos[terrain_groups1,4] = tensor_clamp(self.default_dof_pos[terrain_groups1,4] + joint_position_offset[:, 3], self.dof_limits_lower[4], self.dof_limits_upper[4])    #* hip_r_joint
                    self.dof_pos[terrain_groups1,5] = tensor_clamp(self.default_dof_pos[terrain_groups1,5] + joint_position_offset[:, 4], self.dof_limits_lower[5], self.dof_limits_upper[5])    #* shoulder_r_joint
                    self.dof_pos[terrain_groups1,6] = tensor_clamp(self.default_dof_pos[terrain_groups1,6] + joint_position_offset[:, 5], self.dof_limits_lower[6], self.dof_limits_upper[6])    #* thigh_r_joint
                    self.dof_pos[terrain_groups1,7] = self.default_dof_pos[terrain_groups1,7] #* wheel_r_joint
                if not torch.numel(terrain_groups2) == 0:
                    joint_position_offset = self.get_joint_offset(terrain_groups2, self.dof_limits_lower, self.dof_limits_upper, 128, 6, 6, 128, 6, 6, False)
                    self.dof_pos[terrain_groups2,0] = tensor_clamp(self.default_dof_pos[terrain_groups2,0] + joint_position_offset[:, 0], self.dof_limits_lower[0], self.dof_limits_upper[0])    #* hip_l_joint
                    self.dof_pos[terrain_groups2,1] = tensor_clamp(self.default_dof_pos[terrain_groups2,1] + joint_position_offset[:, 1], self.dof_limits_lower[1], self.dof_limits_upper[1])    #* shoulder_l_joint
                    self.dof_pos[terrain_groups2,2] = tensor_clamp(self.default_dof_pos[terrain_groups2,2] + joint_position_offset[:, 2], self.dof_limits_lower[2], self.dof_limits_upper[2])    #* thigh_l_joint
                    self.dof_pos[terrain_groups2,3] = self.default_dof_pos[terrain_groups2,3] #* wheel_l_joint
                    self.dof_pos[terrain_groups2,4] = tensor_clamp(self.default_dof_pos[terrain_groups2,4] + joint_position_offset[:, 3], self.dof_limits_lower[4], self.dof_limits_upper[4])    #* hip_r_joint
                    self.dof_pos[terrain_groups2,5] = tensor_clamp(self.default_dof_pos[terrain_groups2,5] + joint_position_offset[:, 4], self.dof_limits_lower[5], self.dof_limits_upper[5])    #* shoulder_r_joint
                    self.dof_pos[terrain_groups2,6] = tensor_clamp(self.default_dof_pos[terrain_groups2,6] + joint_position_offset[:, 5], self.dof_limits_lower[6], self.dof_limits_upper[6])    #* thigh_r_joint
                    self.dof_pos[terrain_groups2,7] = self.default_dof_pos[terrain_groups2,7] #* wheel_r_joint
                if not torch.numel(terrain_groups3) == 0:
                    joint_position_offset = self.get_joint_offset(terrain_groups3, self.dof_limits_lower, self.dof_limits_upper, 128, 6, 6, 128, 6, 6, False)
                    self.dof_pos[terrain_groups3,0] = tensor_clamp(self.default_dof_pos[terrain_groups3,0] + joint_position_offset[:, 0], self.dof_limits_lower[0], self.dof_limits_upper[0])    #* hip_l_joint
                    self.dof_pos[terrain_groups3,1] = tensor_clamp(self.default_dof_pos[terrain_groups3,1] + joint_position_offset[:, 1], self.dof_limits_lower[1], self.dof_limits_upper[1])    #* shoulder_l_joint
                    self.dof_pos[terrain_groups3,2] = tensor_clamp(self.default_dof_pos[terrain_groups3,2] + joint_position_offset[:, 2], self.dof_limits_lower[2], self.dof_limits_upper[2])    #* thigh_l_joint
                    self.dof_pos[terrain_groups3,3] = self.default_dof_pos[terrain_groups3,3] #* wheel_l_joint
                    self.dof_pos[terrain_groups3,4] = tensor_clamp(self.default_dof_pos[terrain_groups3,4] + joint_position_offset[:, 3], self.dof_limits_lower[4], self.dof_limits_upper[4])    #* hip_r_joint
                    self.dof_pos[terrain_groups3,5] = tensor_clamp(self.default_dof_pos[terrain_groups3,5] + joint_position_offset[:, 4], self.dof_limits_lower[5], self.dof_limits_upper[5])    #* shoulder_r_joint
                    self.dof_pos[terrain_groups3,6] = tensor_clamp(self.default_dof_pos[terrain_groups3,6] + joint_position_offset[:, 5], self.dof_limits_lower[6], self.dof_limits_upper[6])    #* thigh_r_joint
                    self.dof_pos[terrain_groups3,7] = self.default_dof_pos[terrain_groups3,7] #* wheel_r_joint
                if not torch.numel(terrain_groups4) == 0:
                    joint_position_offset = self.get_joint_offset(terrain_groups4, self.dof_limits_lower, self.dof_limits_upper, 128, 6, 6, 128, 6, 6, False)
                    self.dof_pos[terrain_groups4,0] = tensor_clamp(self.default_dof_pos[terrain_groups4,0] + joint_position_offset[:, 0], self.dof_limits_lower[0], self.dof_limits_upper[0])    #* hip_l_joint
                    self.dof_pos[terrain_groups4,1] = tensor_clamp(self.default_dof_pos[terrain_groups4,1] + joint_position_offset[:, 1], self.dof_limits_lower[1], self.dof_limits_upper[1])    #* shoulder_l_joint
                    self.dof_pos[terrain_groups4,2] = tensor_clamp(self.default_dof_pos[terrain_groups4,2] + joint_position_offset[:, 2], self.dof_limits_lower[2], self.dof_limits_upper[2])    #* thigh_l_joint
                    self.dof_pos[terrain_groups4,3] = self.default_dof_pos[terrain_groups4,3] #* wheel_l_joint
                    self.dof_pos[terrain_groups4,4] = tensor_clamp(self.default_dof_pos[terrain_groups4,4] + joint_position_offset[:, 3], self.dof_limits_lower[4], self.dof_limits_upper[4])    #* hip_r_joint
                    self.dof_pos[terrain_groups4,5] = tensor_clamp(self.default_dof_pos[terrain_groups4,5] + joint_position_offset[:, 4], self.dof_limits_lower[5], self.dof_limits_upper[5])    #* shoulder_r_joint
                    self.dof_pos[terrain_groups4,6] = tensor_clamp(self.default_dof_pos[terrain_groups4,6] + joint_position_offset[:, 5], self.dof_limits_lower[6], self.dof_limits_upper[6])    #* thigh_r_joint
                    self.dof_pos[terrain_groups4,7] = self.default_dof_pos[terrain_groups4,7] #* wheel_r_joint
                if not torch.numel(terrain_groups5) == 0:
                    joint_position_offset = self.get_joint_offset(terrain_groups5, self.dof_limits_lower, self.dof_limits_upper, 128, 6, 6, 128, 6, 6, False)
                    self.dof_pos[terrain_groups5,0] = tensor_clamp(self.default_dof_pos[terrain_groups5,0] + joint_position_offset[:, 0], self.dof_limits_lower[0], self.dof_limits_upper[0])    #* hip_l_joint
                    self.dof_pos[terrain_groups5,1] = tensor_clamp(self.default_dof_pos[terrain_groups5,1] + joint_position_offset[:, 1], self.dof_limits_lower[1], self.dof_limits_upper[1])    #* shoulder_l_joint
                    self.dof_pos[terrain_groups5,2] = tensor_clamp(self.default_dof_pos[terrain_groups5,2] + joint_position_offset[:, 2], self.dof_limits_lower[2], self.dof_limits_upper[2])    #* thigh_l_joint
                    self.dof_pos[terrain_groups5,3] = self.default_dof_pos[terrain_groups5,3] #* wheel_l_joint
                    self.dof_pos[terrain_groups5,4] = tensor_clamp(self.default_dof_pos[terrain_groups5,4] + joint_position_offset[:, 3], self.dof_limits_lower[4], self.dof_limits_upper[4])    #* hip_r_joint
                    self.dof_pos[terrain_groups5,5] = tensor_clamp(self.default_dof_pos[terrain_groups5,5] + joint_position_offset[:, 4], self.dof_limits_lower[5], self.dof_limits_upper[5])    #* shoulder_r_joint
                    self.dof_pos[terrain_groups5,6] = tensor_clamp(self.default_dof_pos[terrain_groups5,6] + joint_position_offset[:, 5], self.dof_limits_lower[6], self.dof_limits_upper[6])    #* thigh_r_joint
                    self.dof_pos[terrain_groups5,7] = self.default_dof_pos[terrain_groups5,7] #* wheel_r_joint
                if not torch.numel(terrain_groups6) == 0:
                    joint_position_offset = self.get_joint_offset(terrain_groups6, self.dof_limits_lower, self.dof_limits_upper, 128, 6, 6, 128, 6, 6, False)
                    self.dof_pos[terrain_groups6,0] = tensor_clamp(self.default_dof_pos[terrain_groups6,0] + joint_position_offset[:, 0], self.dof_limits_lower[0], self.dof_limits_upper[0])    #* hip_l_joint
                    self.dof_pos[terrain_groups6,1] = tensor_clamp(self.default_dof_pos[terrain_groups6,1] + joint_position_offset[:, 1], self.dof_limits_lower[1], self.dof_limits_upper[1])    #* shoulder_l_joint
                    self.dof_pos[terrain_groups6,2] = tensor_clamp(self.default_dof_pos[terrain_groups6,2] + joint_position_offset[:, 2], self.dof_limits_lower[2], self.dof_limits_upper[2])    #* thigh_l_joint
                    self.dof_pos[terrain_groups6,3] = self.default_dof_pos[terrain_groups6,3] #* wheel_l_joint
                    self.dof_pos[terrain_groups6,4] = tensor_clamp(self.default_dof_pos[terrain_groups6,4] + joint_position_offset[:, 3], self.dof_limits_lower[4], self.dof_limits_upper[4])    #* hip_r_joint
                    self.dof_pos[terrain_groups6,5] = tensor_clamp(self.default_dof_pos[terrain_groups6,5] + joint_position_offset[:, 4], self.dof_limits_lower[5], self.dof_limits_upper[5])    #* shoulder_r_joint
                    self.dof_pos[terrain_groups6,6] = tensor_clamp(self.default_dof_pos[terrain_groups6,6] + joint_position_offset[:, 5], self.dof_limits_lower[6], self.dof_limits_upper[6])    #* thigh_r_joint
                    self.dof_pos[terrain_groups6,7] = self.default_dof_pos[terrain_groups6,7] #* wheel_r_joint
                if not torch.numel(terrain_groups7) == 0:
                    joint_position_offset = self.get_joint_offset(terrain_groups7, self.dof_limits_lower, self.dof_limits_upper, 128, 6, 6, 128, 6, 6, False)
                    self.dof_pos[terrain_groups7,0] = tensor_clamp(self.default_dof_pos[terrain_groups7,0] + joint_position_offset[:, 0], self.dof_limits_lower[0], self.dof_limits_upper[0])    #* hip_l_joint
                    self.dof_pos[terrain_groups7,1] = tensor_clamp(self.default_dof_pos[terrain_groups7,1] + joint_position_offset[:, 1], self.dof_limits_lower[1], self.dof_limits_upper[1])    #* shoulder_l_joint
                    self.dof_pos[terrain_groups7,2] = tensor_clamp(self.default_dof_pos[terrain_groups7,2] + joint_position_offset[:, 2], self.dof_limits_lower[2], self.dof_limits_upper[2])    #* thigh_l_joint
                    self.dof_pos[terrain_groups7,3] = self.default_dof_pos[terrain_groups7,3] #* wheel_l_joint
                    self.dof_pos[terrain_groups7,4] = tensor_clamp(self.default_dof_pos[terrain_groups7,4] + joint_position_offset[:, 3], self.dof_limits_lower[4], self.dof_limits_upper[4])    #* hip_r_joint
                    self.dof_pos[terrain_groups7,5] = tensor_clamp(self.default_dof_pos[terrain_groups7,5] + joint_position_offset[:, 4], self.dof_limits_lower[5], self.dof_limits_upper[5])    #* shoulder_r_joint
                    self.dof_pos[terrain_groups7,6] = tensor_clamp(self.default_dof_pos[terrain_groups7,6] + joint_position_offset[:, 5], self.dof_limits_lower[6], self.dof_limits_upper[6])    #* thigh_r_joint
                    self.dof_pos[terrain_groups7,7] = self.default_dof_pos[terrain_groups7,7] #* wheel_r_joint
                if not torch.numel(terrain_groups8) == 0:
                    joint_position_offset = self.get_joint_offset(terrain_groups8, self.dof_limits_lower, self.dof_limits_upper, 128, 6, 6, 128, 6, 6, False)
                    self.dof_pos[terrain_groups8,0] = tensor_clamp(self.default_dof_pos[terrain_groups8,0] + joint_position_offset[:, 0], self.dof_limits_lower[0], self.dof_limits_upper[0])    #* hip_l_joint
                    self.dof_pos[terrain_groups8,1] = tensor_clamp(self.default_dof_pos[terrain_groups8,1] + joint_position_offset[:, 1], self.dof_limits_lower[1], self.dof_limits_upper[1])    #* shoulder_l_joint
                    self.dof_pos[terrain_groups8,2] = tensor_clamp(self.default_dof_pos[terrain_groups8,2] + joint_position_offset[:, 2], self.dof_limits_lower[2], self.dof_limits_upper[2])    #* thigh_l_joint
                    self.dof_pos[terrain_groups8,3] = self.default_dof_pos[terrain_groups8,3] #* wheel_l_joint
                    self.dof_pos[terrain_groups8,4] = tensor_clamp(self.default_dof_pos[terrain_groups8,4] + joint_position_offset[:, 3], self.dof_limits_lower[4], self.dof_limits_upper[4])    #* hip_r_joint
                    self.dof_pos[terrain_groups8,5] = tensor_clamp(self.default_dof_pos[terrain_groups8,5] + joint_position_offset[:, 4], self.dof_limits_lower[5], self.dof_limits_upper[5])    #* shoulder_r_joint
                    self.dof_pos[terrain_groups8,6] = tensor_clamp(self.default_dof_pos[terrain_groups8,6] + joint_position_offset[:, 5], self.dof_limits_lower[6], self.dof_limits_upper[6])    #* thigh_r_joint
                    self.dof_pos[terrain_groups8,7] = self.default_dof_pos[terrain_groups8,7] #* wheel_r_joint
                if not torch.numel(terrain_groups9) == 0:
                    joint_position_offset = self.get_joint_offset(terrain_groups9, self.dof_limits_lower, self.dof_limits_upper, 128, 6, 6, 128, 6, 6, False)
                    self.dof_pos[terrain_groups9,0] = tensor_clamp(self.default_dof_pos[terrain_groups9,0] + joint_position_offset[:, 0], self.dof_limits_lower[0], self.dof_limits_upper[0])    #* hip_l_joint
                    self.dof_pos[terrain_groups9,1] = tensor_clamp(self.default_dof_pos[terrain_groups9,1] + joint_position_offset[:, 1], self.dof_limits_lower[1], self.dof_limits_upper[1])    #* shoulder_l_joint
                    self.dof_pos[terrain_groups9,2] = tensor_clamp(self.default_dof_pos[terrain_groups9,2] + joint_position_offset[:, 2], self.dof_limits_lower[2], self.dof_limits_upper[2])    #* thigh_l_joint
                    self.dof_pos[terrain_groups9,3] = self.default_dof_pos[terrain_groups9,3] #* wheel_l_joint
                    self.dof_pos[terrain_groups9,4] = tensor_clamp(self.default_dof_pos[terrain_groups9,4] + joint_position_offset[:, 3], self.dof_limits_lower[4], self.dof_limits_upper[4])    #* hip_r_joint
                    self.dof_pos[terrain_groups9,5] = tensor_clamp(self.default_dof_pos[terrain_groups9,5] + joint_position_offset[:, 4], self.dof_limits_lower[5], self.dof_limits_upper[5])    #* shoulder_r_joint
                    self.dof_pos[terrain_groups9,6] = tensor_clamp(self.default_dof_pos[terrain_groups9,6] + joint_position_offset[:, 5], self.dof_limits_lower[6], self.dof_limits_upper[6])    #* thigh_r_joint
                    self.dof_pos[terrain_groups9,7] = self.default_dof_pos[terrain_groups9,7] #* wheel_r_joint
                if not torch.numel(terrain_groups10) == 0:
                    joint_position_offset = self.get_joint_offset(terrain_groups10, self.dof_limits_lower, self.dof_limits_upper, 128, 6, 6, 128, 6, 6, False)
                    self.dof_pos[terrain_groups10,0] = tensor_clamp(self.default_dof_pos[terrain_groups10,0] + joint_position_offset[:, 0], self.dof_limits_lower[0], self.dof_limits_upper[0])    #* hip_l_joint
                    self.dof_pos[terrain_groups10,1] = tensor_clamp(self.default_dof_pos[terrain_groups10,1] + joint_position_offset[:, 1], self.dof_limits_lower[1], self.dof_limits_upper[1])    #* shoulder_l_joint
                    self.dof_pos[terrain_groups10,2] = tensor_clamp(self.default_dof_pos[terrain_groups10,2] + joint_position_offset[:, 2], self.dof_limits_lower[2], self.dof_limits_upper[2])    #* thigh_l_joint
                    self.dof_pos[terrain_groups10,3] = self.default_dof_pos[terrain_groups10,3] #* wheel_l_joint
                    self.dof_pos[terrain_groups10,4] = tensor_clamp(self.default_dof_pos[terrain_groups10,4] + joint_position_offset[:, 3], self.dof_limits_lower[4], self.dof_limits_upper[4])    #* hip_r_joint
                    self.dof_pos[terrain_groups10,5] = tensor_clamp(self.default_dof_pos[terrain_groups10,5] + joint_position_offset[:, 4], self.dof_limits_lower[5], self.dof_limits_upper[5])    #* shoulder_r_joint
                    self.dof_pos[terrain_groups10,6] = tensor_clamp(self.default_dof_pos[terrain_groups10,6] + joint_position_offset[:, 5], self.dof_limits_lower[6], self.dof_limits_upper[6])    #* thigh_r_joint
                    self.dof_pos[terrain_groups10,7] = self.default_dof_pos[terrain_groups10,7] #* wheel_r_joint
                if not torch.numel(terrain_groups11) == 0:
                    joint_position_offset = self.get_joint_offset(terrain_groups11, self.dof_limits_lower, self.dof_limits_upper, 128, 6, 6, 128, 6, 6, False)
                    self.dof_pos[terrain_groups11,0] = tensor_clamp(self.default_dof_pos[terrain_groups11,0] + joint_position_offset[:, 0], self.dof_limits_lower[0], self.dof_limits_upper[0])    #* hip_l_joint
                    self.dof_pos[terrain_groups11,1] = tensor_clamp(self.default_dof_pos[terrain_groups11,1] + joint_position_offset[:, 1], self.dof_limits_lower[1], self.dof_limits_upper[1])    #* shoulder_l_joint
                    self.dof_pos[terrain_groups11,2] = tensor_clamp(self.default_dof_pos[terrain_groups11,2] + joint_position_offset[:, 2], self.dof_limits_lower[2], self.dof_limits_upper[2])    #* thigh_l_joint
                    self.dof_pos[terrain_groups11,3] = self.default_dof_pos[terrain_groups11,3] #* wheel_l_joint
                    self.dof_pos[terrain_groups11,4] = tensor_clamp(self.default_dof_pos[terrain_groups11,4] + joint_position_offset[:, 3], self.dof_limits_lower[4], self.dof_limits_upper[4])    #* hip_r_joint
                    self.dof_pos[terrain_groups11,5] = tensor_clamp(self.default_dof_pos[terrain_groups11,5] + joint_position_offset[:, 4], self.dof_limits_lower[5], self.dof_limits_upper[5])    #* shoulder_r_joint
                    self.dof_pos[terrain_groups11,6] = tensor_clamp(self.default_dof_pos[terrain_groups11,6] + joint_position_offset[:, 5], self.dof_limits_lower[6], self.dof_limits_upper[6])    #* thigh_r_joint
                    self.dof_pos[terrain_groups11,7] = self.default_dof_pos[terrain_groups11,7] #* wheel_r_joint
                if not torch.numel(terrain_groups12) == 0:
                    joint_position_offset = self.get_joint_offset(terrain_groups12, self.dof_limits_lower, self.dof_limits_upper, 128, 6, 6, 128, 6, 6, False)
                    self.dof_pos[terrain_groups12,0] = tensor_clamp(self.default_dof_pos[terrain_groups12,0] + joint_position_offset[:, 0], self.dof_limits_lower[0], self.dof_limits_upper[0])    #* hip_l_joint
                    self.dof_pos[terrain_groups12,1] = tensor_clamp(self.default_dof_pos[terrain_groups12,1] + joint_position_offset[:, 1], self.dof_limits_lower[1], self.dof_limits_upper[1])    #* shoulder_l_joint
                    self.dof_pos[terrain_groups12,2] = tensor_clamp(self.default_dof_pos[terrain_groups12,2] + joint_position_offset[:, 2], self.dof_limits_lower[2], self.dof_limits_upper[2])    #* thigh_l_joint
                    self.dof_pos[terrain_groups12,3] = self.default_dof_pos[terrain_groups12,3] #* wheel_l_joint
                    self.dof_pos[terrain_groups12,4] = tensor_clamp(self.default_dof_pos[terrain_groups12,4] + joint_position_offset[:, 3], self.dof_limits_lower[4], self.dof_limits_upper[4])    #* hip_r_joint
                    self.dof_pos[terrain_groups12,5] = tensor_clamp(self.default_dof_pos[terrain_groups12,5] + joint_position_offset[:, 4], self.dof_limits_lower[5], self.dof_limits_upper[5])    #* shoulder_r_joint
                    self.dof_pos[terrain_groups12,6] = tensor_clamp(self.default_dof_pos[terrain_groups12,6] + joint_position_offset[:, 5], self.dof_limits_lower[6], self.dof_limits_upper[6])    #* thigh_r_joint
                    self.dof_pos[terrain_groups12,7] = self.default_dof_pos[terrain_groups12,7] #* wheel_r_joint
                if not torch.numel(terrain_groups13) == 0:
                    joint_position_offset = self.get_joint_offset(terrain_groups13, self.dof_limits_lower, self.dof_limits_upper, 128, 6, 6, 128, 6, 6, False)
                    self.dof_pos[terrain_groups13,0] = tensor_clamp(self.default_dof_pos[terrain_groups13,0] + joint_position_offset[:, 0], self.dof_limits_lower[0], self.dof_limits_upper[0])    #* hip_l_joint
                    self.dof_pos[terrain_groups13,1] = tensor_clamp(self.default_dof_pos[terrain_groups13,1] + joint_position_offset[:, 1], self.dof_limits_lower[1], self.dof_limits_upper[1])    #* shoulder_l_joint
                    self.dof_pos[terrain_groups13,2] = tensor_clamp(self.default_dof_pos[terrain_groups13,2] + joint_position_offset[:, 2], self.dof_limits_lower[2], self.dof_limits_upper[2])    #* thigh_l_joint
                    self.dof_pos[terrain_groups13,3] = self.default_dof_pos[terrain_groups13,3] #* wheel_l_joint
                    self.dof_pos[terrain_groups13,4] = tensor_clamp(self.default_dof_pos[terrain_groups13,4] + joint_position_offset[:, 3], self.dof_limits_lower[4], self.dof_limits_upper[4])    #* hip_r_joint
                    self.dof_pos[terrain_groups13,5] = tensor_clamp(self.default_dof_pos[terrain_groups13,5] + joint_position_offset[:, 4], self.dof_limits_lower[5], self.dof_limits_upper[5])    #* shoulder_r_joint
                    self.dof_pos[terrain_groups13,6] = tensor_clamp(self.default_dof_pos[terrain_groups13,6] + joint_position_offset[:, 5], self.dof_limits_lower[6], self.dof_limits_upper[6])    #* thigh_r_joint
                    self.dof_pos[terrain_groups13,7] = self.default_dof_pos[terrain_groups13,7] #* wheel_r_joint
                if not torch.numel(terrain_groups14) == 0:
                    joint_position_offset = self.get_joint_offset(terrain_groups14, self.dof_limits_lower, self.dof_limits_upper, 128, 6, 6, 128, 6, 6, False)
                    self.dof_pos[terrain_groups14,0] = tensor_clamp(self.default_dof_pos[terrain_groups14,0] + joint_position_offset[:, 0], self.dof_limits_lower[0], self.dof_limits_upper[0])    #* hip_l_joint
                    self.dof_pos[terrain_groups14,1] = tensor_clamp(self.default_dof_pos[terrain_groups14,1] + joint_position_offset[:, 1], self.dof_limits_lower[1], self.dof_limits_upper[1])    #* shoulder_l_joint
                    self.dof_pos[terrain_groups14,2] = tensor_clamp(self.default_dof_pos[terrain_groups14,2] + joint_position_offset[:, 2], self.dof_limits_lower[2], self.dof_limits_upper[2])    #* thigh_l_joint
                    self.dof_pos[terrain_groups14,3] = self.default_dof_pos[terrain_groups14,3] #* wheel_l_joint
                    self.dof_pos[terrain_groups14,4] = tensor_clamp(self.default_dof_pos[terrain_groups14,4] + joint_position_offset[:, 3], self.dof_limits_lower[4], self.dof_limits_upper[4])    #* hip_r_joint
                    self.dof_pos[terrain_groups14,5] = tensor_clamp(self.default_dof_pos[terrain_groups14,5] + joint_position_offset[:, 4], self.dof_limits_lower[5], self.dof_limits_upper[5])    #* shoulder_r_joint
                    self.dof_pos[terrain_groups14,6] = tensor_clamp(self.default_dof_pos[terrain_groups14,6] + joint_position_offset[:, 5], self.dof_limits_lower[6], self.dof_limits_upper[6])    #* thigh_r_joint
                    self.dof_pos[terrain_groups14,7] = self.default_dof_pos[terrain_groups14,7] #* wheel_r_joint
                if not torch.numel(terrain_groups15) == 0:
                    joint_position_offset = self.get_joint_offset(terrain_groups15, self.dof_limits_lower, self.dof_limits_upper, 128, 6, 6, 128, 6, 6, False)
                    self.dof_pos[terrain_groups15,0] = tensor_clamp(self.default_dof_pos[terrain_groups15,0] + joint_position_offset[:, 0], self.dof_limits_lower[0], self.dof_limits_upper[0])    #* hip_l_joint
                    self.dof_pos[terrain_groups15,1] = tensor_clamp(self.default_dof_pos[terrain_groups15,1] + joint_position_offset[:, 1], self.dof_limits_lower[1], self.dof_limits_upper[1])    #* shoulder_l_joint
                    self.dof_pos[terrain_groups15,2] = tensor_clamp(self.default_dof_pos[terrain_groups15,2] + joint_position_offset[:, 2], self.dof_limits_lower[2], self.dof_limits_upper[2])    #* thigh_l_joint
                    self.dof_pos[terrain_groups15,3] = self.default_dof_pos[terrain_groups15,3] #* wheel_l_joint
                    self.dof_pos[terrain_groups15,4] = tensor_clamp(self.default_dof_pos[terrain_groups15,4] + joint_position_offset[:, 3], self.dof_limits_lower[4], self.dof_limits_upper[4])    #* hip_r_joint
                    self.dof_pos[terrain_groups15,5] = tensor_clamp(self.default_dof_pos[terrain_groups15,5] + joint_position_offset[:, 4], self.dof_limits_lower[5], self.dof_limits_upper[5])    #* shoulder_r_joint
                    self.dof_pos[terrain_groups15,6] = tensor_clamp(self.default_dof_pos[terrain_groups15,6] + joint_position_offset[:, 5], self.dof_limits_lower[6], self.dof_limits_upper[6])    #* thigh_r_joint
                    self.dof_pos[terrain_groups15,7] = self.default_dof_pos[terrain_groups15,7] #* wheel_r_joint
                if not torch.numel(terrain_groups16) == 0:
                    joint_position_offset = self.get_joint_offset(terrain_groups16, self.dof_limits_lower, self.dof_limits_upper, 128, 6, 6, 128, 6, 6, False)
                    self.dof_pos[terrain_groups16,0] = tensor_clamp(self.default_dof_pos[terrain_groups16,0] + joint_position_offset[:, 0], self.dof_limits_lower[0], self.dof_limits_upper[0])    #* hip_l_joint
                    self.dof_pos[terrain_groups16,1] = tensor_clamp(self.default_dof_pos[terrain_groups16,1] + joint_position_offset[:, 1], self.dof_limits_lower[1], self.dof_limits_upper[1])    #* shoulder_l_joint
                    self.dof_pos[terrain_groups16,2] = tensor_clamp(self.default_dof_pos[terrain_groups16,2] + joint_position_offset[:, 2], self.dof_limits_lower[2], self.dof_limits_upper[2])    #* thigh_l_joint
                    self.dof_pos[terrain_groups16,3] = self.default_dof_pos[terrain_groups16,3] #* wheel_l_joint
                    self.dof_pos[terrain_groups16,4] = tensor_clamp(self.default_dof_pos[terrain_groups16,4] + joint_position_offset[:, 3], self.dof_limits_lower[4], self.dof_limits_upper[4])    #* hip_r_joint
                    self.dof_pos[terrain_groups16,5] = tensor_clamp(self.default_dof_pos[terrain_groups16,5] + joint_position_offset[:, 4], self.dof_limits_lower[5], self.dof_limits_upper[5])    #* shoulder_r_joint
                    self.dof_pos[terrain_groups16,6] = tensor_clamp(self.default_dof_pos[terrain_groups16,6] + joint_position_offset[:, 5], self.dof_limits_lower[6], self.dof_limits_upper[6])    #* thigh_r_joint
                    self.dof_pos[terrain_groups16,7] = self.default_dof_pos[terrain_groups16,7] #* wheel_r_joint
                if not torch.numel(terrain_groups17) == 0:
                    joint_position_offset = self.get_joint_offset(terrain_groups17, self.dof_limits_lower, self.dof_limits_upper, 128, 6, 6, 128, 6, 6, False)
                    self.dof_pos[terrain_groups17,0] = tensor_clamp(self.default_dof_pos[terrain_groups17,0] + joint_position_offset[:, 0], self.dof_limits_lower[0], self.dof_limits_upper[0])    #* hip_l_joint
                    self.dof_pos[terrain_groups17,1] = tensor_clamp(self.default_dof_pos[terrain_groups17,1] + joint_position_offset[:, 1], self.dof_limits_lower[1], self.dof_limits_upper[1])    #* shoulder_l_joint
                    self.dof_pos[terrain_groups17,2] = tensor_clamp(self.default_dof_pos[terrain_groups17,2] + joint_position_offset[:, 2], self.dof_limits_lower[2], self.dof_limits_upper[2])    #* thigh_l_joint
                    self.dof_pos[terrain_groups17,3] = self.default_dof_pos[terrain_groups17,3] #* wheel_l_joint
                    self.dof_pos[terrain_groups17,4] = tensor_clamp(self.default_dof_pos[terrain_groups17,4] + joint_position_offset[:, 3], self.dof_limits_lower[4], self.dof_limits_upper[4])    #* hip_r_joint
                    self.dof_pos[terrain_groups17,5] = tensor_clamp(self.default_dof_pos[terrain_groups17,5] + joint_position_offset[:, 4], self.dof_limits_lower[5], self.dof_limits_upper[5])    #* shoulder_r_joint
                    self.dof_pos[terrain_groups17,6] = tensor_clamp(self.default_dof_pos[terrain_groups17,6] + joint_position_offset[:, 5], self.dof_limits_lower[6], self.dof_limits_upper[6])    #* thigh_r_joint
                    self.dof_pos[terrain_groups17,7] = self.default_dof_pos[terrain_groups17,7] #* wheel_r_joint
                if not torch.numel(terrain_groups18) == 0:
                    joint_position_offset = self.get_joint_offset(terrain_groups18, self.dof_limits_lower, self.dof_limits_upper, 128, 6, 6, 128, 6, 6, False)
                    self.dof_pos[terrain_groups18,0] = tensor_clamp(self.default_dof_pos[terrain_groups18,0] + joint_position_offset[:, 0], self.dof_limits_lower[0], self.dof_limits_upper[0])    #* hip_l_joint
                    self.dof_pos[terrain_groups18,1] = tensor_clamp(self.default_dof_pos[terrain_groups18,1] + joint_position_offset[:, 1], self.dof_limits_lower[1], self.dof_limits_upper[1])    #* shoulder_l_joint
                    self.dof_pos[terrain_groups18,2] = tensor_clamp(self.default_dof_pos[terrain_groups18,2] + joint_position_offset[:, 2], self.dof_limits_lower[2], self.dof_limits_upper[2])    #* thigh_l_joint
                    self.dof_pos[terrain_groups18,3] = self.default_dof_pos[terrain_groups18,3] #* wheel_l_joint
                    self.dof_pos[terrain_groups18,4] = tensor_clamp(self.default_dof_pos[terrain_groups18,4] + joint_position_offset[:, 3], self.dof_limits_lower[4], self.dof_limits_upper[4])    #* hip_r_joint
                    self.dof_pos[terrain_groups18,5] = tensor_clamp(self.default_dof_pos[terrain_groups18,5] + joint_position_offset[:, 4], self.dof_limits_lower[5], self.dof_limits_upper[5])    #* shoulder_r_joint
                    self.dof_pos[terrain_groups18,6] = tensor_clamp(self.default_dof_pos[terrain_groups18,6] + joint_position_offset[:, 5], self.dof_limits_lower[6], self.dof_limits_upper[6])    #* thigh_r_joint
                    self.dof_pos[terrain_groups18,7] = self.default_dof_pos[terrain_groups18,7] #* wheel_r_joint
                if not torch.numel(terrain_groups19) == 0:
                    joint_position_offset = self.get_joint_offset(terrain_groups19, self.dof_limits_lower, self.dof_limits_upper, 128, 6, 6, 128, 6, 6, False)
                    self.dof_pos[terrain_groups19,0] = tensor_clamp(self.default_dof_pos[terrain_groups19,0] + joint_position_offset[:, 0], self.dof_limits_lower[0], self.dof_limits_upper[0])    #* hip_l_joint
                    self.dof_pos[terrain_groups19,1] = tensor_clamp(self.default_dof_pos[terrain_groups19,1] + joint_position_offset[:, 1], self.dof_limits_lower[1], self.dof_limits_upper[1])    #* shoulder_l_joint
                    self.dof_pos[terrain_groups19,2] = tensor_clamp(self.default_dof_pos[terrain_groups19,2] + joint_position_offset[:, 2], self.dof_limits_lower[2], self.dof_limits_upper[2])    #* thigh_l_joint
                    self.dof_pos[terrain_groups19,3] = self.default_dof_pos[terrain_groups19,3] #* wheel_l_joint
                    self.dof_pos[terrain_groups19,4] = tensor_clamp(self.default_dof_pos[terrain_groups19,4] + joint_position_offset[:, 3], self.dof_limits_lower[4], self.dof_limits_upper[4])    #* hip_r_joint
                    self.dof_pos[terrain_groups19,5] = tensor_clamp(self.default_dof_pos[terrain_groups19,5] + joint_position_offset[:, 4], self.dof_limits_lower[5], self.dof_limits_upper[5])    #* shoulder_r_joint
                    self.dof_pos[terrain_groups19,6] = tensor_clamp(self.default_dof_pos[terrain_groups19,6] + joint_position_offset[:, 5], self.dof_limits_lower[6], self.dof_limits_upper[6])    #* thigh_r_joint
                    self.dof_pos[terrain_groups19,7] = self.default_dof_pos[terrain_groups19,7] #* wheel_r_joint

                #* Calculate base_link height(over Z-axis) based on dof_pos and env_ids when reset -JH
                base_height_tensor = self.get_base_height(self.dof_pos, env_ids)
                self.root_states[env_ids] = self.base_init_state
                self.root_states[env_ids, 2] = base_height_tensor
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
        #* Update random velocity on joints -JH
        self.dof_vel[env_ids] = velocities

        #* Resetting to initial root states with regard to roll pitch yaw -JH
        #self.root_states[env_ids] = self.initial_root_states[env_ids]
        # Introduce small perturbations in roll, pitch, and yaw (comment these lines if you want no randomness)
        # roll = torch_rand_float(-0.005, 0.005, (num_resets, 1), self.device).flatten()
        # pitch = torch_rand_float(-0.005, 0.005, (num_resets, 1), self.device).flatten()
        # yaw = torch_rand_float(-0.005, 0.005, (num_resets, 1), self.device).flatten()
        #self.root_states[env_ids, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
        #* Resetting to initial root states with regard to roll pitch yaw -JH

        # Update the simulation environment with the new initial states
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        #* Send random velocity on reset -JH
        if self.sendRandomVelocity == True:
            gaussian_commands = self.get_gaussian_velocity(env_ids, self.command_x_range[0], self.command_x_range[1], self.command_y_range[0], self.command_y_range[1],
                                                           self.command_yaw_range[0], self.command_yaw_range[1], 3, 3, 3)
            
            gaussian_commands_x = torch.clamp(gaussian_commands[:, 0], self.command_x_range[0], self.command_x_range[1])
            gaussian_commands_y = torch.clamp(gaussian_commands[:, 1], self.command_y_range[0], self.command_y_range[1])
            gaussian_commands_yaw = torch.clamp(gaussian_commands[:, 2], self.command_yaw_range[0], self.command_yaw_range[1])

            self.commands[env_ids, 0] = gaussian_commands_x
            self.commands[env_ids, 1] = gaussian_commands_y
            self.commands[env_ids, 2] = gaussian_commands_yaw
            self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.5).unsqueeze(1)

            #self.commands[env_ids, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
            #self.commands[env_ids, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
            #self.commands[env_ids, 2] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
            #self.commands[env_ids, 3] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
            #self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.1).unsqueeze(1)
        else:
            self.commands[env_ids, 0] = torch_rand_float(0.0, 0.0, (len(env_ids), 1), device=self.device).squeeze()
            self.commands[env_ids, 1] = torch_rand_float(0.0, 0.0, (len(env_ids), 1), device=self.device).squeeze()
            self.commands[env_ids, 2] = torch_rand_float(0.0, 0.0, (len(env_ids), 1), device=self.device).squeeze()
            #self.commands[env_ids, 3] = torch_rand_float(0.0, 0.0, (len(env_ids), 1), device=self.device).squeeze()
            self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.1).unsqueeze(1)

        #* Resetting buffers -JH
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_obs_buf[env_ids] = 0.
        self.last_last_obs_buf[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids])
            self.episode_sums[key][env_ids] = 0.
        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

    def push_robots(self):
        # Only apply forces in the x-direction (forward/backward)
        #self.root_states[:, 7] = torch_rand_float(-1., 1., (self.num_envs, 1), device=self.device)
        # Set the y-direction velocity to 0
        #self.root_states[:, 8] = torch.zeros(self.num_envs, device=self.device)
        self.root_states[:, 7:9] = torch_rand_float(-0.5, 0.5, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def update_terrain_level(self, env_ids):
        if not self.init_done or not self.curriculum:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        self.terrain_levels[env_ids] -= 1 * (distance < torch.norm(self.commands[env_ids, :2])*self.max_episode_length_s)
        self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        #* Generate robot on terrain sequentially -JH
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]] #* env id 에 할당된 terrain_type tensor 에 맵핑됨 -JH

    def pre_physics_step(self, actions):
        #* Convert to the device and scale by max effort -JH
        if self.test_mode:
            self.actions = actions.clone().to(self.device)
        else:
            delay = torch.rand((self.num_envs, 1), device=self.device) # 0~1 [
            self.actions = (1 - delay) * actions + delay * self.actions # a <- previous_a + a
            #self.actions += 0.02 * torch.randn_like(self.actions) * self.actions # a <- a + N
            
        #print("self actions: ", self.actions)
        # scale the wheel effort
        # self.actions[:,5] *= 0.1
        # self.actions[:,9] *= 0.1

        # for i in range(10):
            # self.actions[:, i] = 0
        # self.actions[:, 2] = 0.0
        # self.actions[:, 6] = -0.0
        # self.actions[:, 3] = -0.5
        # self.actions[:, 7] = -0.5
        # self.actions[:, 4] = 5.0
        # self.actions[:, 8] = 5.0
        # self.actions[:, 5] = 0.1
        # self.actions[:, 9] = 0.1
        # print("shoulder_left", self.dof_pos[0, 2])
        # print("shoulder_right", self.dof_pos[0, 6])
        # print('---------------------------------------------')
        # print(self.actions[0].shape)
        # print("action:", self.actions[0])
        # print('---------------------------------------------') # action test!!!!

        for i in range(self.decimation):
            if not self.usePd:
                torques = self.actions * self.scale_effort_joints
                torques[:, 3] = self.actions[:, 3] * self.scale_effort_wheels
                torques[:, 7] = self.actions[:, 7] * self.scale_effort_wheels
            else:
                torques = torch.clip(self.Kp*(self.action_scale*self.actions + self.default_dof_pos - self.dof_pos) - self.Kd*self.dof_vel,
                         -23., 23.)
                torques[:, 3] = self.actions[:, 3] * self.scale_effort_wheels # for Left wheel, not adjust PD control
                torques[:, 7] = self.actions[:, 7] * self.scale_effort_wheels # for Right wheel, not adjust PD control
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.torques = torques.view(self.torques.shape)
            #print("self.actions[0]: ", self.actions[0])
            #print("torque: ", self.torques[0])
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        
    #* Basic physics step implementation -JH
    # def pre_physics_step(self, actions):
    #     actions = actions.to(self.device)
    #     torques = gymtorch.unwrap_tensor(actions * self.scale_effort)
    #     self.gym.set_dof_actuation_force_tensor(self.sim, torques)
    #     self.gym.refresh_dof_state_tensor(self.sim)

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
       
        self.progress_buf += 1
        self.randomize_buf += 1
        self.common_step_counter += 1
        
        if self.common_step_counter % self.push_interval == 0:
            if self.cfg["env"]["learn"]["pushRobots"] == True:
                self.push_robots()

        # prepare quantities
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])

        # if self.sendRandomVelocity == True:
            # self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 2] - heading), -1., 1.)
            #self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
        # else:
            # self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 2] - heading), 0., 0.)
            #self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), 0., 0.)

        #* Dispaly reward vis window if flag's True -JH
        if self.reward_vis:
            episode_done = self.check_episode_done(self.reset_buf, self.progress_buf, self.max_episode_length)
            if episode_done:
                self.log_episode_sums()
                self.reset_episode_sums(self.torch_zeros)
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        #* Update Previous actions and joint velocities -JH
        self.last_last_obs_buf[:] = self.last_obs_buf[:]
        self.last_obs_buf[:] = self.obs_buf[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
  
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

    def compute_reward(self, actions): #  새로 업데이트 했으예~
        # Retrieve environment observations from buffer
        # For joint positions
        hip_pos = self.obs_buf[:, [0, 4]]  # left_hip_link, right_hip_link
        shoulder_pos = self.obs_buf[:, [1, 5]]  # left_shoulder_link, right_shoulder_link
        thigh_pos = self.obs_buf[:, [2, 6]]  # left_thigh_link, right_thigh_link
        wheel_pos = self.obs_buf[:, [3, 7]]  # left_wheel_link, right_wheel_link

        # For joint velocities
        hip_vel = self.obs_buf[:, [8, 12]]  # left_hip_link, right_hip_link
        shoulder_vel = self.obs_buf[:, [9, 13]]  # left_shoulder_link, right_shoulder_link
        thigh_vel = self.obs_buf[:, [10, 14]]  # left_thigh_link, right_thigh_link
        wheel_vel = self.obs_buf[:, [11, 15]]  # left_wheel_link, right_wheel_link

        # For body states
        body_pos = self.privileged_obs_buf[:, 0:3]
        body_orientation = self.obs_buf[:, 16:19]
        body_linear_vel = self.obs_buf[:, 19:22]
        body_angular_vel = self.obs_buf[:, 22:25]
        robot_command = self.obs_buf[:, 99:102]

        base_link_height = body_pos[:, 2]

        if not self.allow_legs_contacts:
            base_contact = torch.norm(self.contact_forces[:, self.base_indices, :], dim=1) > 1.
            legs_contact = torch.norm(self.contact_forces[:, self.thighs_indices, :], dim=2) > 1.
            shoulders_contact = torch.norm(self.contact_forces[:, self.shoulders_indices, :], dim=2) > 1.
            # caster_contact = torch.norm(self.contact_forces[:, self.caster_indices, :], dim=1 ) > 1.
            #print(f"base_contact for env_id 1:", base_contact[1])
            #print(f"legs_contact for env_id 1:", legs_contact[1])
            #print(f"legs_contact for env_id 1:", wheel_contact[1])

        #* Compute reward and reset conditions
        self.rew_buf[:], self.reset_buf[:] = self.compute_postech_reward(
            hip_pos, shoulder_pos, thigh_pos, wheel_pos,
            hip_vel, shoulder_vel, thigh_vel, wheel_vel,
            body_pos, body_orientation, body_linear_vel, body_angular_vel,
            robot_command, self.reset_buf, self.progress_buf, self.max_episode_length, base_link_height,
            legs_contact, base_contact, shoulders_contact, self.projected_gravity, self.env_origins, self.torques,
            self.last_dof_vel, self.dof_vel, self.last_actions, self.actions
        )
        if self.always_positive_reward:
            self.rew_buf = torch.clip(self.rew_buf, min=0., max=None)

    def compute_postech_reward(self, hip_pos, shoulder_pos, thigh_pos, wheel_pos,
                                     hip_vel, shoulder_vel, thigh_vel, wheel_vel,
                                     body_pos, body_orientation, body_linear_vel, body_angular_vel,
                                     commands, reset_buf, progress_buf, max_episode_length, base_link_height,
                                     legs_contact, base_contact, shoulders_contact, 
                                     projected_gravity, env_origins, torques,
                                     last_dof_vel, dof_vel, last_actions, actions):
        
        reward_alive_time = self.reward_scale["alive_time"] * progress_buf

        #epsilon = 1e-6
        #reward_orientation = self.reward_scale["orientation"] * torch.exp(-torch.abs(body_orientation[:, 0]), + epsilon) * torch.exp(-torch.abs(body_orientation[:, 1]) + epsilon)
        #reward_orientation = self.reward_scale["orientation"] * (np.pi - torch.abs(body_orientation[:, 1])) * (np.pi - torch.abs(body_orientation[:, 0]))
        reward_orientation =  self.reward_scale["orientation"] * (torch.sum(torch.square(projected_gravity[:, :2]), dim=1))

        # velocity tracking reward
        lin_vel_error = torch.sum(torch.square(commands[:, :2] - body_linear_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(commands[:, 2] - body_angular_vel[:, 2])
        reward_lin_vel_xy = self.reward_scale["lin_vel_xy"] * torch.exp(-lin_vel_error)
        reward_ang_vel_z = self.reward_scale["ang_vel_z"] * torch.exp(-ang_vel_error)
        
        # New Reward Term for Minimizing Rate of Change in Roll and Pitch
        reward_lin_vel_z = self.reward_scale["lin_vel_z"] * torch.square(body_linear_vel[:, 2])
        reward_ang_vel_xy = self.reward_scale["ang_vel_xy"] * torch.sum(torch.square(body_angular_vel[:, :2]), dim=1)    

        # wheel velocity reward over body velocity
        reward_wheel_lin_x = self.reward_scale["wheel_vel_x"] * torch.exp(-torch.square(body_linear_vel[:, 0] - ((-wheel_vel[:, 1] + wheel_vel[:, 0]) / 2 )))
        reward_wheel_ang_z = self.reward_scale["wheel_vel_z"] * torch.exp(-torch.square(body_angular_vel[:, 2] - ((-wheel_vel[:, 1] - wheel_vel[:, 0]) / 0.24862))) # 2 * 0.12431

        condition_mask_lin_x = (body_linear_vel[:, 0] < 0.15).float()
        condition_mask_ang_z = (body_angular_vel[:, 2] < 0.15).float()
        reward_wheel_lin_x *= 1-condition_mask_lin_x
        reward_wheel_ang_z *= 1-condition_mask_ang_z

        # orientation penalty
        rew_gravity = self.reward_scale["gravity"] * torch.sum(torch.square(projected_gravity[:, :2]), dim=1)

        # Base Contact Penalty
        reward_base_contact = self.reward_scale["contact"] * torch.sum(base_contact, dim=1)
        
        # Leg Contact Penalty
        reward_legs_contact = self.reward_scale["contact"] * torch.sum(legs_contact, dim=1)

        # reward for stationary position
        reward_position = self.reward_scale["position"] * (torch.norm(env_origins[:, :2] - body_pos[:, :2], dim=1))
        
        # reward for heading angle
        reward_heading = self.reward_scale["heading"] * (torch.abs(body_orientation[:, 2] - 0))
 
        # Height reward component
        reward_height = self.reward_scale["height"] * (torch.abs(base_link_height - (self.approx_measured_heights + self.noisy_target_height)))
        # Apply a sharper gradient to the height reward
        #reward_height = self.reward_scale["height"] * torch.exp(-torch.square(base_link_height - self.reward_scale["target_height"]))

        # reward_hip_alignment = k_hip_align * ((torch.exp(-torch.abs(hip_pos[:, 0] - 0)) + torch.exp(-torch.abs(hip_pos[:, 1] - 0))) / 2)
        # reward_hip_alignment = self.reward_scale["hip_align"] * ((torch.exp(torch.abs(hip_pos[:, 0] - 0)) + torch.exp(torch.abs(hip_pos[:, 1] - 0))) / 2) #! 너무크게 패널티 줘도 괜찮을까..
        reward_hip_alignment = self.reward_scale["hip_align"] * (((torch.abs(hip_pos[:, 0] - 0)) + (torch.abs(hip_pos[:, 1] - 0))) / 2) #! exponential 빼고 해보자
        
        # left :  lower="-0.383972"  upper="0.7854"
        # right:  lower="-0.7854"    upper="0.383972"
        reward_hip_des = self.reward_scale["des_hip"] * (torch.abs(hip_pos[:, 0] * hip_pos[:, 1]))

        #reward_shoulder_alignment = k_shoulder_align * (torch.abs(shoulder_pos[:, 0]) - torch.abs(shoulder_pos[:, 1])) # seems wrong...
        # reward_shoulder_alignment = k_shoulder_align * (0.25 - torch.abs((shoulder_pos[:, 0]) + shoulder_pos[:, 1]))  # pos is the opposite value. Mirroed setup!!
        # reward_leg_alignment = k_leg_align * (0.25 - torch.abs((thigh_pos[:, 0]) + thigh_pos[:, 1]))

        # reward for shoulder alignment
        reward_shoulder_alignment = self.reward_scale["shoulder_align"] * (torch.abs(shoulder_pos[:, 0] - shoulder_pos[:, 1]))

        # reward for leg alignment
        reward_leg_alignment = self.reward_scale["leg_align"] * torch.abs(thigh_pos[:, 0] - thigh_pos[:, 1])
 
        reward_torque = self.reward_scale["torque"] * torch.sum(torch.square(torques), dim=1)

        # joint acc penalty
        reward_joint_acc =  self.reward_scale["joint_acc"] * torch.sum(torch.square(last_dof_vel - dof_vel), dim=1)

        # action rate penalty
        reward_action_rate = self.reward_scale["act_rate"] * torch.sum(torch.square(last_actions - actions), dim=1)

        # reward for shoulder angle(pos)
        reward_shoulder_pos = self.reward_scale["shoulder_pos"] * torch.exp(torch.sum(torch.abs(shoulder_pos[:, 0:2] + 0.55), dim=1)) #* 0.7766715
        
        # rewward for leg angle(pos)
        reward_leg_pos = self.reward_scale["leg_pos"] * torch.exp(torch.sum(torch.abs(thigh_pos[:, 0:2] - 0.698132), dim=1)) #* 0.698132

                # Total reward
        reward = reward_orientation + reward_lin_vel_xy + reward_ang_vel_z + reward_ang_vel_xy + reward_lin_vel_z + reward_wheel_lin_x + reward_wheel_ang_z + rew_gravity + \
                 reward_base_contact + reward_legs_contact + reward_height + reward_hip_alignment + reward_hip_des +  \
                 reward_shoulder_alignment + reward_leg_alignment + reward_position + reward_heading + reward_torque + reward_joint_acc + reward_action_rate + reward_alive_time + \
                 reward_shoulder_pos + reward_leg_pos

        # orientation_beta_ = 0.0001
        # height_beta_ = -5.
        # hip_beta_ = 0.11
        # shoulder_beta_ = -0.35 #!0.001
        # leg_beta_ = -0.35 #! 0.001
        # action_beta_ = 0.01
        # torque_beta_ = 0.003
        # epsilon_scaler_ = 0.01

        # reward_orientation_ = orientation_beta_ * torch.exp(-torch.sum(torch.square(projected_gravity[:, :2]), dim=1))
        # reward_height_ = height_beta_ * torch.square(base_link_height - self.reward_scale["target_height"] )
        # reward_hip_ = hip_beta_ * torch.exp(-torch.mean(torch.abs(hip_pos[:, 0:2] - 0), dim=1))
        # reward_shoulder_ = shoulder_beta_ * torch.exp(torch.mean(torch.abs(shoulder_pos[:, 0:2] + 0.7766715), dim=1))
        # reward_leg_ = leg_beta_ * torch.exp(torch.mean(torch.abs(thigh_pos[:, 0:2] - 0.698132), dim=1))
        # reward_action_ = action_beta_ * torch.exp(-torch.sum(torch.square(last_actions - actions), dim=1))
        # reward_torque_ = torque_beta_ * torch.exp(-torch.sum(torch.abs(torques) * epsilon_scaler_, dim=1))
        
#################### JW reward ##########################################
        # orientation_beta_ = 0.5
        # height_beta_ = -5.
        # hip_beta_ = 1.
        # shoulder_beta_ = 1.
        # leg_beta_ = 1.
        # action_beta_ = 1.
        # torque_beta_ = 1.
        # epsilon_scaler_ = 0.01

        # reward_orientation_ = orientation_beta_ * torch.exp(-torch.sum(torch.square(projected_gravity[:, :2]), dim=1))
        # reward_height_ = height_beta_ * torch.square(base_link_height - self.reward_scale["target_height"] )
        # reward_hip_ = torch.exp(hip_beta_ * -torch.mean(torch.abs(hip_pos[:, 0:2] - 0), dim=1))
        # reward_shoulder_ = torch.exp(shoulder_beta_ * -torch.mean(torch.abs(shoulder_pos[:, 0:2] + 0.7766715), dim=1))
        # reward_leg_ = torch.exp(leg_beta_ * -torch.mean(torch.abs(thigh_pos[:, 0:2] - 0.698132), dim=1))
        # reward_action_ = torch.exp(action_beta_ * -torch.sum(torch.square(last_actions - actions) * epsilon_scaler_, dim=1))
        # reward_torque_ = torch.exp(torque_beta_ * -torch.sum(torch.abs(torques) * epsilon_scaler_, dim=1))

        # reward_jw_ = reward_height_ + reward_orientation_ + ((2 - reward_hip_) * 0.5) * ((2 - reward_shoulder_) * 0.5) * ((2 - reward_leg_) * 0.5) * ((2 - reward_action_) * 0.5) * ((2 - reward_torque_) * 0.5)

        # print("reward_orientation: ", reward_orientation_[0])
        # print("reward_height: ", reward_height_[0])
        # print("reward_hip: ", reward_hip_[0])
        # print("reward_shoulder: ", reward_shoulder_[0])
        # print("reward_leg: ", reward_leg_[0])
        # print("reward_action: ", reward_action_[0])
        # print("reward_torque: ", reward_torque_[0])
        # print("reward_total: ", reward_jw_[0])
        # print("--------------------------------")

        # print("reward_orientation: ", reward_orientation_[0])
        # print("reward_height: ", reward_height_[0])
        # print("reward_hip: ", ((1- reward_hip_[0]) * 0.5))
        # print("reward_shoulder: ", ((1 - reward_shoulder_[0]) * 0.5))
        # print("reward_leg: ", ((1 - reward_leg_[0]) * 0.5))
        # print("reward_action: ", ((1 - reward_action_[0]) * 0.5))
        # print("reward_torque: ", (1 - reward_torque_[0]) * 0.5)
        # print("reward_total: ", reward_jw_[0])
        # print("--------------------------------")
##############################################################################
        #reset_condition = torch.any(legs_contact, dim=1) | torch.any(base_contact, dim=1) | torch.any(shoulders_contact, dim=1) | (base_link_height < 0.05)

        #* Define the condition for any other contacts and the height condition -JH
        #collide_condition = torch.any(legs_contact, dim=1) | torch.any(base_contact, dim=1) | torch.any(shoulders_contact, dim=1)
        collide_condition = torch.any(base_contact, dim=1) | torch.any(legs_contact, dim=1)
        #flipped_condition = (torch.abs(body_orientation[:, 0]) > 3.14) | (torch.abs(body_orientation[:, 1]) > 3.14)
        edge_reset_condition = torch.full((body_pos.shape[0],), False, dtype=torch.bool, device=self.device)
        # print("tot row: ", self.terrain.tot_rows)
        # print("tot_col: ", self.terrain.tot_cols)
        # raise

        if self.cfg["env"]["terrain"]["terrainType"]  in ["postech_terrain", "trimesh"]:     
            edge_reset_condition = torch.where(body_pos[:, 1] > 165., True, False) #columns
            edge_reset_condition |= torch.where(body_pos[:, 1] < 0., True, False) #columns
            edge_reset_condition |= torch.where(body_pos[:, 0] > 85., True, False) #rows
            edge_reset_condition |= torch.where(body_pos[:, 0] < 0., True, False) #rows

        #* Modify reset_condition to be true only if caster_contact is true and no other conditions are met -JH
        complement_condition = collide_condition | edge_reset_condition

        reset = torch.where(complement_condition, torch.ones_like(reset_buf), reset_buf)
        reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

        #* Update episode reward sums -JH
        self.episode_sums["alive_time"] += reward_alive_time
        self.episode_sums["orientation"] += reward_orientation
        self.episode_sums["lin_vel_xy"] += reward_lin_vel_xy
        self.episode_sums["ang_vel_z"] += reward_ang_vel_z
        self.episode_sums["wheel_vel_x"] += reward_wheel_lin_x
        self.episode_sums["wheel_vel_z"] += reward_wheel_ang_z
        self.episode_sums["lin_vel_z"] += reward_lin_vel_z
        self.episode_sums["ang_vel_xy"] += reward_ang_vel_xy
        self.episode_sums["gravity"] += rew_gravity
        self.episode_sums["base_contact"] += reward_base_contact
        self.episode_sums["legs_contact"] += reward_legs_contact
        self.episode_sums["height"] += reward_height
        self.episode_sums["hip_alignment"] += reward_hip_alignment
        self.episode_sums["hip_desired"] += reward_hip_des
        self.episode_sums["shoulder_alignment"] += reward_shoulder_alignment
        self.episode_sums["leg_alignment"] += reward_leg_alignment
        self.episode_sums["position"] += reward_position
        self.episode_sums["heading"] += reward_heading
        self.episode_sums["torque"] += reward_torque
        self.episode_sums["joint_acc"] += reward_joint_acc
        self.episode_sums["action_rate"] += reward_action_rate
        self.episode_sums["shoulder_pos"] += reward_shoulder_pos
        self.episode_sums["leg_pos"] += reward_leg_pos

        # self.episode_sums["reward_orientation_"] += reward_orientation_
        # self.episode_sums["reward_hip_"] += reward_hip_
        # self.episode_sums["reward_shoulder_"] += reward_shoulder_
        # self.episode_sums["reward_leg_"] += reward_leg_
        # self.episode_sums["reward_action_"] += reward_action_
        # self.episode_sums["reward_torque_"] += reward_torque_

        return reward, reset
    
    def reset_episode_sums(self, torch_zeros_):
        #* Initializes or resets the episode sums to zero at the start of each episode -JH
        self.episode_sums = {
            "alive_time": torch_zeros_(),
            "orientation": torch_zeros_(),
            "lin_vel_xy": torch_zeros_(),
            "ang_vel_z": torch_zeros_(),
            "wheel_vel_x": torch_zeros_(),
            "wheel_vel_z": torch_zeros_(),
            "lin_vel_z": torch_zeros_(),
            "ang_vel_xy": torch_zeros_(),
            "gravity": torch_zeros_(),
            "base_contact": torch_zeros_(),
            "legs_contact": torch_zeros_(),
            "height": torch_zeros_(),
            "hip_alignment": torch_zeros_(),
            "hip_desired": torch_zeros_(),
            "shoulder_alignment": torch_zeros_(),
            "leg_alignment": torch_zeros_(),
            "position": torch_zeros_(),
            "heading": torch_zeros_(),
            "torque": torch_zeros_(),
            "joint_acc": torch_zeros_(),
            "action_rate": torch_zeros_(),
            "shoulder_pos": torch_zeros_(),
            "leg_pos": torch_zeros_(),
            # "reward_orientation_": torch_zeros_(),
            # "reward_hip_": torch_zeros_(),
            # "reward_shoulder_": torch_zeros_(),
            # "reward_leg_": torch_zeros_(),
            # "reward_action_" : torch_zeros_(),
            # "reward_torque_" : torch_zeros_(),
        }
        
    def log_episode_sums(self):
        # Create a blank image where text will be added
        img_height = 1350
        img_width = 450
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)  # White color in BGR
        title_color = (0,255,0)
        thickness = 1
        line_type = cv2.LINE_AA

        total_scalar_sum = 0

        # Starting Y position
        y_position = 25

        # Display Episode Reward Sums
        cv2.putText(img, "Episode Reward Sums:", (10, y_position), font, font_scale, title_color, thickness, line_type)
        y_position += 25  # Increment y position for next line

        for key, value in self.episode_sums.items():
            text = f"{key}: {value.mean().item()}"
            cv2.putText(img, text, (10, y_position), font, font_scale, color, thickness, line_type)
            y_position += 25  # Increment y position for next line

        # Correctly accumulating total rewards
        for key in self.episode_sums.keys():
            self.total_reward_sums[key] += self.episode_sums[key].mean().item()
            total_scalar_sum += self.episode_sums[key].mean().item()
        
        y_position += 25

        cv2.putText(img, "Episode Reward Sum:", (10, y_position), font, font_scale, title_color, thickness, line_type)
        y_position += 25
        cv2.putText(img, str(total_scalar_sum), (10, y_position), font, font_scale, color, thickness, line_type)

        y_position += 50

        # Moving the print statement outside the loop to print the total once fully updated
        cv2.putText(img, "Total Reward Sums:", (10, y_position), font, font_scale, title_color, thickness, line_type)
        y_position += 25  # Increment y position for next line

        for key, value in self.total_reward_sums.items():
            text = f"{key}: {value}"
            cv2.putText(img, text, (10, y_position), font, font_scale, color, thickness, line_type)
            y_position += 25  # Increment y position for next line

        # Display the image
        cv2.imshow("Reward Sums", img)
        cv2.waitKey(1) 
        #cv2.destroyAllWindows()

    def check_episode_done(self, reset_buf, progress_buf, max_episode_length):
        # Example condition: episode length exceeded or specific reset condition met
        if torch.any(reset_buf) or torch.any(progress_buf >= max_episode_length - 1):
            return True
        return False
            
    def get_base_height(self, dof_pos, env_ids):
        epsilon = 1e-03
        radius = 0.053
        num_envs = len(env_ids)
        base_heights = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

        # Precompute repeated constants
        sin_piover2 = torch.sin(torch.tensor(1.5707963267948966))
        cos_piover2 = torch.cos(torch.tensor(1.5707963267948966))

        sin_pi = torch.sin(torch.tensor(3.14159265358979324))
        cos_pi = torch.cos(torch.tensor(3.14159265358979324))

        # Batched translation matrices
        def T_x(l):
            result = torch.eye(4, 4, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(num_envs, 1, 1)
            result[:, 0, 3] = l
            return result

        def T_y(l):
            result = torch.eye(4, 4, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(num_envs, 1, 1)
            result[:, 1, 3] = l
            return result

        def T_z(l):
            result = torch.eye(4, 4, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(num_envs, 1, 1)
            result[:, 2, 3] = l
            return result

        # Batched rotation matrices around Z-axis
        def R_z(r):
            result = torch.eye(4, 4, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(num_envs, 1, 1)
            result[:, 0, 0] = torch.cos(r)
            result[:, 0, 1] = -torch.sin(r)
            result[:, 1, 0] = torch.sin(r)
            result[:, 1, 1] = torch.cos(r)
            return result

        # Using the same rotation matrices for all envs, as they are constant in this case
        R_x_pi = torch.tensor([[1, 0, 0, 0],
                                    [0, cos_pi, -sin_pi, 0],
                                    [0, sin_pi, cos_pi, 0],
                                    [0, 0, 0, 1]], dtype=torch.float32, device=self.device)

        R_x_piover2 = torch.tensor([[1, 0, 0, 0],
                                    [0, cos_piover2, -sin_piover2, 0],
                                    [0, sin_piover2, cos_piover2, 0],
                                    [0, 0, 0, 1]], dtype=torch.float32, device=self.device)

        R_y_piover2 = torch.tensor([[cos_piover2, 0, sin_piover2, 0],
                                    [0, 1, 0, 0],
                                    [-sin_piover2, 0, cos_piover2, 0],
                                    [0, 0, 0, 1]], dtype=torch.float32, device=self.device)
        
        R_y_mpiover2 = torch.tensor([[-cos_piover2, 0, -sin_piover2, 0],
                                    [0, 1, 0, 0],
                                    [sin_piover2, 0, -cos_piover2, 0],
                                    [0, 0, 0, 1]], dtype=torch.float32, device=self.device)
        
        R_z_piover2 = torch.tensor([[cos_piover2, -sin_piover2, 0, 0],
                                    [sin_piover2, cos_piover2, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=torch.float32, device=self.device)

        left_transformations = (T_x(-0.0545) @ T_y(0.08) @ T_z(-0.0803) @ R_z_piover2 @ R_x_piover2 @
                                R_z(dof_pos[env_ids, 0]) @ T_x(0.0265) @ T_z(0.06) @ R_y_piover2 @
                                R_z(dof_pos[env_ids, 1]) @ T_x(0.17097) @ T_y(-0.13845) @ T_z(0.07005) @
                                R_z(dof_pos[env_ids, 2]) @ T_x(-0.21779) @ T_y(0.0311) @ T_z(0.03988))

        right_transformations = (T_x(-0.0545) @ T_y(-0.08) @ T_z(-0.0803) @ R_z_piover2 @ R_x_piover2 @
                                 R_z(dof_pos[env_ids, 4]) @ T_x(-0.0265) @ T_z(0.06) @ R_x_pi @ R_y_mpiover2 @
                                 R_z(-dof_pos[env_ids, 5]) @ T_x(0.17097) @ T_y(0.13845) @ T_z(0.07005) @
                                 R_z(-dof_pos[env_ids, 6]) @ T_x(-0.21779) @ T_y(-0.0311) @ T_z(0.03988))
    
        left_joint_z_pos = left_transformations[:, 2, 3]
        right_joint_z_pos = right_transformations[:, 2, 3]

        base_heights = -(((left_joint_z_pos + right_joint_z_pos) / 2) + epsilon)

        base_heights += radius

        return base_heights

    def get_joint_offset(self, env_ids, dof_limits_lower, dof_limits_upper, l_h_dev, l_s_dev, l_l_dev, r_h_dev, r_s_dev, r_l_dev, all_random):
        #* TWBR Joint limits -JH
        left_hip_joint_lower_limit = dof_limits_lower[0].item()
        left_hip_joint_upper_limit = dof_limits_upper[0].item()
        left_shoulder_joint_lower_limit = dof_limits_lower[1].item()
        left_shoulder_joint_upper_limit = dof_limits_upper[1].item()
        left_leg_joint_lower_limit = dof_limits_lower[2].item()
        left_leg_joint_upper_limit = dof_limits_upper[2].item()
    
        right_hip_joint_lower_limit = dof_limits_lower[4].item()
        right_hip_joint_upper_limit = dof_limits_upper[4].item()
        right_shoulder_joint_lower_limit = dof_limits_lower[5].item()
        right_shoulder_joint_upper_limit = dof_limits_upper[5].item()
        right_leg_joint_lower_limit = dof_limits_lower[6].item()
        right_leg_joint_upper_limit = dof_limits_upper[6].item()
    
        #* TWBR Joint deviation -JH
        std_dev_l_hip = (left_hip_joint_upper_limit - left_hip_joint_lower_limit) / l_h_dev  # 99.7% should fall within +/- 3 std devs
        std_dev_l_shoulder = (left_shoulder_joint_upper_limit - left_shoulder_joint_lower_limit) / l_s_dev
        std_dev_l_leg = (left_leg_joint_upper_limit - left_leg_joint_lower_limit) / l_l_dev
        std_dev_r_hip = (right_hip_joint_upper_limit - right_hip_joint_lower_limit) / r_h_dev
        std_dev_r_shoulder = (right_shoulder_joint_upper_limit - right_shoulder_joint_lower_limit) / r_s_dev
        std_dev_r_leg = (right_leg_joint_upper_limit - right_leg_joint_lower_limit) / r_l_dev
    
        mean_l_hip = (left_hip_joint_lower_limit + left_hip_joint_upper_limit) / 2 - 0.200714
        mean_l_shoulder = (left_shoulder_joint_lower_limit + left_shoulder_joint_upper_limit) / 2 - 0.7854
        mean_l_leg = (left_leg_joint_lower_limit + left_leg_joint_upper_limit) / 2 - 0.7854
    
        mean_r_hip = (right_hip_joint_lower_limit + right_hip_joint_upper_limit) / 2 + 0.200714
        mean_r_shoulder =(right_shoulder_joint_lower_limit + right_shoulder_joint_upper_limit) / 2 - 0.7854
        mean_r_leg = (right_leg_joint_lower_limit + right_leg_joint_upper_limit) / 2 - 0.7854
    
        num_envs = len(env_ids)
        joint_position_offset = torch.zeros((num_envs,6), device=self.device)

        joint_position_offset[:, 0] = torch.clamp(torch.normal(mean=mean_l_hip, std=std_dev_l_hip, size=(num_envs,)), left_hip_joint_lower_limit, left_hip_joint_upper_limit)
        joint_position_offset[:, 1] = torch.clamp(torch.normal(mean=mean_l_shoulder, std=std_dev_l_shoulder, size=(num_envs,)), left_shoulder_joint_lower_limit, left_shoulder_joint_upper_limit)
        #joint_position_offset[:, 2] = torch.clamp(torch.normal(mean=mean_l_leg, std=std_dev_l_leg, size=(num_envs,)), left_leg_joint_lower_limit, left_leg_joint_upper_limit)
        joint_position_offset[:, 2] = torch.clamp(joint_position_offset[:, 1], left_leg_joint_lower_limit, left_leg_joint_upper_limit)
        
        if not all_random:
            joint_position_offset[:, 3] = torch.clamp(torch.normal(mean=mean_r_hip, std=std_dev_r_hip, size=(num_envs,)), right_hip_joint_lower_limit, right_hip_joint_upper_limit)
            joint_position_offset[:, 4] = joint_position_offset[:, 1]
            joint_position_offset[:, 5] = joint_position_offset[:, 2]
        else:
            joint_position_offset[:, 3] = torch.clamp(torch.normal(mean=mean_r_hip, std=std_dev_r_hip, size=(num_envs,)), right_hip_joint_lower_limit, right_hip_joint_upper_limit)
            joint_position_offset[:, 4] = torch.clamp(torch.normal(mean=mean_r_shoulder, std=std_dev_r_shoulder, size=(num_envs,)), right_shoulder_joint_lower_limit, right_shoulder_joint_upper_limit)
            joint_position_offset[:, 5] = torch.clamp(torch.normal(mean=mean_r_leg, std=std_dev_r_leg, size=(num_envs,)), right_leg_joint_lower_limit, right_leg_joint_upper_limit)



        return joint_position_offset

    def get_joint_offset_prev(self, env_ids, dof_limits_lower, dof_limits_upper, l_h_dev, l_s_dev, l_l_dev, r_h_dev, r_s_dev, r_l_dev, all_random):
        #* TWBR Joint limits -JH
        left_hip_joint_lower_limit = dof_limits_lower[0].item()
        left_hip_joint_upper_limit = dof_limits_upper[0].item()
        left_shoulder_joint_lower_limit = dof_limits_lower[1].item()
        left_shoulder_joint_upper_limit = dof_limits_upper[1].item()
        left_leg_joint_lower_limit = dof_limits_lower[2].item()
        left_leg_joint_upper_limit = dof_limits_upper[2].item()
    
        right_hip_joint_lower_limit = dof_limits_lower[4].item()
        right_hip_joint_upper_limit = dof_limits_upper[4].item()
        right_shoulder_joint_lower_limit = dof_limits_lower[5].item()
        right_shoulder_joint_upper_limit = dof_limits_upper[5].item()
        right_leg_joint_lower_limit = dof_limits_lower[6].item()
        right_leg_joint_upper_limit = dof_limits_upper[6].item()
    
        #* TWBR Joint deviation -JH
        std_dev_l_hip = (left_hip_joint_upper_limit - left_hip_joint_lower_limit) / l_h_dev  # 99.7% should fall within +/- 3 std devs
        std_dev_l_shoulder = (left_shoulder_joint_upper_limit - left_shoulder_joint_lower_limit) / l_s_dev

        std_dev_r_hip = (right_hip_joint_upper_limit - right_hip_joint_lower_limit) / r_h_dev
        std_dev_r_shoulder = (right_shoulder_joint_upper_limit - right_shoulder_joint_lower_limit) / r_s_dev
    
        mean_l_hip = (left_hip_joint_lower_limit + left_hip_joint_upper_limit) / 2 - 0.200714
        mean_l_shoulder = (left_shoulder_joint_lower_limit + left_shoulder_joint_upper_limit) / 2 + 0.26180049999999994
    
        mean_r_hip = (right_hip_joint_lower_limit + right_hip_joint_upper_limit) / 2 + 0.200714
        mean_r_shoulder =(right_shoulder_joint_lower_limit + right_shoulder_joint_upper_limit) / 2 + 0.26180049999999994

    
        num_envs = len(env_ids)
        joint_position_offset = torch.zeros((num_envs,6), device=self.device)
        joint_position_offset[:, 0] = torch.normal(mean=mean_l_hip, std=std_dev_l_hip, size=(num_envs,))
        joint_position_offset[:, 1] = torch.normal(mean=mean_l_shoulder, std=std_dev_l_shoulder, size=(num_envs,))
        joint_position_offset[:, 2] = torch.clamp(-joint_position_offset[:, 1], left_leg_joint_lower_limit, left_leg_joint_upper_limit)
        
        if not all_random:
            joint_position_offset[:, 3] = torch.normal(mean=mean_r_hip, std=std_dev_r_hip, size=(num_envs,))
            joint_position_offset[:, 4] = joint_position_offset[:, 1]
            joint_position_offset[:, 5] = joint_position_offset[:, 2]
        else:
            joint_position_offset[:, 3] = torch.normal(mean=mean_r_hip, std=std_dev_r_hip, size=(num_envs,))
            joint_position_offset[:, 4] = torch.normal(mean=mean_r_shoulder, std=std_dev_r_shoulder, size=(num_envs,))
            joint_position_offset[:, 5] = torch.clamp(-joint_position_offset[:, 4], right_leg_joint_lower_limit, right_leg_joint_upper_limit)

        return joint_position_offset

    def get_gaussian_velocity(self, env_ids, x_limits_lower, x_limits_upper, y_limits_lower, y_limits_upper, yaw_limits_lower, yaw_limits_upper, x_std_dev, y_std_dev, yaw_std_dev):
        #* TWBR velocity range deviation -JH
        std_dev_x = (x_limits_upper - x_limits_lower) / x_std_dev  # 99.7% should fall within +/- 3 std devs
        std_dev_y = (y_limits_upper - y_limits_lower) / y_std_dev
        std_dev_yaw = (yaw_limits_upper - yaw_limits_lower) / yaw_std_dev

        mean_x = (x_limits_lower + x_limits_upper) / 2.
        mean_y = (y_limits_lower + y_limits_upper) / 2.
        mean_yaw = (yaw_limits_lower + yaw_limits_upper) / 2.

        num_envs = len(env_ids)
        gaussian_velocity = torch.zeros((num_envs, 3), device=self.device)  # env_ids의 수와 3개 차원 (x, y, yaw)에 맞는 텐서

        # 각 env_id에 대해 가우시안 랜덤 값을 생성
        gaussian_velocity[:, 0] = torch.normal(mean=mean_x, std=std_dev_x, size=(num_envs,))
        gaussian_velocity[:, 1] = 0. #* torch.normal(mean=mean_y, std=std_dev_y, size=(num_envs,))
        gaussian_velocity[:, 2] = torch.normal(mean=mean_yaw, std=std_dev_yaw, size=(num_envs,))

        return gaussian_velocity
    
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
        self.num_per_env = int(num_robots / self.num_maps) #! (column*rows)별 로봇 배치숫자
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.border = int(self.border_size/self.horizontal_scale)
        self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border
        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg["curriculum"]:
            self.curriculum(num_robots, num_terrains=self.env_cols, num_levels=self.env_rows)
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

    def curriculum(self, num_robots, num_terrains, num_levels):
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


# @torch.jit.script
# def compute_postech_reward(hip_pos, shoulder_pos, thigh_pos, wheel_pos,
#                             hip_vel, shoulder_vel, thigh_vel, wheel_vel,
#                             body_pos, body_orientation, body_linear_vel, body_angular_vel,
#                             commands, reset_buf, progress_buf, max_episode_length, base_link_height, legs_contact, base_contact, shoulders_contact, projected_gravity, env_origins, torques,
#                             last_dof_vel, dof_vel, last_actions, actions):
#     # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    
#     epsilon = 1e-6  # Small constant to avoid division by zero
#     # k_orientation = 0.01 # 0.001
#     # k_lin_vel_z = -4.0 # -0.025
#     # k_ang_vel_xy = -0.05 # -0.0015
#     # k_gravity = 0.0 # 0.0
#     # k_contact = -0.25 # -0.02
#     # k_height = 0.0 # 0.001
#     # k_hip_align = 0.005 # 0.01
#     # k_des_hip = -0.005
#     # k_shoulder_align = 0.005 # 0.1
#     # k_position = -0.0 # -0.01
#     # k_torque = -0.000002 # 0.00001
#     # k_joint_acc = -0.0005 # 0.0002
#     # k_act_rate = -0.01 # 0.002
    
#     k_lin_vel_xy = 1.0 # 1.0
#     k_ang_vel_z = 0.5 # 0.5
#     k_alive_time = 0.0 
#     k_orientation = 1.0 / 2    
#     k_lin_vel_z = -1.0 / 5         # -0.5
#     k_ang_vel_xy = -0.1 / 10      # 0 .01
#     k_gravity = 0.0            # 0.0
#     k_contact = -1.0 * 5
#     k_height = -1.0 / 100
#     k_hip_align = -1.0 / 2       # -1.0 / 10 
#     k_des_hip = -0.0            # -0.005 sth wrong
#     k_shoulder_align = -1.0 / 2    # 0.0065 
#     k_leg_align = -1.0 / 3         # 0.0065
#     k_position = 1.0 / 5
#     k_torque = -0.0001 * 2            # -0.0001
#     k_joint_acc = -0.00005 / 2 # maybe 100?
#     k_act_rate = -0.025 / 5 # -0.0025
#     # Orientation reward: Prioritizing upright stance
#     normalized_roll = torch.atan2(torch.sin(body_orientation[:, 0]), torch.cos(body_orientation[:, 0]))
#     normalized_pitch = torch.atan2(torch.sin(body_orientation[:, 1]), torch.cos(body_orientation[:, 1]))
#     #reward_orientation = k_orientation * (1.0 - torch.abs(normalized_roll)/torch.tensor(np.pi)) * (1.0 - torch.abs(normalized_pitch)/torch.tensor(np.pi))
#     reward_orientation = k_orientation * torch.exp(-torch.abs(normalized_roll)) * torch.exp(-torch.abs(normalized_pitch))
#     # velocity tracking reward
#     lin_vel_error = torch.sum(torch.square(commands[:, :2] - body_linear_vel[:, :2]), dim=1)
#     ang_vel_error = torch.square(commands[:, 2] - body_angular_vel[:, 2])
#     reward_lin_vel_xy = k_lin_vel_xy * torch.exp(-lin_vel_error/0.25)
#     reward_ang_vel_z = k_ang_vel_z * torch.exp(-ang_vel_error/0.25)
    
#     # New Reward Term for Minimizing Rate of Change in Roll and Pitch
#     reward_ang_vel_xy = k_ang_vel_xy * torch.sum(torch.square(body_angular_vel[:, :2]), dim=1)
#     reward_lin_vel_z = k_lin_vel_z * torch.square(body_linear_vel[:, 2])
#     # orientation penalty
#     rew_gravity = k_gravity * torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
#     # Base Contact Penalty
#     reward_base_contact = k_contact * torch.sum(base_contact, dim=1)
    
#     # Base Contact Penalty
#     reward_legs_contact = k_contact * torch.sum(legs_contact, dim=1)
#     reward_position = k_position * torch.exp(-torch.norm(env_origins[:, :2] - body_pos[:, :2], dim=1))
#     # Height reward component
#     target_height = 0.39
#     sigma = 0.05   # Standard deviation, determines the flexibility around the target height
#     # Calculate the height difference from the target
#     height_diff = base_link_height - target_height
#     # Gaussian reward for height maintenance (GPT4 suggests)
#     #reward_height = k_height * torch.exp(-(height_diff ** 2) / (2 * sigma ** 2))
#     #reward_height = k_height * torch.exp(-torch.abs(base_link_height - target_height))
#     reward_height = k_height * torch.square(base_link_height - target_height)
#     # reward_hip_alignment = k_hip_align * ((torch.exp(-torch.abs(hip_pos[:, 0] - 0)) + torch.exp(-torch.abs(hip_pos[:, 1] - 0))) / 2)
#     reward_hip_alignment = k_hip_align * ((torch.exp(torch.abs(hip_pos[:, 0] - 0)) + torch.exp(torch.abs(hip_pos[:, 1] - 0))) / 2)
    
#     # left :  lower="-0.383972"  upper="0.7854"
#     # right:  lower="-0.7854"    upper="0.383972"
#     reward_hip_des = k_des_hip * (torch.abs(hip_pos[:, 0] * hip_pos[:, 1]))
#     #reward_shoulder_alignment = k_shoulder_align * (torch.abs(shoulder_pos[:, 0]) - torch.abs(shoulder_pos[:, 1])) # seems wrong...
#     # reward_shoulder_alignment = k_shoulder_align * (0.25 - torch.abs((shoulder_pos[:, 0]) + shoulder_pos[:, 1]))  # pos is the opposite value. Mirroed setup!!
#     # reward_leg_alignment = k_leg_align * (0.25 - torch.abs((thigh_pos[:, 0]) + thigh_pos[:, 1]))
#     reward_shoulder_alignment = k_shoulder_align * (torch.abs(shoulder_pos[:, 0] + shoulder_pos[:, 1]))
#     reward_leg_alignment = k_leg_align * torch.abs(thigh_pos[:, 0] + thigh_pos[:, 1])
#     reward_torque = k_torque * torch.sum(torch.square(torques), dim=1)
#     # joint acc penalty
#     reward_joint_acc =  k_joint_acc * torch.sum(torch.square(last_dof_vel - dof_vel), dim=1)
#     # action rate penalty
#     reward_action_rate = k_act_rate * torch.sum(torch.square(last_actions - actions), dim=1)
#     reward_alive_time = k_alive_time * progress_buf
#     # Total reward
#     reward = reward_orientation + reward_lin_vel_xy + reward_ang_vel_z + reward_ang_vel_xy + reward_lin_vel_z + rew_gravity + \
#              reward_base_contact + reward_legs_contact + reward_height + reward_hip_alignment + reward_hip_des +  \
#              reward_shoulder_alignment + reward_leg_alignment + reward_position + reward_torque + reward_joint_acc + reward_action_rate + reward_alive_time
#     reset = torch.any(legs_contact, dim=1) | torch.any(base_contact, dim=1) | torch.any(shoulders_contact, dim=1)
#     reset = torch.where(reset, torch.ones_like(reset_buf), reset_buf)
#     reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

#     # Update episode reward sums
#     # self.episode_sums["alive_time"] += reward_alive_time
#     # self.episode_sums["orientation"] += reward_orientation
#     # self.episode_sums["lin_vel_xy"] += reward_lin_vel_xy
#     # self.episode_sums["ang_vel_z"] += reward_ang_vel_z
#     # self.episode_sums["lin_vel_z"] += reward_lin_vel_z
#     # self.episode_sums["ang_vel_xy"] += reward_ang_vel_xy
#     # self.episode_sums["gravity"] += rew_gravity
#     # self.episode_sums["base_contact"] += reward_base_contact
#     # self.episode_sums["legs_contact"] += reward_legs_contact
#     # self.episode_sums["height"] += reward_height
#     # self.episode_sums["hip_alignment"] += reward_hip_alignment
#     # self.episode_sums["hip_desired"] += reward_hip_des
#     # self.episode_sums["shoulder_alignment"] += reward_shoulder_alignment
#     # self.episode_sums["leg_alignment"] += reward_leg_alignment
#     # self.episode_sums["position"] += reward_position
#     # self.episode_sums["torque"] += reward_torque
#     # self.episode_sums["joint_acc"] += reward_joint_acc
#     # self.episode_sums["action_rate"] += reward_action_rate

#     return reward, reset

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
