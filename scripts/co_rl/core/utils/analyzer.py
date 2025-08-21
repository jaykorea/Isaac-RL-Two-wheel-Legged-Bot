import os
import numpy as np
import matplotlib.pyplot as plt

class Analyzer:
    def __init__(self, env, analyze_items, log_dir, max_episode_length=999): 
        """
        Initialize the Analyzer with environment and configuration.
        """
        # Initialize the analyzer with environment and configuration        
        self.env = env
        # Set maximum episode length
        self.max_episode_length = max_episode_length
        # Store items to analyze (e.g. ['joint_vel', 'joint_torque'])
        self.analyze_items = analyze_items
        # Get observation info from the environment (defined in the environment's observation manager)
        self.obs_info = self.env.unwrapped.observation_manager.active_terms['obs_info']
        # Get observation configurations
        self.obs_info_cfgs = self.env.unwrapped.observation_manager._group_obs_term_cfgs['obs_info']
        # Get observation dimensions
        self.obs_info_dims = self.env.unwrapped.observation_manager._group_obs_term_dim['obs_info']
        # Store joint names from the environment
        self.joint_names = self.env.unwrapped.scene["robot"].joint_names
        
        # Validate analyze_items
        if not isinstance(analyze_items, list):
            raise ValueError("analyze_items should be a list of items to analyze.")

        for item in self.analyze_items:
            if item not in self.obs_info:
                raise ValueError(
                    f"[Analyzer] Invalid analyze item: '{item}' is not found in obs_info keys: {self.obs_info}"
                )
                
        # Create a directory to store analysis results
        self.log_dir = os.path.join(log_dir, "exported")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize a variable to save data  (e.g. {'joint_vel': [], 'joint_torque': []})
        self.data_store = {item: [] for item in self.analyze_items}

        self.obs_indices = {}
        idx = 0
        for key, shape in zip(self.obs_info, self.obs_info_dims):
            dim = shape[0]
            self.obs_indices[key] = (idx, idx + dim)
            idx += dim
            
        # Get observation scales for normalization
        # This will be used to scale the observations before saving
        # e.g. {'joint_pos': array([1., 1., ...]), 'joint_vel': (0.15)}
        self.obs_scales = {}
        for i, key in enumerate(self.obs_info):
            cfg = self.obs_info_cfgs[i]
            scale = cfg.scale.cpu().numpy() if cfg.scale != None else np.ones(self.obs_info_dims[i][0])
            self.obs_scales[key] = scale

        # Initialize: Create running trajectories for each environment
        # e.g. {'joint_vel': [[...], [...], [...], [...]], 'joint_torque': [[...], [...], [...], [...]]}
        num_envs = self.env.num_envs
        self._running_trajectories = {
            item: [[] for _ in range(num_envs)]
            for item in self.analyze_items
        }
        
    def append(self, obs_info):
        """Append observations to the analyzer for analysis.
        obs_info: A dictionary containing observation data for each environment.
        """
        obs = obs_info.cpu().numpy()
        num_envs = obs.shape[0]

        # Get the episode lengths from the environment
        # This is used to determine when an episode ends
        # e.g. [0, 1, 2, ..., 9]
        episode_lengths = self.env.unwrapped.episode_length_buf.cpu().numpy()
        
        # Iterate through each info and extract the relevant slices
        for key in self.analyze_items:
            start, end = self.obs_indices[key]
            raw = obs[:, start:end]                          
            scaled = raw * self.obs_scales[key] 

            for env_id in range(num_envs):
                self._running_trajectories[key][env_id].append(scaled[env_id])

        # Check if the episode has ended for any environment
        # If so, save the trajectories for that environment
        # and clear the running trajectories for that environment
        done_mask = (episode_lengths == self.max_episode_length - 1)
        for env_id in np.where(done_mask)[0]:
            for key in self.analyze_items:
                traj = np.stack(self._running_trajectories[key][env_id], axis=0)
                self.data_store[key].append(traj)
                self._running_trajectories[key][env_id].clear()

    def export(self):
        """Export the collected data to CSV files and plots."""
        
        # Check if there is any data to export
        for item in self.analyze_items:
            if item not in self.data_store or len(self.data_store[item]) == 0:
                print(f"[Analyzer] Warning: No data for {item}, skipping.")
                continue

            # Concatenate the data for the item across all environments
            data = np.concatenate(self.data_store[item], axis=0)
            header = self.joint_names if data.shape[1] == len(self.joint_names) else None
            self._save_csv(f"{item}.csv", data, header=header)

        self._plot_all()

    def _save_csv(self, filename, data, header=None):
        """Save the data to a CSV file."""
        path = os.path.join(self.log_dir, filename)
        if header is not None:
            np.savetxt(path, data, delimiter=",", header=",".join(header), comments='')
        else:
            np.savetxt(path, data, delimiter=",")
        print(f"[Analyzer] Saved: {path}")

    def _plot_all(self):
        """Plot all the collected data."""
        data_dict = {}
        ref_dim = None

        # Check if there is any data to plot
        for key in self.analyze_items:
            if key not in self.data_store or len(self.data_store[key]) == 0:
                print(f"[Analyzer] Warning: No data for {key}, skipping.")
                continue
            # Concatenate the data for the key across all environments
            data = np.concatenate(self.data_store[key], axis=0)  # [T, D]
            if ref_dim is None:
                ref_dim = data.shape[1]
            elif data.shape[1] != ref_dim:
                raise ValueError(f"All analyze_items must have same dimension. Mismatch found in '{key}'")

            data_dict[key] = data

        # Set up the plot
        if ref_dim is not None:
            D = ref_dim
            cols, rows = 4, (D + 3) // 4

            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), constrained_layout=True)
            axes = axes.flat if isinstance(axes, np.ndarray) else [axes]

            # If the items are joint torque and velocity, plot them as scatter
            is_scatter = set(self.analyze_items) == {'joint_vel', 'joint_torque'}

            for i in range(D):
                ax = axes[i]
                # Plot the data for each item
                if is_scatter:
                    x = data_dict['joint_torque'][:, i]
                    y = data_dict['joint_vel'][:, i]
                    ax.scatter(x, y, s=1, alpha=0.6)
                    ax.set_xlabel("Torque")
                    ax.set_ylabel("Velocity")
                else:
                    for key, data in data_dict.items():
                        ax.plot(np.arange(data.shape[0]), data[:, i], label=key, linewidth=1)
                    ax.set_xlabel("Time step")
                    ax.set_ylabel("Value")
                    ax.legend(fontsize=7)

                title = self.joint_names[i] if i < len(self.joint_names) else f"Joint[{i}]"
                ax.set_title(title, fontsize=9)

            for i in range(D, len(axes)):
                axes[i].axis('off')

            title_text = "Torque-Velocity Scatter" if is_scatter else "Joint-wise Comparison of Observations Over Time"
            fig.suptitle(title_text, fontsize=14)
            plt.show()
