import os
import numpy as np
import matplotlib.pyplot as plt

class Analyzer:
    def __init__(self, env, analyze_items, joint_names, log_dir, max_episode_length=999): 
        self.env = env
        self.max_episode_length = max_episode_length
        self.analyze_items = analyze_items
        self.obs_info = self.env.unwrapped.observation_manager.active_terms['obs_info']
        self.obs_info_dims = self.env.unwrapped.observation_manager._group_obs_term_dim['obs_info']

        if not isinstance(analyze_items, list):
            raise ValueError("analyze_items should be a list of items to analyze.")

        for item in self.analyze_items:
            if item not in self.obs_info:
                raise ValueError(
                    f"[Analyzer] Invalid analyze item: '{item}' is not found in obs_info keys: {self.obs_info}"
                )

        self.joint_names = joint_names
        self.log_dir = os.path.join(log_dir, "exported")
        os.makedirs(self.log_dir, exist_ok=True)
        self.data_store = {item: [] for item in self.analyze_items}

        # 초기화: 환경 수만큼 running trajectory 저장
        num_envs = self.env.num_envs
        self._running_trajectories = {
            item: [[] for _ in range(num_envs)]
            for item in self.analyze_items
        }

    def append(self, obs_info):
        obs = obs_info.cpu().numpy()
        num_envs = obs.shape[0]

        if not hasattr(self, 'obs_indices'):
            self.obs_indices = {}
            idx = 0
            for key, shape in zip(self.obs_info, self.obs_info_dims):
                dim = shape[0]
                self.obs_indices[key] = (idx, idx + dim)
                idx += dim

        episode_lengths = self.env.unwrapped.episode_length_buf.cpu().numpy()
        for env_id in range(num_envs):
            for key in self.analyze_items:
                start, end = self.obs_indices[key]
                value = obs[env_id, start:end]
                self._running_trajectories[key][env_id].append(value)

            # Trajectory 저장 조건: max 에피소드 길이에 도달한 경우
            if episode_lengths[env_id] == self.max_episode_length - 1:
                for key in self.analyze_items:
                    traj = np.stack(self._running_trajectories[key][env_id], axis=0)
                    self.data_store[key].append(traj)
                    self._running_trajectories[key][env_id].clear()


    def export(self):
        for item in self.analyze_items:
            if item not in self.data_store or len(self.data_store[item]) == 0:
                print(f"[Analyzer] Warning: No data for {item}, skipping.")
                continue

            data = np.concatenate(self.data_store[item], axis=0)
            header = self.joint_names if data.shape[1] == len(self.joint_names) else None
            self._save_csv(f"{item}.csv", data, header=header)

        self._plot_all()

    def _save_csv(self, filename, data, header=None):
        path = os.path.join(self.log_dir, filename)
        if header is not None:
            np.savetxt(path, data, delimiter=",", header=",".join(header), comments='')
        else:
            np.savetxt(path, data, delimiter=",")
        print(f"[Analyzer] Saved: {path}")

    def _plot_all(self):
        data_dict = {}
        ref_dim = None

        for key in self.analyze_items:
            if key not in self.data_store or len(self.data_store[key]) == 0:
                print(f"[Analyzer] Warning: No data for {key}, skipping.")
                continue

            data = np.concatenate(self.data_store[key], axis=0)  # [T, D]
            if ref_dim is None:
                ref_dim = data.shape[1]
            elif data.shape[1] != ref_dim:
                raise ValueError(f"All analyze_items must have same dimension. Mismatch found in '{key}'")

            data_dict[key] = data

        D = ref_dim
        cols, rows = 4, (D + 3) // 4

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), constrained_layout=True)
        axes = axes.flat if isinstance(axes, np.ndarray) else [axes]

        is_scatter = set(self.analyze_items) == {'joint_vel', 'joint_torque'}

        for i in range(D):
            ax = axes[i]

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
