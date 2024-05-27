import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig

def get_cfg(cfg_path, num_envs, task_name):
    with initialize(config_path=cfg_path):
        cfg = compose(config_name="config", overrides=[f"task={task_name}"])
        cfg.task.env.numEnvs = num_envs
        return cfg