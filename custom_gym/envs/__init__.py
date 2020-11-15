from gym.envs.registration import register

register(id='UAVEnv-v0',
         entry_point='envs.custom_env_dir:UAVEnv') # Puoi settare anche un 'ma_episode_steps=' e un 'reward_threshold='