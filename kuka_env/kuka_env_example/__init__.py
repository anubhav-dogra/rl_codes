from gymnasium.envs.registration import register
# from gym.envs.registration import register

register(
    id='iiwaEnvPos-v0',
    entry_point='kuka_env_example.envs:iiwaEnvPos',

)
