from gym.envs.registration import register

register(
    id='gidi-v0',
    entry_point='gidi_env.envs:GidiEnv'
)

register(
    id='gidi-v1',
    entry_point='gidi_env.envs:GidiNew'
)
