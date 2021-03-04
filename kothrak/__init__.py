from gym.envs.registration import register

register(
    id='kothrak-v0',
    entry_point='kothrak.envs:KothrakEnv',
)
