from gym.envs.registration import register

register(
  id='centrifuge-v0',
  entry_point='my_env.envs.CentrifugeEnv'
)