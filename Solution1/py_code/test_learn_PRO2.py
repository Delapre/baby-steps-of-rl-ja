import gym
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import ppo2
from env_centrifuge import CentrifugeEnv

env = CentrifugeEnv()
env = DummyVecEnv([lambda: env])

model = PRO2('MlpPolicy', env, verbose=1)

model = PRO2.load('centrifuge_model')

model.learn(total_timesteps =12800)

model.save('centrifuge_model')

state = env.reset()

total_reward = 0

while True:
    env.render()
    action, _ = model.predict(state)

    state, reward, done, info = env.step(action)

    total_reward += reward[0]

    if done:
        print('reward: ', total_reward)

        state = env.reset()

        total_reward = 0
