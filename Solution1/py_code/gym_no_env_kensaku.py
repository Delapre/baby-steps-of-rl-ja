from gym import envs
import gym

for spec in envs.registry.all():
    try:
        env = gym.make(spec.id)
        print(spec.id)
    except:
        pass