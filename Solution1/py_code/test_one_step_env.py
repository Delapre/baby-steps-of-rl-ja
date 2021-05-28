import gym
from env_centrifuge import CentrifugeEnv

env = CentrifugeEnv()

state = env.reset()

print("start steps")
while True :


    action = env.action_space.sample()
    
    print("action  : ",action,"   state : ",state)
    state, reward, done, info, = env.step(action)
    
    env.render()

    print("step : ",info ,'   reward :',reward,"     done : ",done)

    if done:
        print("done")
        break
