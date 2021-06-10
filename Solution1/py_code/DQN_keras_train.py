import numpy as np
import gym
import time

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from dqn_models import DQNModels

from env_centrifuge import CentrifugeEnv

#----------------------------------------------
# パラメータ
#----------------------------------------------

window_haba = 4

#----------------------------------------------


#----------------------------------------------
# 環境
#----------------------------------------------
env = CentrifugeEnv()
# env = DummyVecEnv([lambda: env])

ENV_NAME = 'Centrifuge-v0'

#----------------------------------------------

# Get the environment and extract the number of actions.
# env = gym.make(ENV_NAME)
# np.random.seed(123)
# env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(window_haba,) + env.observation_space.shape))
model.add(BatchNormalization())
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))


print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
# even the metrics!
# memory = SequentialMemory(limit=60000, window_length=1)
memory = SequentialMemory(limit=60000, window_length=window_haba)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

start = time.time()
# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=10000, visualize=False, nb_max_episode_steps=60, verbose=2)

elapsed_time = time.time() - start
# After training is done, we save the final weights.
dqn.save_weights('dqn_{ENV_NAME}_weights.h5f', overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")