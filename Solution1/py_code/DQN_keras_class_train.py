import numpy as np
import gym
import time
import rev

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from keras.callbacks import TensorBoard

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from dqn_models import DQNModels

from env_centrifuge_continuous import CentrifugeEnv

###############################################
# 環境
#----------------------------------------------
env = CentrifugeEnv()
# env = DummyVecEnv([lambda: env])

ENV_NAME = 'Centrifuge-v0'

###############################################
# パラメータ
#----------------------------------------------

window_haba = 8

# log_filepath = "./Solution1/py_code/logs/" + ENV_NAME
# print(log_filepath)

model_mei = "dqn_latm"
tsuika = None
###############################################

# Get the environment and extract the number of actions.
# env = gym.make(ENV_NAME)
# np.random.seed(123)
# env.seed(123)
nb_actions = env.action_space.n
print("*** nb_actions : ",nb_actions)
input_tensor =  (window_haba,) + env.observation_space.shape
print("*** input_tensor : ",input_tensor)

dqn_model = DQNModels()
model = dqn_model.dqn_lstm(input_tensor,nb_actions)


# Finally, we configure and compile our agent. You can use every built-in
# tensorflow.keras optimizer and even the metrics!
# memory = SequentialMemory(limit=60000, window_length=1)
memory = SequentialMemory(limit=60000, window_length=window_haba)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,\
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

model.summary()

###############################################
# fit
#----------------------------------------------
start = time.time()

saki,revision,param_data = rev.rev(tsuika,ENV_NAME,model_mei)
log_filepath = saki + revision + "/logs"
callbacks = [TensorBoard (log_dir=log_filepath, histogram_freq=0)]

# Okay, now it's time to learn something! We visualize the training here for show,
#  but this slows down training quite a lot. You can always safely abort the 
# training prematurely using Ctrl + C.
dqn.fit(env, nb_steps=600, visualize=False, nb_max_episode_steps=60, verbose=2,\
         callbacks = callbacks)

elapsed_time = time.time() - start
# After training is done, we save the final weights.

###############################################
# post process
#----------------------------------------------
dqn.save_weights(saki + revision + 'dqn_{ENV_NAME}_weights.h5f', overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")