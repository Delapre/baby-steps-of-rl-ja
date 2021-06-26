import numpy as np
import gym

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from dqn_models import DQNModels

from env_centrifuge_continuous import CentrifugeEnv
#----------------------------------------------
# パラメータ
#----------------------------------------------

window_haba = 4
model_params = "./Solution1/kekka/dqn_lstm_Centrifuge-v0_3"

window_haba = 8
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
print("*** nb_actions : ",nb_actions)
input_tensor =  (window_haba,) + env.observation_space.shape
print("*** input_tensor : ",input_tensor)

# Next, we build a very simple model.
dqn_model = DQNModels()
model = dqn_model.dqn_lstm(input_tensor,nb_actions)

print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=window_haba)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
# dqn.fit(env, nb_steps=600, visualize=False, nb_max_episode_steps=60, verbose=2)

dqn.load_weights('dqn_{ENV_NAME}_weights.h5f')

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)