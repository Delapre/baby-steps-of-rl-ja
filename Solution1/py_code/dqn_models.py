import numpy as np
import gym
import time

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Activation, Flatten, LSTM, BatchNormalization

class DQNModels():

    # def __init__(self,input_tensor, action_space):
        # self.input_tensor = input_tensor
        # self.action_space = action_space

    def dqn_lstm(self, input_tensor, action_space):
        self.input_tensor = input_tensor
        self.action_space = action_space
        # print("shape : ", self.observation_space)
        model = Sequential()
        model.add(LSTM(16, input_shape=self.input_tensor))
        model.add(BatchNormalization())
        # model.add(Dense(16))
        # model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(self.action_space))
        model.add(Activation('linear'))
        
        return model

    def dqn_dense(window_haba,observation_space,action_space):
        model = Sequential()
        model.add(Flatten(input_shape=(window_haba,) + observation_space))
        model.add(BatchNormalization())
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(action_space))
        model.add(Activation('linear'))
        
        return model