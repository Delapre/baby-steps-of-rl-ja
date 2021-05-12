import random
import os
import argparse
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import gym
from fn_framework import FNAgent, Trainer, Observer


class ValueFunctionAgent(FNAgent):

    def save(self, model_path):
        joblib.dump(self.model, model_path)

    @classmethod
    def load(cls, env, model_path, epsilon=0.0001):
        actions = list(range(env.action_space.n))
        with open('log.txt', 'a') as f:
            print("load_actions:",actions,file = f)
        agent = cls(epsilon, actions)
        agent.model = joblib.load(model_path)
        agent.initialized = True
        return agent

    def initialize(self, experiences):
        scaler = StandardScaler()
        estimator = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1)
        self.model = Pipeline([("scaler", scaler), ("estimator", estimator)])

        states = np.vstack([e.s for e in experiences])
        with open('log.txt', 'a') as f:
            print("initialize_states:",states,file=f)

        self.model.named_steps["scaler"].fit(states)

        # Avoid the predict before fit.
        self.update([experiences[0]], gamma=0)
        self.initialized = True
        print("Done initialization. From now, begin training!")

    def estimate(self, s):
        estimated = self.model.predict(s)[0]
        return estimated

    def _predict(self, states):
        if self.initialized:
            predicteds = self.model.predict(states)
        else:
            size = len(self.actions) * len(states)
            predicteds = np.random.uniform(size=size)
            predicteds = predicteds.reshape((-1, len(self.actions)))
        return predicteds

    def update(self, experiences, gamma):
        with open('log.txt', 'a') as f:
            print("update_experiences[0]:\n",experiences[0],file=f)
        states = np.vstack([e.s for e in experiences])
        n_states = np.vstack([e.n_s for e in experiences])
        with open('log.txt', 'a') as f:
            print("update_states:\n",states,"\n   update_n_states:\n",n_states,file=f)


        estimateds = self._predict(states)
        future = self._predict(n_states)
        with open('log.txt', 'a') as f:
            print("update_estimateds:\n",estimateds,"\n   update_future:\n",future,file=f)

        for i, e in enumerate(experiences):
            reward = e.r
            if not e.d:
                reward += gamma * np.max(future[i])
            estimateds[i][e.a] = reward
            with open('log.txt', 'a') as f:
                print("estimateds[i][e.a]:\n",e.a,file=f)


        estimateds = np.array(estimateds)
        states = self.model.named_steps["scaler"].transform(states)
        with open('log.txt', 'a') as f:
            print("model_update_states:\n",states,"\n   model_update_estimateds:\n",estimateds,file=f)

        self.model.named_steps["estimator"].partial_fit(states, estimateds)


class CartPoleObserver(Observer):

    def transform(self, state):
        return np.array(state).reshape((1, -1))


class ValueFunctionTrainer(Trainer):

    def train(self, env, episode_count=220, epsilon=0.1, initial_count=-1,
              render=False):
        actions = list(range(env.action_space.n))
        agent = ValueFunctionAgent(epsilon, actions)
        with open('log.txt', 'a') as f:
            print("trainer_train_actions:\n",actions,"\n   trainer_train_agent:\n",agent,file=f)
        self.train_loop(env, agent, episode_count, initial_count, render)
        return agent

    def begin_train(self, episode, agent):
        agent.initialize(self.experiences)

    def step(self, episode, step_count, agent, experience):
        if self.training:
            batch = random.sample(self.experiences, self.batch_size)
            agent.update(batch, self.gamma)

    def episode_end(self, episode, step_count, agent):
        rewards = [e.r for e in self.get_recent(step_count)]
        self.reward_log.append(sum(rewards))

        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval:]
            self.logger.describe("reward", recent_rewards, episode=episode)


def main(play):
    env = CentrifugeObserver(gym.make("CartPole-v0"))
    trainer = ValueFunctionTrainer()
    path = trainer.logger.path_of("value_function_agent.pkl")

    if play:
        agent = ValueFunctionAgent.load(env, path)
        agent.play(env)
    else:
        if os.path.isfile('./log.txt'):os.remove("log.txt")
        trained = trainer.train(env)
        trainer.logger.plot("Rewards", trainer.reward_log,
                            trainer.report_interval)
        trained.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VF Agent")
    parser.add_argument("--play", action="store_true",
                        help="play with trained model")

    args = parser.parse_args()
    main(args.play)
