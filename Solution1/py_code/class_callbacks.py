# ログを記録するためのクラスの定義
class EpisodeLogger(rl.callbacks.Callback):
    def __init__(self):
        self.epsode_rewards = {}
    def on_episode_begin(self, episode, logs):
        self.epsode_rewards[episode] = []
    def on_step_end(self, step, logs):
        episode_rewards = logs['episode_rewards']
        self.episode_rewards[episode].append(logs['reward'])
 
