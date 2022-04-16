import math 

class Epsilon():
    def __init__(self, epsilon_start=1, p_init=0.9, p_end=0.05, decay_rate=200, max_episodes=10000, max_steps=10000):
        self.val = epsilon_start
        self.p_init = p_init
        self.p_end = p_end
        self.decay = decay_rate
        self.max_episodes = max_episodes
        self.max_steps = max_steps 

    def linear_epsilon_decay(self, episode_num):
        current_episode_rate = (self.max_episodes - episode_num) / self.max_episodes
        epsilon_decay_rate = max(current_episode_rate, 0)
        self.val = ((self.p_init - self.p_end) * (epsilon_decay_rate)) + self.p_end

    def quad_epsilon_decay(self, episode_num):
        current_episode_rate = (self.max_episodes - episode_num) / self.max_episodes
        epsilon_decay_rate = math.exp(-1. * episode_num / self.decay)
        self.val = ((self.p_init - self.p_end) * (epsilon_decay_rate)) + self.p_end


