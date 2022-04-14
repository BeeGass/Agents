class Epsilon():
    def __init__(self, env, epsilon_start=1, p_init=0.9, p_end=0.05, decay_rate=200):
        self.p_init = p_init
        self.p_end = p_end
        self.decay = decay_rate
        self.epsilon = epsilon_start
        self.environment = env 

    def linear_epsilon_decay(self, episode_num):
        current_episode_rate = (self.environment.max_episodes - episode_num) / self.environment.max_episodes
        epsilon_decay_rate = max(current_episode_rate, 0)
        return ((self.p_init - self.p_end) * (epsilon_decay_rate)) + self.p_end

    def quad_epsilon_decay(self, episode_num):
        current_episode_rate = (self.environment.max_episodes - episode_num) / self.environment.max_episodes
        epsilon_decay_rate = math.exp(-1. * episode_num / epsilon_decay)
        return ((self.p_init - self.p_end) * (epsilon_decay_rate)) + self.p_end


