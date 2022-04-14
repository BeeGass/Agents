import gym
from gym.wrappers import (
        FrameStack, 
        AtariPreprocessing, 
        RecordEpisodeStatistics
)

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

class Environment():

    def __init__(self, env_name, seed, max_steps=1000, max_episodes=10000):
        self.env_name = env_name
        self.seed = seed 
        self.max_steps = max_steps
        self.max_episodes = max_episodes

    def make_env(env_name, seed=42):
        env = gym.make(env_name)
        env = RecordEpisodeStatistics(env)
        env = ClipRewardEnv(env)
        env = EpisodicLifeEnv(env)
        env = AtariPreprocessing(env)
        env = FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env 

