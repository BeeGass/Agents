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
 
