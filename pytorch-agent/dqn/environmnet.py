import gym
from gym.wrappers import FrameStack, AtariPreprocessing

class Gym_Env():
    def __init__(self, env_name, max_steps=1000, max_episodes=10000):
        self.le_env = FrameStack(AtariPreprocessing(gym.make(env_name, frameskip=1)), 4)
        self.max_steps = max_steps
        self.max_episodes = max_episodes