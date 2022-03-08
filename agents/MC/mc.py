#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.
    You could test the correctness of your code
    by typing 'nosetests -v mc_test.py' in the terminal.

    You don't have to follow the comments to write your code. They are provided
    as hints in case you need.
'''
#-------------------------------------------------------------------------

def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and his otherwise

    Parameters:
    -----------
    observation

    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    # get parameters from observation
    score, dealer_score, usable_ace = observation
    action = 1 
    if score >= 20:
        action = 0 

    ############################
    return action

def generate_trajectory(policy, env):
    state = env.reset()
    episode_buffer = []

    # loop until episode generation is done
    while True:
        # select an action 
        action = policy(state)
            
        # return a reward and new state 
        next_state, reward, done, _ = env.step(action)

        # append state, action, reward to episode
        episode_buffer.insert(0, (state, action, reward)) # format episode to start at timestep 0 and end at timestep T 

        if done:
            break
        state = next_state
        # update state to new state

    return episode_buffer


def mc_prediction(policy, env, n_episodes, gamma = 1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.

    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value

    Note: at the begining of each episode, you need initialize the environment using env.reset()
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)

    ############################
    # YOUR IMPLEMENTATION HERE #
    for e in range(n_episodes):
        state = env.reset()

        # initialize the episode
        episode_buffer = [] # list that will contain the samples of the state, action and reward of the episode

        # generate empty episode list
        state_visited = [] 

        # loop until episode generation is done
        while True:
            # select an action 
            action = policy(state)
            
            # return a reward and new state 
            next_state, reward, done, _ = env.step(action)

            # append state, action, reward to episode
            episode_buffer.insert(0, (state, action, reward)) # format episode to start at timestep 0 and end at timestep T 

            if done:
                break
            state = next_state
            # update state to new state 


        G = 0 # average reward 
        # loop for each step of episode, t = T-1, T-2,...,0
        for step in episode_buffer:
            (state, _, reward) = step
            # compute G 
            G = (gamma * G) + reward 

            # unless state_t appears in states 
            if state not in state_visited:
                state_visited.append(state)

                # update return_count 
                returns_count[state] += 1 

                # update return_sum
                returns_sum[state] += G 

                # calculate average return for this state over all sampled episodes
                V[state] = returns_sum[state] / returns_count[state] 



    ############################

    return V

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.

    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1

    Returns:
    --------
    action: int
        action based current state
    Hints:
    ------
    With probability (1 − epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    random_val = random.random()
    action = random.randrange(nA)
    if random_val < 1 - epsilon:
        action = np.argmax(Q[state])
    
    ############################
    return action

def mc_control_epsilon_greedy(env, n_episodes, gamma = 1.0, epsilon = 0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-(0.1/n_episodes) during each episode
    and episode must > 0.
    """

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    nA = env.action_space.n
    eps = 1
    Q = defaultdict(lambda: np.zeros(nA))
    version = 'a'
    ############################
    # YOUR IMPLEMENTATION HERE #
    for e in range(1, n_episodes + 1):
        state = env.reset()

        # initialize the episode 
        episode_buffer = []

        # generate empty episode list
        state_action_visited = []

        # loop until one episode generation is done
        while True:

            # get an action from epsilon greedy policy 
            action = epsilon_greedy(Q, state, nA, epsilon)

            # return a reward and new state
            next_state, reward, done, _ = env.step(action)

            # append state, action, reward to episode
            episode_buffer.insert(0, ((state, action), reward)) 

            # update state to new state 
            if done:
                break 
            state = next_state 


        G = 0 
        # loop for each step of episode, t = T-1, T-2, ...,0 
        for (state_action, reward) in episode_buffer:

            # compute G
            G = (gamma * G) + reward 

            # unless the pair state_t, action_t appears in <state action> pair list
            if state_action not in state_action_visited:
                state_action_visited.append(state_action)

                # update return_count
                returns_count[state_action] += 1

                # update return_sum
                returns_sum[state_action] += G

                (state, action) = state_action

                # calculate average return for this state over all sampled episodes 
                if version == 'a':
                    Q[state][action] = returns_sum[state_action] / returns_count[state_action]
                else:
                    Q[state][action] = Q[state][action] + ((1 / returns_count[state_action]) * (G - Q[state][action])) 
        eps = 1/e
        epsilon = epsilon - (0.1 / n_episodes) 

    return Q
