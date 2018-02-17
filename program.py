#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import input

import gym
import deeprl_hw1.lake_envs as lake_env
import time
import deeprl_hw1.rl as rl

import matplotlib.pyplot as plt
import numpy as np

import plotly.plotly as py
import plotly.tools as tls
import datetime


# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api
def plot_heatmap(value_func):
   

    a = np.reshape(value_func, (4,4))
    plt.imshow(a, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()

def run_random_policy(env):
    """Run a random policy for the given environment.

    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    initial_state = env.reset()
    env.render()
    time.sleep(1)  # just pauses so you can see the output

    policy, value_func, p_iterations, v_iterations= rl.policy_iteration_sync(env,0.9)
    rl.print_policy, ('L', 'D', 'R', 'U')
    #plot_heatmap(value_func)
    #print(policy, value_func, p_iterations, v_iterations)

    return(policy, value_func, p_iterations, v_iterations)


def print_env_info(env):
    print('Environment has %d states and %d actions.' % (env.nS, env.nA))


def print_model_info(env, state, action):
    transition_table_row = env.P[state][action]
    print(
        ('According to transition function, '
         'taking action %s(%d) in state %d leads to'
         ' %d possible outcomes') % (lake_env.action_names[action],
                                     action, state, len(transition_table_row)))
    for prob, nextstate, reward, is_terminal in transition_table_row:
        state_type = 'terminal' if is_terminal else 'non-terminal'
        print(
            '\tTransitioning to %s state %d with probability %f and reward %f'
            % (state_type, nextstate, prob, reward))


def main():
    # create the environment
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    a=datetime.datetime.now()

    #print_env_info(env)
    #print_model_info(env, 0, lake_env.DOWN)
    #print_model_info(env, 1, lake_env.DOWN)
    #print_model_info(env, 14, lake_env.RIGHT)
    

    
    initial_state = env.reset()
    env.render()
    time.sleep(1)

    print(rl.value_iteration_sync(env,0.9))
    b=datetime.datetime.now()
    print(b-a)
   
   
if __name__ == '__main__':
    main()
