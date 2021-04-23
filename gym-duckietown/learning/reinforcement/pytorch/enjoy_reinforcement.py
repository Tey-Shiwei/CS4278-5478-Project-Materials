import ast
import argparse
import logging

import os
import numpy as np

# Duckietown Specific
from gym_duckietown.envs import DuckietownEnv
from reinforcement.pytorch.ddpg import DDPG
from utils.env import launch_env
from utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def _enjoy():          
    # Launch the env with our helper function
    # env = launch_env()
    SEED = 1
    MAP = 'map4'
    env = DuckietownEnv(
        map_name = MAP, domain_rand = False, draw_bbox = False,
        max_steps = 1500,
        seed = SEED)
    print("Initialized environment")

    # Wrappers
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")
    policy.load(filename='ddpg_40000', directory='reinforcement/pytorch/models/')

    obs = env.reset()
    done = False

    while True:
        while not done:
            action = policy.predict(np.array(obs))
            print(action)
            # Perform action
            obs, reward, done, _ = env.step(action)
            env.render()
        done = False
        obs = env.reset()        

if __name__ == '__main__':
    _enjoy()
