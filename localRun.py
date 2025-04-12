import gym
import simple_driving
# import pybullet_envs
import pybullet as p
import numpy as np
import math
from collections import defaultdict
import pickle
import torch
import random

######################### renders image from third person perspective for validating policy ##############################
# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='tp_camera')
##########################################################################################################################

######################### renders image from onboard camera ###############################################################
# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='fp_camera')
##########################################################################################################################

######################### if running locally you can just render the environment in pybullet's GUI #######################
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)
env = env.unwrapped
##########################################################################################################################

state, info = env.reset()
# frames = []
# frames.append(env.render())

for i in range(2000):
    action = env.action_space.sample() # This take an action
    state, reward, done, _, info = env.step(action) # Pick a random action from the list of valid actions.
    # frames.append(env.render())  # if running locally not necessary unless you want to grab onboard camera image
    if done:
        break

env.close()