# CHECK THE ENVIRONMENT #

import gym
import gym_rendezvous
from stable_baselines.common.env_checker import check_env

env = gym.make('rendezvous-v1')

# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)