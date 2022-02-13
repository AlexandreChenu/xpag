from datetime import datetime
from IPython import embed
import numpy as np
import torch
import gym
import gym_gmazes
import string
import collections
import functools
import random
import time
from datetime import datetime
import brax
import jax
from brax import envs
from brax.envs import to_torch
from brax.io import metrics
# from brax.io import torch
from IPython import embed
import argparse
import os
import logging
import xpag
from xpag.plotting.basics import plot_episode_2d
import re

# have torch allocate on device first, to prevent JAX from swallowing up all the
# GPU memory. By default JAX will pre-allocate 90% of the available GPU memory:
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
v = torch.ones(1, device='cuda')
print(torch.cuda.memory_allocated(device='cuda'), 'bytes')

gmaze_frame_skip = 2  # only used by gym-gmazes environments
gmaze_walls = []  # only used by gym-gmazes environments
env_name = 'HalfCheetah-v3'
# env_name = 'brax-halfcheetah-v0'
num_envs = 1
episode_max_length = 1000
buffer_name = 'DefaultBuffer'
buffer_size = 1e6
sampler_name = 'DefaultSampler'
goalenv_sampler_name = 'HER'  # only for environments with goals
agent_name = 'SAC'
seed = 0

agent, env, replay_buffer, sampler, datatype, device = xpag.tl.configure(
    env_name, num_envs, gmaze_frame_skip, gmaze_walls, episode_max_length, buffer_name,
    buffer_size, sampler_name, goalenv_sampler_name, agent_name, seed
)

save_dir = os.path.join(os.path.expanduser("~"),
                        "results",
                        "xpag",
                        datetime.now().strftime("%Y%m%d_%H%M%S"))

plot_episode = functools.partial(
    plot_episode_2d,
    plot_env_function=env.plot if hasattr(env, "plot") else None
)
plot_episode = None
max_t = int(1e6)
train_ratio = 1.
batch_size = 256
start_random_t = 0
eval_freq = 1000 * 5
eval_eps = 5
save_freq = 0

xpag.tl.learn(agent, env, num_envs, episode_max_length,
              max_t, train_ratio, batch_size, start_random_t, eval_freq, eval_eps,
              save_freq, replay_buffer, sampler, datatype, device, save_dir=save_dir,
              save_episode=False, plot_function=plot_episode)
