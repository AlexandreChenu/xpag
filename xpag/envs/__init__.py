import gym
from gym.envs.registration import register

# dubins maze environment
from .dubins_mazeenv.mazeenv_cst_speed import DubinsMazeEnv
from .dubins_mazeenv.mazeenv_cst_speed_wrappers import DubinsMazeEnvGCPHERSB3

# fetch environment from Go-Explore 2 (Ecoffet et al.)
# from .fetchenv.fetch_env import MyComplexFetchEnv
# from .fetchenv.fetchenv_wrappers import ComplexFetchEnvGCPHERSB3


## mazeenv from Guillaume Matheron with a Dubins car
print("REGISTERING DubinsMazeEnv")
register(
    id='DubinsMazeEnvGCPHERSB3-v0',
    # entry_point='envs.dubins_mazeenv.mazeenv_wrappers:DubinsMazeEnvGCPHERSB3')
    entry_point='xpag.envs.dubins_mazeenv.mazeenv_cst_speed_wrappers:DubinsMazeEnvGCPHERSB3')

print("REGISTERING DubinsMazeEnv")
register(
    id='DubinsMazeEnv-v0',
    # entry_point='envs.dubins_mazeenv.mazeenv_wrappers:DubinsMazeEnvGCPHERSB3')
    entry_point='xpag.envs.dubins_mazeenv.mazeenv_cst_speed:DubinsMazeEnv',
    max_episode_steps=30)


## fetch environment from Go-Explore 2 (Ecoffet et al.)
# print("REGISTERING FetchEnv-v0 & FetchEnvGCPHERSB3-v0")
# register(
#     id='FetchEnv-v0',
#     entry_point='envs.fetchenv.fetch_env:MyComplexFetchEnv',
# )
#
# register(
#     id='FetchEnvGCPHERSB3-v0',
#     entry_point='envs.fetchenv.fetchenv_wrappers:ComplexFetchEnvGCPHERSB3',
# )
#
# ## Humanoid environment from Mujoco
# print("REGISTERING HumanoidEnv-v0 & HumanoidEnvGCPHERSB3-v0")
# register(
#     id='HumanoidEnv-v0',
#     entry_point='envs.humanoid.humanoidenv:MyHumanoidEnv',
# )
#
# register(
#     id='HumanoidEnvGCPHERSB3-v0',
#     entry_point='envs.humanoid.humanoidenv_wrappers:HumanoidEnvGCPHERSB3',
# )
