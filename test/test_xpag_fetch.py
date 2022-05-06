#!/usr/bin python -w

import os
import jax
import flax

import xpag
from xpag.wrappers import gym_vec_env
from xpag.buffers import DefaultEpisodicBuffer
from xpag.samplers import DefaultEpisodicSampler, HER, HER_DCIL
from xpag.goalsetters import DefaultGoalSetter
from xpag.agents import SAC, SAC_bonus
from xpag.tools import learn
from xpag.tools import mujoco_notebook_replay


import os
import numpy as np
import matplotlib.pyplot as plt
from xpag.tools.eval import single_rollout_eval
from xpag.tools.utils import hstack
from xpag.tools.logging import eval_log_reset
from xpag.tools.timing import timing_reset
from xpag.buffers import Buffer
from xpag.agents.agent import Agent
from xpag.goalsetters.goalsetter import GoalSetter
from typing import Dict, Any, Tuple, Union
from collections import OrderedDict

import gym_gfetch

import argparse

def plot_traj(trajs, eval_env, save_dir, it=0):
    fig = plt.figure()
    fig.set_figheight(5.2)
    fig.set_figwidth(5.2)
    ax = fig.add_subplot(projection='3d')

    ax.set_xlim((0., 1.5))
    ax.set_ylim((-1., 2.))
    ax.set_zlim((0., 2.2))

    fake_env = eval_env.env_fns[0]().unwrapped
    fake_env.reset()

    #print("traj = ", traj)
    for traj in trajs:

        for i in range(traj[0].shape[0]):
            success = False
            X = []
            Y = []
            Z = []

            X_object = []
            Y_object = []
            Z_object = []

            #print("\n desired_goal = ", fake_env.goal)
            for goal in traj:
                #print("achieved_goal = ", goal)
                #if np.linalg.norm(goal[i] - fake_env.goal) < 0.05:
                    #print("achieved_goal[i] = ", goal[0])
                    #success = True
                X.append(goal[i][0])
                Y.append(goal[i][1])
                Z.append(goal[i][2])

                X_object.append(goal[i][3])
                Y_object.append(goal[i][4])
                Z_object.append(goal[i][5])
            if success:
                ax.plot(X, Y, Z, c="m")
            else:
                ax.plot(X, Y, Z, c="pink")
                ax.plot(X_object, Y_object, Z_object, c = "brown")

    #X_eval = [goal[0] for goal in eval_traj]
    #Y_eval = [goal[1] for goal in eval_traj]
    #Z_eval = [goal[2] for goal in eval_traj]
    #ax.plot(X_eval, Y_eval, Z_eval, c = "red")

    visu_success_zones(eval_env, ax)

    for azim_ in range(45,360,90):
        #ax.view_init(elev=0., azim = azim_)
        ax.view_init(azim = azim_)
        plt.savefig(save_dir + "/trajs_" + str(azim_) + "_it_" + str(it) + ".png")

    plt.close(fig)
    return

def visu_success_zones(eval_env, ax):
    """
    Visualize success zones as sphere of radius eps_success around skill-goals
    """
    fake_env = eval_env.env_fns[0]()
    L_states = fake_env.skill_manager.L_states
    i = 0
    for state in L_states:
        goal = fake_env.project_to_goal_space(state)

        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

        x = goal[0] + 0.075*np.cos(u)*np.sin(v)
        y = goal[1] + 0.075*np.sin(u)*np.sin(v)
        z = goal[2] + 0.075*np.cos(v)
        if i ==0:
            ax.plot_wireframe(x, y, z, color="red", alpha = 0.1)
        else:
            ax.plot_wireframe(x, y, z, color="blue", alpha = 0.1)
        i+= 1

    return

# def eval_traj(eval_env, agent):
#     traj = []
#     eval_env.reset()
#     init_indx = torch.ones((eval_env.num_envs,1)).int()
#     observation = eval_env.set_skill(init_indx)
#     eval_done = False
#     #print("observation = ", observation)
#     #print("eval_env.skill_manager.indx_start = ", eval_env.skill_manager.indx_start)
#     #print("eval_env.skill_manager.indx_goal = ", eval_env.skill_manager.indx_goal)
#
#     while eval_env.skill_manager.indx_goal[0] <= eval_env.skill_manager.nb_skills and not eval_done:
#         skill_success = False
#         for i_step in range(0,eval_env.max_episode_steps[0]):
#             #print("eval_env.skill_manager.indx_goal = ", eval_env.skill_manager.indx_goal)
#             traj.append(eval_env.project_to_goal_space(torch.from_numpy(observation["observation"])))
#             action = agent.select_action(hstack(observation["observation"], observation["desired_goal"]),
#             deterministic=True,
#             )
#             observation, _, done, info = eval_env.step(action)
#             if info["is_success"][0]:
#                 skill_success = True
#                 break
#         #print(torch.eq(eval_env.skill_manager.indx_goal.sum(),torch.tensor(eval_env.skill_manager.nb_skills)).numpy())
#         if (torch.eq(eval_env.skill_manager.indx_goal.sum(),torch.tensor(eval_env.skill_manager.nb_skills)).numpy()):
#             eval_done = True
#         observation = eval_env.shift_goal()
#     return traj


if (__name__=='__main__'):

    parser = argparse.ArgumentParser(description='Argument for DCIL')
    parser.add_argument('--demo_path', help='demostration file')
    parsed_args = parser.parse_args()

    env_args = {}
    env_args["demo_path"] = str(parsed_args.demo_path)

    num_envs = 1  # the number of rollouts in parallel during training
    env, eval_env, env_info = gym_vec_env('GFetchDCIL-v0', num_envs, env_args)
    print("env_info = ", env_info)

    batch_size = 256
    gd_steps_per_step = 1
    start_training_after_x_steps = 2_500
    max_steps = 100_000
    evaluate_every_x_steps = 1_000
    save_agent_every_x_steps = 100_000

    now = datetime.now()
    dt_string = '%s_%s' % (datetime.now().strftime('%Y%m%d'), str(os.getpid()))
    #save_dir = os.path.join('/gpfswork/rech/kcr/ubj56je', 'results', 'xpag', 'DCIL_XPAG_FETCH', dt_string)
    save_dir = os.path.join(os.path.expanduser('~'), 'results', 'xpag', 'DCIL_XPAG_FETCH', dt_string)
    os.mkdir(save_dir)
    ## log file for success ratio
    f_ratio = open(save_dir + "/ratio.txt", "w")

    save_episode = True
    plot_projection = None

    agent = SAC(
    env_info['observation_dim'] if not env_info['is_goalenv']
    else env_info['observation_dim'] + env_info['desired_goal_dim'],
    env_info['action_dim'],
    {}
    )
    sampler = DefaultEpisodicSampler() if not env_info['is_goalenv'] else HER_DCIL(env.env_fns[0]().compute_reward, env)
    print("sampler = ", sampler)

    buffer = DefaultEpisodicBuffer(
        max_episode_steps=env_info['max_episode_steps'],
        buffer_size=1_000_000,
        sampler=sampler
    )
    goalsetter = DefaultGoalSetter()

    eval_log_reset()
    timing_reset()
    observation = env.reset()
    trajs = []
    traj = []
    num_rollouts = 0
    num_success = 0
    info_train = None

    for i in range(max_steps // env_info["num_envs"]):
        traj.append(observation["achieved_goal"])
        if not i % max(evaluate_every_x_steps // env_info["num_envs"], 1):
            single_rollout_eval(
                i * env_info["num_envs"],
                eval_env,
                env_info,
                agent,
                save_dir=save_dir,
                plot_projection=plot_projection,
                save_episode=save_episode,
            )
            plot_traj(trajs, eval_env, save_dir, it=i)
            traj = []
            trajs = []

            if info_train is not None:
                print("critic loss: ", info_train["critic_loss"])
            if num_rollouts > 0:
                print("ratio is: ", float(num_success/num_rollouts))
                f_ratio.write(str(float(num_success/num_rollouts)) + "\n")
                num_rollouts = 0
                num_success = 0

            # print("obs_rms['achieved_goal'].mean = ", obs_rms["achieved_goal"].mean )

        if not i % max(save_agent_every_x_steps // env_info["num_envs"], 1):
            if save_dir is not None:
                agent.save(os.path.join(save_dir, "agent"))

        if i * env_info["num_envs"] < start_training_after_x_steps:
            action = env_info["action_space"].sample()
        else:
            if env.do_normalize:
                ## TODO (Alex): add normalization for non-dict observations
                norm_observation = env.normalize(observation)
                action = agent.select_action(
                    norm_observation
                    if not env_info["is_goalenv"]
                    else hstack(env._normalize(observation["observation"], env.obs_rms["observation"]),
                                env._normalize(observation["desired_goal"], env.obs_rms["achieved_goal"])),
                    deterministic=False,
                )
            else:
                action = agent.select_action(
                    observation
                    if not env_info["is_goalenv"]
                    else hstack(observation["observation"], observation["desired_goal"]),
                    deterministic=False,
                )

            for _ in range(max(round(gd_steps_per_step * env_info["num_envs"]), 1)):
                batch = buffer.sample(batch_size)
                info_train = agent.train_on_batch(batch)
                # print("mean achieved_goal TRAIN = ", env.obs_rms["achieved_goal"].mean)


        next_observation, reward, done, info = env.step(action)

        _info = {}
        for key in info[0].keys():
            _info[key] = np.array([info[i][key] for i in range(len(info))]).reshape(env_info["num_envs"],-1)

        step = {
            "observation": observation,
            "action": action.reshape(env_info["num_envs"],-1),
            "reward": reward.reshape(env_info["num_envs"],-1),
            "truncation": _info["truncation"],
            "done": done.reshape(env_info["num_envs"],-1),
            "next_observation": next_observation,
        }

        if env_info["is_goalenv"]:
            step["is_success"] = _info["is_success"]
            step["next_goal"] = _info["next_goal"]
            step["next_goal_avail"] = _info["next_goal_avail"]

        buffer.insert(step)
        observation = next_observation

        if done.max():
            num_rollouts += 1
            if _info["is_success"].max() == 1:
                num_success += 1
            traj.append(observation["achieved_goal"])
            trajs.append(traj)
            traj = []
            # use store_done() if the buffer is an episodic buffer
            if hasattr(buffer, "store_done"):
                buffer.store_done()
            observation = env.reset_done()
            #print("achieved_goal = ", observation["achieved_goal"])
    f_ratio.close()
