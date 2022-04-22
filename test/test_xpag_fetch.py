#!/usr/bin python -w

import os
import jax
import flax

import xpag
from xpag.wrappers import gym_vec_env
from xpag.buffers import DefaultEpisodicBuffer
from xpag.samplers import DefaultEpisodicSampler, HER
from xpag.goalsetters import DefaultGoalSetter
from xpag.agents import SAC, SAC_bonus
from xpag.tools import learn
from xpag.tools import mujoco_notebook_replay

from xpag.tools.eval import single_rollout_eval
from xpag.tools.utils import hstack
from xpag.tools.logging import eval_log_reset
from xpag.tools.timing import timing_reset
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
import torch

import gym_gfetch


def visu_success_zones(eval_env, ax):
    """
    Visualize success zones as sphere of radius eps_success around skill-goals
    """
    L_states = copy.deepcopy(eval_env.skill_manager.L_states)

    for state in L_states:
        goal = eval_env.project_to_goal_space(state)

        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

        x = goal[0] + 0.075*np.cos(u)*np.sin(v)
        y = goal[1] + 0.075*np.sin(u)*np.sin(v)
        z = goal[2] + 0.075*np.cos(v)
        ax.plot_wireframe(x, y, z, color="blue", alpha = 0.1)

    return

def plot_traj(traj, traj_eval, eval_env, save_dir, it=0):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(traj[0].shape[0]):
        X = [goal[i][0] for goal in traj]
        Y = [goal[i][1] for goal in traj]
        Z = [goal[i][2] for goal in traj]
        ax.scatter(X, Y, Z, marker=".")

    X_eval = [goal[0] for goal in eval_traj]
    Y_eval = [goal[1] for goal in eval_traj]
    Z_eval = [goal[2] for goal in eval_traj]
    ax.plot(X_eval, Y_eval, Z_eval, c = "red")

    visu_success_zones(eval_env, ax)

    plt.savefig(save_dir + "/trajs_it_"+str(it)+".png")
    plt.close(fig)
    return

def eval_traj(eval_env, agent):
    traj = []
    eval_env.reset()
    init_indx = torch.ones((eval_env.num_envs,1)).int()
    observation = eval_env.set_skill(init_indx)
    eval_done = False
    #print("observation = ", observation)
    #print("eval_env.skill_manager.indx_start = ", eval_env.skill_manager.indx_start)
    #print("eval_env.skill_manager.indx_goal = ", eval_env.skill_manager.indx_goal)

    while eval_env.skill_manager.indx_goal[0] <= eval_env.skill_manager.nb_skills and not eval_done:
        skill_success = False
        for i_step in range(0,eval_env.max_episode_steps[0]):
            #print("eval_env.skill_manager.indx_goal = ", eval_env.skill_manager.indx_goal)
            traj.append(eval_env.project_to_goal_space(torch.from_numpy(observation["observation"])))
            action = agent.select_action(hstack(observation["observation"], observation["desired_goal"]),
            deterministic=True,
            )
            observation, _, done, info = eval_env.step(action)
            if info["is_success"][0]:
                skill_success = True
                break
        #print(torch.eq(eval_env.skill_manager.indx_goal.sum(),torch.tensor(eval_env.skill_manager.nb_skills)).numpy())
        if (torch.eq(eval_env.skill_manager.indx_goal.sum(),torch.tensor(eval_env.skill_manager.nb_skills)).numpy()):
            eval_done = True
        observation = eval_env.shift_goal()
    return traj

if (__name__=='__main__'):

    num_envs = 1  # the number of rollouts in parallel during training
    # env, eval_env, env_info = gym_vec_env('GFetchDCIL-v0', num_envs)
    env, eval_env, env_info = gym_vec_env('HalfCheetah-v3', num_envs)
    print("env_info = ", env_info)

    batch_size = 256
    gd_steps_per_step = 1.5
    start_training_after_x_steps = env_info['max_episode_steps'] * 10
    max_steps = 100_000
    evaluate_every_x_steps = 5_000
    save_agent_every_x_steps = 100_000
    save_dir = os.path.join(os.path.expanduser('~'), 'results', 'xpag', 'train_mujoco')
    save_episode = True
    plot_projection = None

    agent = SAC_bonus(
        env_info['observation_dim'] if not env_info['is_goalenv']
        else env_info['observation_dim'] + env_info['desired_goal_dim'],
        env_info['action_dim'],
        {}
    )

    print("env.compute_reward = ", env.compute_reward)
    sampler = DefaultEpisodicSampler() if not env_info['is_goalenv'] else HER(env.compute_reward)
    buffer = DefaultEpisodicBuffer(
        max_episode_steps=env_info['max_episode_steps'],
        buffer_size=1_000_000,
        sampler=sampler
    )
    goalsetter = DefaultGoalSetter()

    eval_log_reset()
    timing_reset()
    observation = env.reset()
    traj = []
    info_train = None

    print("observation = ", observation)

    for i in range(max_steps // env_info["num_envs"]):
        traj.append(eval_env.project_to_goal_space(torch.from_numpy(observation["observation"])))

        #print("\nobservation = ", observation["observation"][0])
        #print("desired_goal = ", observation["desired_goal"][0])
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
            traj_eval = eval_traj(eval_env, agent)
            plot_traj(traj, traj_eval, eval_env, save_dir, it=i)
            # visu_value(eval_env, agent, save_dir, it=i)
            traj = []
            if info_train is not None:
                print("rewards = ", max(info_train["rewards"]))

        if not i % max(save_agent_every_x_steps // env_info["num_envs"], 1):
            if save_dir is not None:
                agent.save(os.path.join(save_dir, "agent"))

        if i * env_info["num_envs"] < start_training_after_x_steps:
            action = env_info["action_space"].sample()
        else:
            action = agent.select_action(
                observation
                if not env_info["is_goalenv"]
                else hstack(observation["observation"], observation["desired_goal"]),
                deterministic=False,
            )
            for _ in range(max(round(gd_steps_per_step * env_info["num_envs"]), 1)):
                transitions = buffer.sample(batch_size)
                #print("max reward = ", transitions["reward"].max())
                #print("\ndone = ", transitions["done"][:10])
                #print("truncation = ", transitions["truncation"][:10])
                #print("is_success = ", transitions["is_success"][:10])
                #print("reward = ", transitions["reward"][:10])
                #print("observation = ", transitions["observation"][:10])
                #print("next_observation = ", transitions["next_observation"][:10])

                #print("transitions = ", transitions.keys())
                #print("transitions['next_goal'] = ", transitions['next_goal'][:10])
                #print("transitions['next_goal_avail'] = ", transitions['next_goal_avail'][:10])

                #_ = agent.train_on_batch(transitions)
                info_train = agent.train_on_batch(transitions)
                #print("\nis_success = ", info_train["is_success"])
                #print("is_not_relabelled = ", info_train["is_not_relabelled"])
                #print("next_goal_avail = ", info_train["next_goal_avail"])
                #print("add_bonus = ", info_train["add_bonus"])
                #print("rewards = ", max(info_train["rewards"]))
                #print("batch.rewards = ", info_train["batch.rewards"].shape)
                #print("next_goal_q = ", info_train["next_goal_q"].shape)


            #print("agent.critic = ", agent.sac.critic.params)

        next_observation, reward, done, info = env.step(action)
        #print("next_observation['desired_goal'] = ", next_observation['desired_goal'])
        #print("\nnext_observation = ", next_observation)
        #print("one = ", done)
        #print("buffer obs = ", buffer.buffers["observation.observation"][buffer.current_idxs, buffer.current_t-1])
        #print("buffer next obs = ", buffer.buffers["next_observation.observation"][buffer.current_idxs, buffer.current_t-1])
        #print("buffer done = ", buffer.buffers["done"][buffer.current_idxs, buffer.current_t-1])

        step = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "truncation": info["truncation"],
            "done": done,
            "next_observation": next_observation,
        }
        if env_info["is_goalenv"]:
            step["is_success"] = info["is_success"]
            step["next_goal"] = info["next_goal"]
            step["next_goal_avail"] = info["next_goal_avail"]
        buffer.insert(step)
        #print("buffer obs = ", buffer.buffers["observation.observation"][buffer.current_idxs, buffer.current_t-1])
        #print("buffer next obs = ", buffer.buffers["next_observation.observation"][buffer.current_idxs, buffer.current_t-1])
        #print("buffer done = ", buffer.buffers["done"][buffer.current_idxs, buffer.current_t-1])
        observation = next_observation

        if done.max():
            # use store_done() if the buffer is an episodic buffer
            if hasattr(buffer, "store_done"):
                buffer.store_done()
            observation = env.reset_done()
