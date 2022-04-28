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

import gym_gmazes

num_envs = 5  # the number of rollouts in parallel during training
env, eval_env, env_info = gym_vec_env('GMazeDCILDubins-v0', num_envs)

def plot_traj(traj, traj_eval, save_dir, it=0):
    fig, ax = plt.subplots()
    env.plot(ax)
    for i in range(traj[0].shape[0]):
        X = [state[i][0] for state in traj]
        Y = [state[i][1] for state in traj]
        Theta = [state[i][2] for state in traj]
        ax.scatter(X,Y, marker=".")

        for x, y, t in zip(X,Y,Theta):
            dx = np.cos(t)
            dy = np.sin(t)
            #arrow = plt.arrow(x,y,dx*0.1,dy*0.1,alpha = 0.6,width = 0.01, zorder=6)

    X_eval = [state[0][0] for state in traj_eval]
    Y_eval = [state[0][1] for state in traj_eval]
    ax.plot(X_eval, Y_eval, c = "red")

    circles = []
    for state in env.skill_manager.L_states:
    #for state in [[[ 0.33      ,  0.5       , -0.17363015]], [[1.78794995, 1.23542976]]]:
    #for state in [[[1.78794995, 1.23542976]]]:
        circle = plt.Circle((state[0][0], state[0][1]), 0.1, color='m', alpha = 0.6)
        circles.append(circle)
        # ax.add_patch(circle)
    coll = mc.PatchCollection(circles, color="plum", zorder = 4)
    ax.add_collection(coll)

    plt.savefig(save_dir + "/trajs_it_"+str(it)+".png")
    plt.close(fig)
    return

import torch
@torch.no_grad()
def visu_value(eval_env, agent, save_dir, it=0):

    thetas = np.linspace(-torch.pi/2.,torch.pi/2.,100)

    values = []
    obs = eval_env.reset()
    #obs["observation"][0] = torch.tensor([ 0.33      ,  0.5       , -0.17363015])
    obs["observation"][0] = env.skill_manager.L_states[0][0]
    obs["desired_goal"][0] = env.skill_manager.L_states[1][0][:2]
    for theta in list(thetas):
        obs["observation"][0][2] = theta
        #print("obs = ", obs["observation"])
        #print("dg = ", obs["desired_goal"])
        #print("stack = ", hstack(obs["observation"], obs["desired_goal"]))
        action = agent.select_action(hstack(obs["observation"], obs["desired_goal"]),
            deterministic=True,
        )
        #print('obs = ', obs["observation"])
        value_q1, value_q2 = agent.sac.target_critic(hstack(obs["observation"], obs["desired_goal"]), action)
        value = np.minimum(np.array(value_q1)[0], np.array(value_q2)[0])
        values.append(value)

    fig, ax = plt.subplots()
    plt.plot(list(thetas), values,label="learned V(s,g')")
    plt.plot()
    plt.xlabel("theta")
    plt.ylabel("value")
    plt.legend()
    plt.savefig(save_dir + "/value_skill_1_it_"+str(it)+".png")
    plt.close(fig)

    values = []
    #thetas = np.linspace(0.,torch.pi,100)
    thetas = np.linspace(-torch.pi/2.,torch.pi/2.,100)
    obs = eval_env.reset()
    obs["observation"][0] = env.skill_manager.L_states[1][0]
    obs["desired_goal"][0] = env.skill_manager.L_states[2][0][:2]
    for theta in list(thetas):
        obs["observation"][0][2] = theta
        #print("obs = ", obs["observation"])
        #print("dg = ", obs["desired_goal"])
        #print("stack = ", hstack(obs["observation"], obs["desired_goal"]))
        action = agent.select_action(hstack(obs["observation"], obs["desired_goal"]),
            deterministic=True,
        )
        #print('obs = ', obs["observation"])
        value_q1, value_q2 = agent.sac.target_critic(hstack(obs["observation"], obs["desired_goal"]), action)
        value = np.minimum(np.array(value_q1)[0], np.array(value_q2)[0])
        values.append(value)

    fig, ax = plt.subplots()
    plt.plot(list(thetas), values,label="learned V(s,g')")
    plt.plot()
    plt.xlabel("theta")
    plt.ylabel("value")
    plt.legend()
    plt.savefig(save_dir + "/value_skill_2_it_"+str(it)+".png")
    plt.close(fig)

    return values

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
            traj.append(observation["observation"])
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

batch_size = 256
gd_steps_per_step = 1.5
start_training_after_x_steps = env_info['max_episode_steps'] * 50
max_steps = 100_000
evaluate_every_x_steps = 5_000
save_agent_every_x_steps = 100_000
save_dir = os.path.join(os.path.expanduser('~'), 'results', 'xpag', 'train_mujoco')
save_episode = True
plot_projection = None

agent = SAC(
    env_info['observation_dim'] if not env_info['is_goalenv']
    else env_info['observation_dim'] + env_info['desired_goal_dim'],
    env_info['action_dim'],
    {}
)
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

for i in range(max_steps // env_info["num_envs"]):
    traj.append(observation["observation"])

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
        plot_traj(traj, traj_eval, save_dir, it=i)
        visu_value(eval_env, agent, save_dir, it=i)
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
            # print("\ndone = ", transitions["done"][:10])
            # print("truncation = ", transitions["truncation"][:10])
            # print("is_success = ", transitions["is_success"][:10])
            # print("reward = ", transitions["reward"][:10])
            #print("observation = ", transitions["observation"][:10])
            #print("next_observation = ", transitions["next_observation"][:10])

            #print("transitions = ", transitions.keys())
            #print("transitions['next_goal'] = ", transitions['next_goal'][:10])
            #print("transitions['next_goal_avail'] = ", transitions['next_goal_avail'][:10])

            #_ = agent.train_on_batch(transitions)
            info_train = agent.train_on_batch(transitions)
            # print("target_q = ", info_train["target_q"].max())
            # print("q1 = ", info_train["q1"].max())
            # print("q2 = ", info_train["q2"].max())
            if (info_train["target_q"].max() > 2.) or (info_train["q1"].max() > 2.) or (info_train["q2"].max() > 2.):
                print("\n error: excessive target value")
                print("next_q = ", info_train["next_q"])
                print("q1 = ", info_train["q1"])
                print("q2 = ", info_train["q2"])
                print("target_q = ", info_train["target_q"])
                print("rewards = ", info_train["rewards"])
                print("masks = ", info_train["masks"])
                print("done = ", transition["done"][:5])
                print("truncation = ", transition["truncation"][:5])
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
