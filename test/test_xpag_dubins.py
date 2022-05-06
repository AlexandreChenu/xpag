#!/usr/bin python -w

import os

#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import jax

print(jax.lib.xla_bridge.get_backend().platform)

#jax.config.update('jax_platform_name', "cpu")

from datetime import datetime
import argparse

import flax

import xpag
from xpag.wrappers import gym_vec_env
from xpag.buffers import DefaultEpisodicBuffer
from xpag.samplers import DefaultEpisodicSampler, HER, HER_DCIL
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
import copy

import gym_gmazes



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

if (__name__=='__main__'):

    parser = argparse.ArgumentParser(description='Argument for DCIL')
    parser.add_argument('--demo_path', help='demostration file')
    parsed_args = parser.parse_args()

    env_args = {}
    env_args["demo_path"] = str(parsed_args.demo_path)

    num_envs = 1  # the number of rollouts in parallel during training
    env, eval_env, env_info = gym_vec_env('GMazeDCILDubins-v0', num_envs, env_args)

    now = datetime.now()
    dt_string = '%s_%s' % (datetime.now().strftime('%Y%m%d'), str(os.getpid()))

    batch_size = 256
    gd_steps_per_step = 1.5
    start_training_after_x_steps = env_info['max_episode_steps'] * 50
    max_steps = 30_000
    evaluate_every_x_steps = 5_000
    save_agent_every_x_steps = 100_000

    #save_dir = os.path.join('/gpfswork/rech/kcr/ubj56je', 'results', 'xpag', 'DCIL_no_bonus_overshoot', dt_string)
    save_dir = os.path.join(os.path.expanduser('~'), 'results', 'xpag', 'DCIL_XPAG', dt_string)
    os.mkdir(save_dir)

    save_episode = True
    plot_projection = None



    agent = SAC(
        env_info['observation_dim'] if not env_info['is_goalenv']
        else env_info['observation_dim'] + env_info['desired_goal_dim'],
        env_info['action_dim'],
        {}
    )
    sampler = DefaultEpisodicSampler() if not env_info['is_goalenv'] else HER_DCIL(env.compute_reward, env)
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
    num_success = 0
    num_rollouts = 0
    f_ratio = open(save_dir + "/ratio.txt", "w")

    for i in range(max_steps // env_info["num_envs"]):
        traj.append(observation["observation"])

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

            if num_rollouts > 0:
                print("ratio = ", float(num_success/num_rollouts))
                f_ratio.write(str(float(num_success/num_rollouts)) + "\n")
                num_success = 0
                num_rollouts = 0

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
                info_train = agent.train_on_batch(transitions)

        next_observation, reward, done, info = env.step(action)

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
            step["next_goal"] = info["next_goal"].reshape(observation["desired_goal"].shape)
            step["next_goal_avail"] = info["next_goal_avail"].reshape(info["is_success"].shape)
        buffer.insert(step)

        observation = next_observation

        if done.max():
            num_rollouts += 1
            if info["is_success"].max() == 1:
                num_success += 1
            # use store_done() if the buffer is an episodic buffer
            if hasattr(buffer, "store_done"):
                buffer.store_done()
            observation = env.reset_done()

    f_ratio.close()
