# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

import numpy as np
import torch
from xpag.tools.utils import DataType
from xpag.samplers.sampler import Sampler

# class MyHER(HER):
#     ## TODO: package hors XPAG

class HER(Sampler):
    def __init__(
        self,
        compute_reward,
        replay_strategy: str = "future",
        datatype: DataType = DataType.NUMPY,
    ):
        super().__init__()
        self.datatype = datatype
        self.replay_strategy = replay_strategy
        self.replay_k = 2.0
        if self.replay_strategy == "future":
            self.future_p = 1 - (1.0 / (1 + self.replay_k))
        else:
            self.future_p = 0
        self.reward_func = compute_reward

    def sample(self, buffers, batch_size_in_transitions):
        rollout_batch_size = buffers["episode_length"].shape[0]
        batch_size = batch_size_in_transitions
        # select rollouts and steps
        episode_idxs = np.random.choice(
            np.arange(rollout_batch_size),
            size=batch_size,
            replace=True,
            p=buffers["episode_length"][:, 0, 0]
            / buffers["episode_length"][:, 0, 0].sum(),
        )
        t_max_episodes = buffers["episode_length"][episode_idxs, 0].flatten()

        if self.datatype == DataType.TORCH_CPU or self.datatype == DataType.TORCH_CUDA:
            t_samples = (torch.rand_like(t_max_episodes) * t_max_episodes).long()
        else:
            t_samples = np.random.randint(t_max_episodes)
        transitions = {
            key: buffers[key][episode_idxs, t_samples] for key in buffers.keys()
        }
        # HER indexes
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)

        if self.datatype == DataType.TORCH_CPU or self.datatype == DataType.TORCH_CUDA:
            future_offset = (
                torch.rand_like(t_max_episodes) * (t_max_episodes - t_samples)
            ).long()
        else:
            future_offset = np.random.uniform(size=batch_size) * (
                t_max_episodes - t_samples
            )
            future_offset = future_offset.astype(int)
        future_t = (t_samples + future_offset)[her_indexes]
        # replace desired goal with achieved goal
        future_ag = buffers["next_observation.achieved_goal"][
            episode_idxs[her_indexes], future_t
        ]

        obs_dg = transitions["observation.desired_goal"].copy()
        transitions["observation.desired_goal"][her_indexes] = future_ag
        dist_future_ag_to_true_dg = np.linalg.norm(transitions["observation.desired_goal"] - obs_dg, axis=-1)
        is_close_to_true_dg = np.where(dist_future_ag_to_true_dg<=0.05, 1, 0)

        # recomputing rewards
        if self.datatype == DataType.TORCH_CPU or self.datatype == DataType.TORCH_CUDA:
            transitions["reward"] = torch.unsqueeze(
                self.reward_func(
                    transitions["next_observation.achieved_goal"],
                    transitions["observation.desired_goal"],
                    {},
                ),
                -1,
            )
        else:
            transitions["reward"] = np.expand_dims(
                self.reward_func(
                    transitions["next_observation.achieved_goal"],
                    transitions["observation.desired_goal"],
                    {},
                ),
                1,
            )
        transitions = {
            k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
            for k in transitions.keys()
        }
        if self.datatype == DataType.TORCH_CPU or self.datatype == DataType.TORCH_CUDA:
            transitions["observation"] = torch.hstack(
                (
                    transitions["observation.observation"],
                    transitions["observation.desired_goal"],
                )
            )
            transitions["next_observation"] = torch.hstack(
                (
                    transitions["next_observation.observation"],
                    transitions["observation.desired_goal"],
                )
            )
        else:
            transitions["observation"] = np.concatenate(
                [
                    transitions["observation.observation"],
                    transitions["observation.desired_goal"],
                ],
                axis=1,
            )
            transitions["next_observation"] = np.concatenate(
                [
                    transitions["next_observation.observation"],
                    transitions["observation.desired_goal"],
                ],
                axis=1,
            )



        #is_not_relabelled = np.ones(transitions["done"].shape)
        #is_not_relabelled[her_indexes] = 0.
        #is_not_relabelled = is_not_relabelled + is_close_to_true_dg
        transitions["is_not_relabelled"] = is_close_to_true_dg.reshape(transitions["done"].shape)

        ## add done to transitions relabelled as successful
        diff_reward_done = transitions["done"] - transitions["reward"]/1.
        missing_dones = (diff_reward_done < 0)#.int()
        transitions["true_done"] = transitions["done"].copy() + missing_dones
        assert (transitions["true_done"][:] <= 1).all()
        assert (transitions["true_done"][:] >= 0).all()

        ## remove truncation signal to transitions relabelled as successful -> we want the mask to be zero in that case
        sum_truncation_reward = transitions["truncation"] + transitions["reward"]/1.
        remove_truncation = (sum_truncation_reward == 2) ## truncation true and R = 1
        transitions["true_truncation"] = transitions["truncation"].copy() - remove_truncation
        assert (transitions["true_truncation"][:] <= 1).all()
        assert (transitions["true_truncation"][:] >= 0).all()

        return transitions
