# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

from abc import ABC
import numpy as np
import os
from xpag.agents.agent import Agent
from xpag.agents.sac.sac_from_jaxrl import Batch, SACLearner
from xpag.tools.utils import squeeze


class SAC(Agent, ABC):
    def __init__(self, observation_dim, action_dim, params=None):
        """
        Interface to the SAC agent from JAXRL (https://github.com/ikostrikov/jaxrl)
        """

        self._config_string = str(list(locals().items())[1:])
        super().__init__("SAC", observation_dim, action_dim, params)

        if "seed" in self.params:
            start_seed = self.params["seed"]
        else:
            start_seed = 42

        self.jaxrl_params = {
            "actor_lr": 0.001,
            "backup_entropy": True,
            "critic_lr": 0.001,
            "discount": 0.99,
            # "hidden_dims": (512, 512, 512),
            "hidden_dims": (400,300),
            "init_temperature": 0.01,
            "target_entropy": None,
            "target_update_period": 1,
            "tau": 0.005,
            "temp_lr": 0.0003,
        }

        for key in self.jaxrl_params:
            if key in self.params:
                self.jaxrl_params[key] = self.params[key]

        self.sac = SACLearner(
            start_seed,
            np.zeros((1, 1, observation_dim)),
            np.zeros((1, 1, action_dim)),
            **self.jaxrl_params
        )

    def select_action(self, observation, deterministic=True):
        # return self.sac.sample_actions(observation)
        return self.sac.sample_actions(
            observation, distribution="det" if deterministic else "log_prob"
        )

    def train_on_batch(self, batch):
        jaxrl_batch = Batch(
            observations=batch["observation"],
            actions=batch["action"],
            rewards=squeeze(batch["reward"]),
            masks=squeeze(1 - batch["true_done"] * (1 - batch["true_truncation"])),
            next_observations=batch["next_observation"],
        )

        return self.sac.update(jaxrl_batch)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "step.npy"), self.sac.step)
        self.sac.actor.save(os.path.join(directory, "actor"))
        self.sac.critic.save(os.path.join(directory, "critic"))
        self.sac.target_critic.save(os.path.join(directory, "target_critic"))
        self.sac.temp.save(os.path.join(directory, "temp"))

    def load(self, directory):
        self.sac.step = np.load(os.path.join(directory, "step.npy")).item()
        self.sac.actor = self.sac.actor.load(os.path.join(directory, "actor"))
        self.sac.critic = self.sac.critic.load(os.path.join(directory, "critic"))
        self.sac.target_critic = self.sac.target_critic.load(
            os.path.join(directory, "target_critic")
        )
        self.sac.temp = self.sac.temp.load(os.path.join(directory, "temp"))

    def write_config(self, output_file: str):
        print(self._config_string, file=output_file)
