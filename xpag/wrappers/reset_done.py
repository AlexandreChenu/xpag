# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

from brax import jumpy as jp
from brax.envs import env as brax_env
import gym


class ResetDoneWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._last_done = True
        self._last_obs = None
        self.steps = 0

    def reset(self, **kwargs):
        # we assume the reset returns only an obs, no info (return_info=False)
        obs = self.env.reset(**kwargs)
        self._last_done = False
        self._last_obs = obs
        self.steps = 0
        return obs

    def reset_done(self, **kwargs):
        if self._last_done:
            # we assume the reset returns only an obs, no info (return_info=False)
            # obs = self.env.reset(**kwargs)
            obs = self.env.reset_done(**kwargs) ## Note (Alex): change for reset_done
            self._last_done = False
            self._last_obs = obs
            self.steps = 0
        return self._last_obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # print("self.env = ", self.env.unwrapped)
        # print("done reset done = ", done)
        if done:
            self._last_done = True
        # print("self._last_done = ", self._last_done)
        self._last_obs = obs
        self.steps += 1
        info["steps"] = self.steps
        return obs, reward, done, info


class ResetDoneBraxWrapper(brax_env.Wrapper):
    """Adds a reset_done() function to Brax envs."""

    def reset(self, rng: jp.ndarray) -> brax_env.State:
        state = self.env.reset(rng)
        state.info["first_qp"] = state.qp
        state.info["first_obs"] = state.obs
        return state

    def step(self, state: brax_env.State, action: jp.ndarray) -> brax_env.State:
        return self.env.step(state, action)

    def reset_done(self, state: brax_env.State, rng: jp.ndarray):
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(
                    done, [x.shape[0]] + [1] * (len(x.shape) - 1)
                )  # type: ignore
            return jp.where(done, x, y)

        reset_state = self.env.reset(rng)
        qp = jp.tree_map(where_done, reset_state.qp, state.qp)
        obs = where_done(reset_state.obs, state.obs)
        # qp = jp.tree_map(where_done, state.info["first_qp"], state.qp)
        # obs = where_done(state.info["first_obs"], state.obs)
        state = state.replace(qp=qp, obs=obs)
        return state.replace(done=jp.zeros_like(state.done))
