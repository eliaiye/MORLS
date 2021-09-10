import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gym
import gym.spaces as spaces
from stable_baselines3_X import PPO
import random

import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from collections import deque 
from functools import partial

from MORLS.model.ActorCriticWithLayers import ActorCriticPolicy
from stable_baselines3_X.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3_X.common.buffers import RolloutBuffer

class PPO_X2(OnPolicyAlgorithm):
    def __init__(self, policy, env, learning_rate = 0.0003, n_steps =2048, batch_size = 25, n_epochs=10, gamma = 0.99, 
                 gae_lambda = 0.95, clip_range = 0.2, clip_range_vf = None, ent_coef = 0.0, vf_coef = 0.5,
                 max_grad_norm = 0.5, use_sde=False, sde_sample_freq= -1, target_kl = None, tensorboard_log = None,
                 create_eval_env = False, policy_kwargs = None, verbose = 0, seed = 123, device = "auto", _init_setup_model = True):
        super(PPO_X2, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                gym.spaces.Box,
                gym.spaces.Discrete,
                gym.spaces.MultiDiscrete,
                gym.spaces.MultiBinary
            )
        )
#         assert (
#             batch_size > 1
#         ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.n_steps
            assert (
                buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        #super(PPO_X2, self)._setup_model()
        self.set_random_seed(self.seed)
        self.rollout_buffer = RolloutBuffer(self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = ActorCriticPolicy(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
#             use_sde=self.use_sde,
#             **self.policy_kwargs  # pytype:disable=not-instantiable
        )
#         self.policy = ActorCriticPolicy(  # pytype:disable=not-instantiable
#             1,
#             2)
        # Initialize schedules for policy/value clipping
        #self.clip_range = get_schedule_fn(self.clip_range)
#         if self.clip_range_vf is not None:
#             if isinstance(self.clip_range_vf, (float, int)):
#                 assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

#             self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.train(True)      
        # Update optimizer learning rate
        #self._update_learning_rate(self.policy.optimizer)
        
        # Compute current clip range
        #clip_range = self.clip_range(self._current_progress_remaining)
        clip_range = self.clip_range
        # Optional: clip range for the value function
#         if self.clip_range_vf is not None:
#             clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
#             approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                # TODO: investigate why there is no issue with the gradient
                # if that line is commented (as in SAC)
#                 if self.use_sde:
#                     self.policy.reset_noise(self.batch_size)
                    

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
        
                #values, log_prob, entropy = self.policy.forward(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
#                 else:
#                     # Clip the different between old and new value
#                     # NOTE: this depends on the reward scaling
#                     values_pred = rollout_data.old_values + th.clamp(
#                         values - rollout_data.old_values, -clip_range_vf, clip_range_vf
#                     )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    #entropy_loss = -th.mean(entropy)
                    entropy_loss = -th.mean(-log_prob)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
#                     approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
#                     approx_kl_divs.append(approx_kl_div)

#                 if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
#                     continue_training = False
#                     if self.verbose >= 1:
#                         print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
#                     break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
#        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

#         # Logs
#         self.logger.record("train/entropy_loss", np.mean(entropy_losses))
#         self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
#         self.logger.record("train/value_loss", np.mean(value_losses))
#         self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
#         self.logger.record("train/clip_fraction", np.mean(clip_fractions))
#         self.logger.record("train/loss", loss.item())
#         self.logger.record("train/explained_variance", explained_var)
#         if hasattr(self.policy, "log_std"):
#             self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

#         self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
#         self.logger.record("train/clip_range", clip_range)
#         if self.clip_range_vf is not None:
#             self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self,
        total_timesteps: int,
        callback = None,
        log_interval: int = 1,
        eval_env = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PPO",
        eval_log_path = None,
        reset_num_timesteps: bool = True,
    ):

        total_timesteps = self._setup_learn(
            total_timesteps, reset_num_timesteps)

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            #iteration += 1
            self._current_progress_remaining = 1.0 - float(self.num_timesteps) / float(total_timesteps)

            self.train()

        return self
    
    def _setup_learn(
        self,
        total_timesteps: int,
        reset_num_timesteps = True,
    ):
        """
        Initialize different variables needed for training.
        """
        #self.start_time = time.time()

#         if self.ep_info_buffer is None or reset_num_timesteps:
#             # Initialize buffers if they don't exist, or reinitialize if resetting counters
#             self.ep_info_buffer = deque(maxlen=100)
#             self.ep_success_buffer = deque(maxlen=100)

#         if self.action_noise is not None:
#             self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            self._last_obs = self.env.reset()  # pytype: disable=annotation-type-mismatch
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

#         if eval_env is not None and self.seed is not None:
#             eval_env.seed(self.seed)

#         eval_env = self._get_eval_env(eval_env)

        # Configure logger's outputs if no logger was passed
#         if not self._custom_logger:
#             self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        #callback = self._init_callback(callback, eval_env, eval_freq, n_eval_episodes, log_path)

        #return total_timesteps, callback
        return total_timesteps