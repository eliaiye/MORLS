import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import mujoco_py

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from MORLS.model.ActorCriticwithentropy import ActorCriticPolicy
from MORLS.policy.PPO_X_sb3 import PPO_X2

def main():
	env_mujoco =gym.make('HalfCheetah-v2')
	policy_mujoco = PPO_X2(17, 6)
	policy_mujoco.train(env=env_mujoco, total_timesteps=100000, timesteps=1024, timeout=1024, max_grad_norm = None) # 97 iterations, 1 actor, 4 epochs, batch size=1024/4=256
if __name__ == "__main__": main() 