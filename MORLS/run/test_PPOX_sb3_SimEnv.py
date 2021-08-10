import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from MORLS.model.ActorCriticwithentropy import ActorCriticPolicy
from MORLS.policy.PPO_X_sb3 import PPO_X2
from MORLS.env.SimEnv import SimEnv
from stable_baselines3 import PPO

def main():
	env =SimEnv()
	policy=PPO_X2(1,2)
	policy.train(env=env, total_timesteps=20000, timesteps=100, k_epoch=10,max_grad_norm = None) # 100 iterations, 1 actor, 10 epochs, batch size=100/4=25

if __name__ == "__main__": main() 