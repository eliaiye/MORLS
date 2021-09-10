import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from MORLS.model.ActorCriticWithLayers import ActorCriticPolicy
from MORLS.policy.PPO_XWithLayers import PPO_X2
from MORLS.env.SimEnv import SimEnv

def eval_policy(env, policy, n_ep=5, timeout=1024):
    for i_episode in range(n_ep):
        ep_reward_ls = []
        ep_len_ls=[]
        ep_reward = 0
        ep_len=0
        observation =env.reset()
        for t in range(timeout):
            with torch.set_grad_enabled(False):
                observation = torch.unsqueeze(torch.tensor(observation).float(),0)
                action, _=policy.policy.predict(observation)
                #except: action, _, __ = policy.forward(observation)
                action = action[0]
                #print("action:", action, type(action))
                observation, reward, done, info = env.step(action)
                #print("reward:", reward)
                ep_reward += reward
                ep_len +=1

                if done or t == timeout-1:
                    #print("Episode finished after {} timesteps".format(t+1))
                    #print("Survival time: {}".format(ep_reward+env.penalty*(t+1)))
                    break

        ep_reward_ls.append(ep_reward) 
        ep_len_ls.append(ep_len)

    return(np.mean(ep_reward_ls), np.mean(ep_len_ls))


def main():
    env =SimEnv()
    policy_sb= PPO_X2("MlpPolicy", env, n_steps=100, batch_size=25, verbose=1, ent_coef=0.0, vf_coef=0.0, seed=42)
    total_ep_len=[]
    total_ep_rew=[]
    for i in tqdm(range(20000//100)):  
        policy_sb.learn(total_timesteps=100, eval_freq=10000, log_interval=10000)
        #policy_mujoco_sb.learn(total_timesteps=1024)
        r, l = eval_policy(env, policy_sb, n_ep=5, timeout=100)
        total_ep_rew.append(r)
        total_ep_len.append(l)

    plt.figure()
    plt.plot(total_ep_rew)
    plt.show()

if __name__ == "__main__": main() 