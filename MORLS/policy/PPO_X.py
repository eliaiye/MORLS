import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from MORLS.model.ActorCritic import ActorCriticPolicy

class PPO_X():
    def __init__(self, input_dim, output_dim, network=ActorCriticPolicy):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.policy = network(input_dim, output_dim)
        
        self.total_obs = []
        self.total_act = []
        self.total_rew = []
        self.total_ep_rew=[]
        self.total_ep_len=[]
        
    def forward(self, observation, action = None):
        return self.policy.forward(observation, action=action)
    
    def rollout(self, env, timesteps, gamma, lamb): # timesteps << episode length
        '''
        Run policy pi_old in environment for T timesteps
        Compute advantage estimates A_1,...,A_T
        '''
        obs_trajectory = []
        action_trajectory = []
        logprob_trajectory = []
        value_trajectory = []
        reward_trajectory = []
        done_trajectory = []
        
        with torch.set_grad_enabled(False):
            done = True
            for i in range(timesteps):
                if done:
                    obs = torch.unsqueeze(torch.tensor(env.reset()).float(),0) # numpy array shape (obs_dim, ) 
                                                                       # -> tensor.size([obs_dim])
                                                                       # -> tensor.size([1, obs_dim])
                    done = False

                action, logprob, value = self.forward(observation = obs) # action: tensor.size([1, act_dim]), logprob: tensor.size([1]), value: tensor.size([1])
                
                
#                 clipped_action=action
#                 #clip action
#                 if isinstance(env.action_space, gym.spaces.Box):   
#                     clipped_action = np.clip(clipped_action, env.action_space.low, env.action_space.high)


                action = torch.squeeze(action) # tensor([act_dim])
                logprob = torch.squeeze(logprob) # tensor(1)
                value = torch.squeeze(value) # tensor(1)

                action_trajectory.append(action)
                logprob_trajectory.append(logprob)
                value_trajectory.append(value)
                obs_trajectory.append(torch.squeeze(obs))
                done_trajectory.append(done)

                obs, reward, done, _ = env.step(action.numpy().astype(float)) # step() takes np array shape(act_dim,)
                #obs, reward, done, _ = env.step(torch.squeeze(clipped_action).numpy().astype(float))# step() takes np array shape(act_dim,)
                
                obs = torch.unsqueeze(torch.tensor(obs).float(),0)
                reward = torch.tensor(reward).float() # tensor.size([])
                
                
                
                
                #print("reward", reward, type(reward), "done", done, type(done))
                
                

                reward_trajectory.append(reward)
            # All trajectories are lists of squeezed tensors, done_trajectory is a list of bool
            # List->stacked tensor
            obs_trajectory = torch.stack(obs_trajectory) # tensor.size([#obs, obs_dim])
            action_trajectory = torch.stack(action_trajectory)
            logprob_trajectory = torch.stack(logprob_trajectory)# tensor.size([#obs])
            value_trajectory = torch.stack(value_trajectory)
            reward_trajectory = torch.stack(reward_trajectory)  

            done_trajectory = np.asarray(done_trajectory, dtype=np.bool_)

            _, __, last_value = self.forward(obs)

            adv_trajectory = torch.zeros_like(reward_trajectory).float()
            delta_trajectory = torch.zeros_like(reward_trajectory).float()
            last_gae_lam = torch.Tensor([0.0]).float()

            for t in reversed(range(timesteps)): #T-1 -> 0
                if t == (timesteps -1):
                    next_non_terminal = 1.0-done
                    next_values = last_value
                else:
                    next_non_terminal = 1.0 - done_trajectory[t+1]
                    next_values = value_trajectory[t+1]
                delta = reward_trajectory[t]+gamma*next_values*next_non_terminal-value_trajectory[t]
                adv_trajectory[t] = last_gae_lam = delta + gamma*lamb*next_non_terminal*last_gae_lam
#             for t in reversed(range(timesteps)): #T-1 -> 0

#                 if done_trajectory[t]:
#                     delta_trajectory[t] = reward_trajectory[t]-value_trajectory[t]
#                     adv_trajectory[t]=delta_trajectory[t]

#                 else:
#                     if t == timesteps-1:
#                         delta_trajectory[t] = reward_trajectory[t]-value_trajectory[t]
#                         adv_trajectory[t]=delta_trajectory[t]
#                     else:
#                         delta_trajectory[t] = reward_trajectory[t]+gamma*value_trajectory[t+1]-value_trajectory[t]
#                         adv_trajectory[t]=delta_trajectory[t]+gamma*lamb*adv_trajectory[t+1]

    #             if t == timesteps-1:
    #                 adv_trajectory[t]=delta_trajectory[t]


    #             else:
    #                 adv_trajectory[t]=delta_trajectory[t]+gamma*lamb*adv_trajectory[t+1] #recursion
            #print("in rollout:",obs_trajectory.size())
        return(obs_trajectory, action_trajectory, logprob_trajectory, adv_trajectory)
    
    def loss(self, epsilon, obs_traj, action_traj, logprob_traj, adv_traj, current_logprob):
        # ratio
        ratio = torch.exp(current_logprob-logprob_traj) # torch.size([#obs])
        
        adv_traj = (adv_traj - adv_traj.mean()) / (adv_traj.std() + 1e-8)
        # loss
        l1 = ratio*adv_traj # torch.size([#obs])
        l2 = torch.clamp(ratio, 1.-epsilon, 1.+epsilon)*adv_traj # torch.size([#obs])
        # Minimize negative L
        PGloss = -torch.mean(torch.min(l1, l2))# torch.size([#obs])
            
        return PGloss # torch.size([])
    
    def train_step(self, epsilon, obs_traj, action_traj, logprob_traj, adv_traj):
        
        #Question: adv needs normalization?
        
        self.policy.train() # module train mode
        with torch.set_grad_enabled(True):
            action, logprob, value = self.forward(obs_traj, action=action_traj)
            
            PGloss = self.loss(epsilon, obs_traj, action_traj, logprob_traj, adv_traj,logprob)
            
            PGloss.backward()
            
            return(PGloss)
    
    def train(self, env, total_timesteps, timesteps=1024, k_epoch=10, num_batch=4, n_actor=1, 
              epsilon=0.2, gamma = 0.99, lamb=0.95, lr = 0.0003, max_grad_norm = 0.5, optimizer = torch.optim.Adam, 
              evaluate=True, eval_env=None, n_ep=5, timeout=None, plot=True):
        
        self.env = env
        self.eval_env = eval_env
        self.timesteps =timesteps
        n_updates = total_timesteps//timesteps # number of iterations
        
        self.policy_optimizer=optimizer(self.policy.parameters(), lr=lr)

        for i in tqdm(range(n_updates)):
            actor=0
            obs_trajectory, action_trajectory, logprob_trajectory, adv_trajectory = self.rollout(env, timesteps, gamma, lamb)
            for actor in range(n_actor-1):
                more_obs_traj, more_action_traj, more_logprob_traj, more_adv_traj = self.rollout(env, timesteps, gamma, lamb)
                obs_trajectory = torch.cat((obs_trajectory, more_obs_traj),0)
                action_trajectory = torch.cat((action_trajectory, more_action_traj),0)
                logprob_trajectory = torch.cat((logprob_trajectory, more_logprob_traj),0)
                adv_trajectory = torch.cat((adv_trajectory, more_adv_traj),0)
            
            indices = np.arange(n_actor*timesteps) # [0, 1, ..., n_actor*timesteps-1]
            
            for k in range(k_epoch):
                np.random.shuffle(indices)
                batch_size = n_actor*timesteps//num_batch
                
                if(timesteps%num_batch):
                    batch_size += 1
                
                for b in range(num_batch):
                    #reset gradient
                    self.policy_optimizer.zero_grad()
                    
                    if b != num_batch-1: # all batches except the last one
                        batch_indices = indices[b*batch_size:(b+1)*batch_size]
                    else:
                        batch_indices = indices[b*batch_size:] # last batch
                
                    batch=[tensor[batch_indices] for tensor in (torch.unsqueeze(obs_trajectory,1), action_trajectory, logprob_trajectory, adv_trajectory)]
                    #print("in train", batch[0])
                    PGloss = self.train_step(epsilon, *batch) 
                    
                    #Question: Do we need to clip gradient?
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                    
                    self.policy_optimizer.step()
            if evaluate:
                self.eval_policy(env=eval_env, n_ep=n_ep, timeout=timeout)
        if plot:
            plt.figure()
            plt.plot(self.total_ep_rew)
            plt.show()
                    
    def eval_policy(self, env, n_ep=5, timeout=None):
        if env == None:
            env = self.env
            self.eval_env = self.env
            
        if timeout==None:
            try:
                timeout = env.timeout
            
            except:
                timeout = self.timesteps
                print("Timeout is set to {}. Please checkout the env documentation to verify the episode termination requirment or manually decide the timeout.".format(self.timesteps))

        for i_episode in range(n_ep):
            obs = []
            act = []
            rew = []
            ep_reward_ls = []
            ep_len_ls=[]
            ep_reward = 0
            ep_len=0
            observation = self.env.reset()
            obs.append(observation)
            for t in range(timeout):
                with torch.set_grad_enabled(False):
                    observation = torch.unsqueeze(torch.tensor(observation).float(),0)
                    action, _, __=self.forward(observation)
                    action = torch.squeeze(action)
                    observation, reward, done, info = self.env.step(action.detach().numpy())
                    #print("reward:", reward)
                    ep_reward += reward
                    ep_len +=1
                    rew.append(reward)
                    act.append(action)
                    obs.append(observation)

                    if done or t == timeout-1:
                        #print("Episode finished after {} timesteps".format(t+1))
                        #print("Survival time: {}".format(ep_reward+env.penalty*(t+1)))
                        break

            ep_reward_ls.append(ep_reward) 
            ep_len_ls.append(ep_len)
            self.total_rew.append(rew)
            self.total_act.append(act)
            self.total_obs.append(obs)

        self.total_ep_rew.append(np.mean(ep_reward_ls))
        self.total_ep_len.append(np.mean(ep_len_ls))
        
         
    def eval(self, obs):
        with torch.set_grad_enabled(False):
            obs = torch.unsqueeze(torch.tensor(obs).float(),0)
            action, _, __=self.forward(obs)
        return(torch.squeeze(action))
            