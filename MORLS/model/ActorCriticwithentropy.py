import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, n_neurons=64, activation=nn.Tanh, distribution = torch.distributions.multivariate_normal.MultivariateNormal):
        '''
        input dim = obs dim
        output dim = action dim
        '''
        super(ActorCriticPolicy, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_neurons = n_neurons
        self.distribution = distribution

        # Policy Network
        self.h0 = nn.Linear(input_dim, n_neurons)
        self.h0_act = activation()
        self.h1 = nn.Linear(n_neurons, n_neurons)
        self.h1_act = activation()
        self.output_layer = nn.Linear(n_neurons, output_dim)

        # Value Network
        self.h0v = nn.Linear(input_dim, n_neurons)
        self.h0_actv = activation()
        self.h1v = nn.Linear(n_neurons, n_neurons)
        self.h1_actv = activation()
        self.value_head = nn.Linear(n_neurons, 1)

        self.variance = torch.nn.Parameter(torch.tensor([0.0]*self.output_dim), requires_grad = True)

        torch.manual_seed(123)        

    def forward(self, obs, action=None):
        '''
        obs: tensor.size([input_dim])
        '''
        # Policy Forward: obs->action
        x = self.h0(obs)
        x = self.h0_act(x)
        x = self.h1(x)
        x = self.h1_act(x)
        action_mean = self.output_layer(x) #tensor.size([#obs,act_dim])

        action_variance = torch.exp(self.variance)
        
        #print("action mean:", action_mean, "action variance:", self.variance)
        
        action_dist = self.distribution(action_mean, torch.diag_embed(action_variance)) #tensor.size([#obs, obs_dim, obs_dim])

        if action is None:
            action = action_dist.sample() #tensor.size([#obs, act_dim])
        
        logprob = action_dist.log_prob(action) #tensor.size([#obs])
        entropy = action_dist.entropy()
        
        # Value Forward: obs-> state value(V)
        x = self.h0v(obs)
        x = self.h0_actv(x)
        x = self.h1v(x)
        x = self.h1_actv(x)
        value = self.value_head(x)
        value = torch.squeeze(value) #tensor.size([#obs,1]) - > tensor.size([#obs])

        return action, logprob, entropy, value # action: tensor.size([#obs, act_dim]), logprob: tensor.size([#obs]), entropy: tensor.size([#obs]), value: tensor.size([#obs])

                                           