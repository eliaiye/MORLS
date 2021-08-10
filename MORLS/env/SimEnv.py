import numpy as np
import gym


class SimEnv(gym.Env):
    def __init__(self, b=10, c=0.5, e=3, y0=2, threshold=1, penalty=0, timeout=100):
        '''
        action_space: (y, delta_t)
        observation_space: (x)
        
        Input
        threshold: when x is lower than this threshold, patient die
        penalty: this parameter adds cost to frequent visits
        '''
        super(SimEnv, self).__init__()
        
        self.b = b
        self.c = c
        self.e = e
        self.y0 = y0
        self.threshold = threshold
        self.penalty = penalty
            
        self.action_space = gym.spaces.Box(low=0,high=float("inf"),shape=(2,),dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(1,), dtype=np.float32)
        self.state = None
        self.timeout = timeout
        self.steps_elapsed = 0
        
    def reset(self):
        self.state = np.random.uniform(self.threshold, 15, 1) # numpy array (1,)
        self.steps_elapsed=0
        return(self.state)
        
        
    def step(self, action):
        delta_t=action[1]
        y=action[0]
        next_obs = self.xfunc(x=self.state, delta_t=delta_t, y=action[0], b=self.b,c= self.c, e=self.e, y0=self.y0)
        
        if next_obs[0] > self.threshold:
            reward = delta_t - self.penalty
        elif (-1/self.c*np.log((self.threshold-self.state[0]+self.b-self.e*(2*y*self.y0-y**2))/self.b) < 0):
            reward = -self.penalty 
        else:
            reward = -1/self.c*np.log((self.threshold-self.state[0]+self.b-self.e*(2*y*self.y0-y**2))/self.b) - self.penalty
            
        self.state = next_obs 
        self.steps_elapsed+=1
        
        return(np.array(self.state).astype(np.float32), reward, next_obs[0] <= self.threshold or self.steps_elapsed > self.timeout , {"x": self.state, "y": action[0], "delta_t": action[1], "dead": next_obs[0] <= self.threshold})
    
    def xfunc(self, x, delta_t, y, b, c, e, y0):
        '''
        Input
        x: last obs/state
        delta_t: time interval btw visits
        y: dosage
        b, c, e, y0: parameters
        Output
        current obs/state
        '''
        return(x-b+b*np.exp(-c*delta_t)+e*(2*y*y0-y**2)) 
    
    def render(self, mode=None):
        pass
    def close(self):
        pass