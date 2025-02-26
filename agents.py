import numpy as np

# These algorithms SOLVES and MDP by knowing the STATE TRANSITIONS PROBABILITIES.
# For VI and PI you need to know the state probabilities to used them in a ENV.
# ====================================================================================================================================== 
class VI:
    # References:
    # https://core-robotics.gatech.edu/files/2020/12/Value_Iteration-1.png
    def __init__(self, mdp, gamma):
        self.mdp = mdp                                          # A dictionary of dictionaries
        self.gamma = gamma                                      # Discount factor
        self.states = mdp.keys()
        self.states_values_table = np.zeros(len(self.states))
        self.policy = {state:0 for state in self.states}        # Initial default policy
    
    def react(self, state):
        return self.policy[state]

        
    def solve(self):
        # Value Iteration
        threshold = 0.05
        delta = 1000000
        while delta >= threshold:         
            acum_error = 0
            for state in self.states:
                state_transition_desc = self.mdp[state]
                actions = state_transition_desc.keys()
                state_action_values = np.zeros(len(actions))
                
                # Calculating values for each action in state S.
                for idx, action in enumerate(actions):
                    state_action_value = 0
                
                    # Weighted sum of the transitions from one state  
                    # to the possibles next states given action A.
                    for transition in state_transition_desc[action]:
                        prob, next_state, reward, _ = transition
                        state_action_value += prob*(reward + self.gamma*self.states_values_table[next_state])
                    state_action_values[idx] = state_action_value
                
                old_state_value = self.states_values_table[state]
                new_state_value = np.max(state_action_values)

                acum_error += (old_state_value - new_state_value)**2
                # Update state value
                self.states_values_table[state] = new_state_value
            
            # Updating delta
            norm_error = float(np.sqrt(acum_error))
            delta = min(delta, norm_error)
        
        # Policy Update
        for state in self.states:
            state_transition_desc = self.mdp[state]
            actions = state_transition_desc.keys()
            state_action_values = np.zeros(len(actions))
            
            # Calculating values for each action in state S.
            for idx, action in enumerate(actions):
                state_action_value = 0
            
                # Wighted sum of the transitions from one state  
                # to the possibles next states given action A.
                for transition in state_transition_desc[action]:
                    prob, next_state, reward, _ = transition
                    state_action_value += prob*(reward + self.gamma*self.states_values_table[next_state])
                state_action_values[idx] = state_action_value
            # Initial default policy
            self.policy[state] = int(np.argmax(state_action_values)) 

class PI:
    # References
    # https://core-robotics.gatech.edu/2020/12/16/bootcamp-summer-2020-week-4-policy-iteration-and-policy-gradient/
    def __init__(self, mdp, gamma):
        self.mdp = mdp
        self.states = mdp.keys()
        self.states_values_table = np.zeros(len(self.states))
        self.gamma = gamma
        # Initial default policy
        self.policy = {state:0 for state in self.states}
    
    def react(self, state):
        return self.policy[state]
    
    def solve(self):
        # Policy Evaluation
        while True:
            delta = 1000
            epsilon = 0.05
            while delta >= epsilon:
                accum_diff = 0
                for state in self.states:
                    temp = self.states_values_table[state]
                    action = self.policy[state]
                    state_value = 0
                    for transition in self.mdp[state][action]:
                        prob, next_state, reward, _ = transition
                        state_value += prob*(reward + self.gamma*self.states_values_table[next_state])
                    # Update state value
                    self.states_values_table[state] = state_value
                    accum_diff += np.abs(temp-self.states_values_table[state])
                delta = float(np.min((delta, accum_diff)))

            # Policy Iteration
            policy_stable = True
            for state in self.states:
                temp = self.policy[state]
                state_action_values_list = []
                for action in self.mdp[state].keys():
                    state_action_value = 0
                    for transition in self.mdp[state][action]:
                        prob, next_state, reward, _ = transition
                        state_action_value += prob*(reward + self.gamma*self.states_values_table[next_state])
                    state_action_values_list.append(state_action_value)
                # Updating policy
                self.policy[state] = int(np.argmax(state_action_values_list))
                if temp != self.policy[state]:
                    policy_stable = False

            # Break when it has already converged
            if policy_stable == True:
                break
# ====================================================================================================================================== 


# These algorithms DISCOVERS the best way to REACT through EXPERIMENTATION and EXPLORATION
# ======================================================================================================================================            
class Q:
    def __init__(self, state_space: int, action_space: int, alpha: float=0.1, gamma: float=0.99, epsilon: float=0.1):
        self.q_table = np.zeros((state_space, action_space))
        self.alpha = alpha      #   learning rate
        self.gamma = gamma      #   discount factor
        self.epsilon = epsilon  #   epsilon greedy
    
    def react(self, state: int):
        p = np.random.uniform(low=0, high=1)
        if p <= self.epsilon or self.q_table[state,:].sum() == 0:
            action = np.random.randint(low=0, high=4)
        else:
            action = np.argmax(self.q_table[state,:])
        return int(action)
    
    def update(self, state: int, action: int, reward: int, state_next: int):
        q = self.q_table[state, action]
        self.q_table[state, action] = q + self.alpha*(reward + self.gamma*self.q_table[state_next,:].max() - q) 


class SARSA:
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.99, epsilon=0.1):             
        self.q_table = np.zeros((state_space, action_space))
        self.alpha = alpha       #   learning rate 
        self.gamma = gamma       #   discount factor
        self.epsilon = epsilon   #   epsilon greedy
        
    
    def react(self, state):
        p = np.random.uniform(low=0, high=1)
        if p <= self.epsilon or self.q_table[state,:].sum() == 0:
            action = np.random.randint(low=0, high=4)
        else:
            action = np.argmax(self.q_table[state, :])
        return int(action)
    
    def update(self, state, action, reward, state_next):
        action_next = self.react(state_next)
        self.q_table[state, action] += self.alpha*(reward + self.gamma*self.q_table[state_next, action_next] - self.q_table[state, action])
# ====================================================================================================================================== 


# Deep learning implementations of Q learning
# ====================================================================================================================================== 
import cv2
import torch
import random
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
#from queue import Queue 
class DQN(torch.nn.Module):
    def __init__(self, img_shape, num_actions):
        super().__init__()
        self.conv1  = torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8)
        self.activ1 = torch.nn.ReLU()
        self.conv2  = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.activ2 = torch.nn.ReLU()
        
        sample = torch.rand(size=(img_shape))
        self.dense1 = torch.nn.Linear(in_features=torch.flatten(self.conv2(self.conv1(sample))).shape[0], 
                                      out_features=256)
        self.activ3 = torch.nn.ReLU()
        self.dense2 = torch.nn.Linear(in_features=256, out_features=num_actions) 
        self.activ4 = torch.nn.ReLU()
        del sample
     
    def forward(self, x):
        x = self.activ1(self.conv1(x))
        x = self.activ2(self.conv2(x))
        x = torch.flatten(x)
        x = self.activ3(self.dense1(x))
        x = self.activ4(self.dense2(x))
        return x
    
class DQNAgent():
    def __init__(self, input_shape, num_actions, memory_size, short_memory_size, batch_size, epsilon):
        self.memory = TensorDictReplayBuffer(storage=LazyTensorStorage(max_size=memory_size), batch_size=32)
        self.batch_size     = batch_size
        self.num_actions    = num_actions
        self.epsilon        = epsilon
        self.actor_net      = DQN(input_shape, num_actions)
        self.target_net     = DQN(input_shape, num_actions)
        self.optimizer
        self.loss           = torch.nn.MSELoss()
        self.short_memory   = []                                                           # All images stored are gray
        self.short_memory_size =  short_memory_size

        
    def state_preprocessor(self, state: np.ndarray) -> np.ndarray:
        """
        Makes input RGB np.uint8 image to GRAY scale and returns
        a stack of the most recent images stored in short_memory.

        Args:
            state (np.ndarray): RGB image. 
                                Shape: (H,W,3) 
                                dtype: np.uint8 

        Returns:
            np.ndarray: stack of the most recent images stored in 
                        short_memory.
                        Shape: (short_memory_size, H, W)
                        dtype: np.uint8
        """
        # State is now in BGR  
        state = state[:,:,::-1]                                                             # Shape: (H, W, C)
        state_gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)                                # Shape: (H, W)
        
        if len(self.short_memory) == self.short_memory_size:
            _ = self.short_memory.pop(0)
        self.short_memory.append(state_gray)
        
        return np.array(self.short_memory)                                                  # Shape: (C, H, W)
        
    def store_experience(self, experience: tuple[np.ndarray, int, float, np.ndarray]) -> None:
        """_summary_

        Args:
            experience (tuple[np.ndarray, int, float, np.ndarray]): _description_
        """
        state, action, reward, next_state = experience
        processed_state = self.state_preprocessor(state)
        
        if processed_state.shape[0] == 4:
            self.memory.add(
                TensorDict(
                    {
                        "state": torch.tensor(processed_state),
                        "action": torch.tensor(action),
                        "reward": torch.tensor(reward),
                        "next_state": torch.tensor(next_state)
                    }
                )
                
            )
    
    
    def epsilon_greedy_policy(self, state: np.ndarray, epsilon: float):
        # state: needs to handle both single and batch
        short_memory_cp = self.short_memory
        prob = torch.rand(1)
        batch_size = state.shape[0]
        current_q_values = self.actor_net(state)                                            # Shape: (batch_size, num_actions)
        
        if prob < epsilon:
            # random action
            q_indices = torch.randint(low=0, high=self.num_actions, size=(batch_size,))     # Shape: (batch_size,)
            q_values = current_q_values[torch.arange(batch_size), q_indices]                # Shape: (batch_size,)
        else:
            # greedy action
            max_q = torch.max(current_q_values, dim=-1, keepdim=False)                      
            q_indices = max_q.indices                                                       # Shape: (batch_size,)
            q_values  = max_q.values                                                        # Shape: (batch_size,)
        
        return q_indices, q_values                                                          # Shape: (batch_size,), (batch_size,)
        

    def learn(self):
        self.optimizer.zero_grad()
        batch = self.memory.sample()           
        states      = batch["state"]                                                        # shape: (NOT YET DEFINE)
        actions     = batch["action"]                                                       # shape: (batch_size,)
        rewards     = batch["reward"]                                                       # shape: (batch_size,)
        next_states = batch["next_state"]                                                   # shape: (NOT YET DEFINE)
        
        next_state_q_values = self.target_net(next_states)                                  # shape: (batch_size, num_actions)
        next_state_max_q_values = torch.max(next_state_q_values, dim=-1).values             # shape: (batch_size,)
        target_q_values = rewards + self.gamma*next_state_max_q_values                      # shape: (batch_size,)
        
        state_q_values = self.actor_net(states)                                             # shape: (batch_size, num_actions)
        current_q_values = state_q_values[torch.arange(self.batch_size), actions]           # shape: (batch_size,)
        self.loss(current_q_values, target_q_values)                                        # shape: (1,)
        self.loss.backward()
        # We should be only updating self.actor NO self.target yet
        self.optimizer.step()
        