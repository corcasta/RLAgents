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
from copy import copy
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
#from queue import Queue 
class DQN(torch.nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.conv1  = torch.nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=8)
        self.activ1 = torch.nn.ReLU()
        
        self.conv2  = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.activ2 = torch.nn.ReLU()
        
        sample = torch.rand(size=(input_shape))
        self.dense1 = torch.nn.Linear(in_features=torch.flatten(self.conv2(self.conv1(sample))).shape[0], 
                                      out_features=256)
        self.activ3 = torch.nn.ReLU()
        
        self.dense2 = torch.nn.Linear(in_features=256, out_features=num_actions) 
        self.activ4 = torch.nn.ReLU()
        del sample
     
    def forward(self, x):
        x = self.activ1(self.conv1(x))
        x = self.activ2(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = self.activ3(self.dense1(x))
        x = self.activ4(self.dense2(x))
        return x

def state_preprocessor(state: np.ndarray, short_memory: list) -> np.ndarray:
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
    # State is now in BGR                                                  # Shape: (H, W)
    state_gray = cv2.cvtColor(state[:,:,::-1], cv2.COLOR_BGR2GRAY) 
    _ = short_memory.pop(0)
    short_memory.append(state_gray)
    return np.array(short_memory)   
        
class DQNAgent():
    def __init__(self, input_shape, num_actions, memory_size, batch_size, epsilon, device):
        self.input_shape = input_shape
        self.memory = TensorDictReplayBuffer(storage=LazyTensorStorage(max_size=memory_size), batch_size=32)
        self.batch_size     = batch_size
        self.num_actions    = num_actions
        self.epsilon        = epsilon
        self.actor_net      = DQN(input_shape, num_actions).to(device)
        self.target_net     = DQN(input_shape, num_actions).to(device)
        self.optimizer      = torch.optim.Adam(params=self.actor_net.parameters()) 
        self.criterion      = torch.nn.MSELoss()
        self.device         = device
        self.gamma = 0.99
        # Deactivating target network parameters for updating through backpropagation
        for params in self.target_net.parameters():
            params.requires_grad = False
        
   
        
    def store_experience(self, experience: tuple[np.ndarray, int, float, np.ndarray]) -> None:
        """
        Stores experience tuple in replay buffer/memory

        Args:
            experience (tuple[np.ndarray, int, float, np.ndarray]): _description_
        """
        state, action, reward, next_state = experience
        
        # Adding some input verification to store experiences               
        if state.shape[0] != self.input_shape[0]:
            raise Exception(f"Unadequate state shape: {state.shape}. Must be: {self.input_shape}") 
        
        if next_state.shape[0] != self.input_shape[0]:
            raise Exception(f"Unadequate next_state shape: {next_state.shape}. Must be: {self.input_shape}")
        
        print(self.device)
        self.memory.add(
            TensorDict(
                {
                    "state": torch.tensor(state, dtype=torch.float32),
                    "action": torch.tensor(action, dtype=torch.float32),
                    "reward": torch.tensor(reward, dtype=torch.float32),
                    "next_state": torch.tensor(next_state,  dtype=torch.float32)
                }
            )
            
        )
    
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Returns action given by the current policy.
        Args:
            state (np.ndarray): Shape can be (batch_size, NUM_CHANNELS, H, W) or (NUM_CHANNELS, H, W)

        Raises:
            Exception: _description_

        Returns:
            np.ndarray: _description_
        """
        # GREEDY POLICY
        prob = torch.rand(1)     
        if prob < self.epsilon:
            # Random action
            q_idx = torch.randint(low=0, high=self.num_actions, size=(1,))                  # Shape: (1,)
        else:
            # Greedy action
            # Flexible input handling whenever you want to provide information to NN.
            if len(state.shape) == 3:
                state = state[np.newaxis, :, :, :]                                                                             # Shape: (1,)
            elif len(state.shape) > 4:
                raise Exception(f"Unadequate input shape: {state.shape}, must be: (batch_size, NUM_CHANNELS, H, W)")
            
            state = torch.tensor(state, dtype=torch.float32).to(self.device)                      # Shape: (1, SHORT_MEMORY_SIZE, H, W)
            q_values = self.actor_net(state)                                      # Shape: (1, num_actions)
            max_q = torch.max(q_values, dim=-1, keepdim=False)                      
            q_idx = max_q.indices       
        
        return q_idx                                                                    
        
        
    def _update_target_network(self, tau: float=0.01):
        with torch.no_grad():
            for target_params, actor_params in zip(self.target_net.parameters(), self.actor_net.parameters()):
                target_params = tau*actor_params + (1-tau)*target_params
    
    
    def update(self):
        self.optimizer.zero_grad()
        batch = self.memory.sample()           
        states      = batch["state"].to(self.device)                                                        # shape: (batch_size, SHORT_MEMORY_SIZE, H, W)
        actions     = batch["action"].to(self.device)                                                       # shape: (batch_size,)
        rewards     = batch["reward"].to(self.device)                                                       # shape: (batch_size,)
        next_states = batch["next_state"].to(self.device)                                                   # shape: (batch_size, SHORT_MEMORY_SIZE, H, W)
        
        #print(f"states shape: {states.shape}")
        #print(f"actions shape: {actions.shape}")
        #print(f"rewards shape: {rewards.shape}")
        #print(f"next_states shape: {next_states.shape}")
        
        next_state_q_values = self.target_net(next_states)                                  # shape: (batch_size, num_actions)
        next_state_max_q_values = torch.max(next_state_q_values, dim=-1).values             # shape: (batch_size,)
        target_q_values = rewards + self.gamma*next_state_max_q_values                      # shape: (batch_size,)
        
        state_q_values = self.actor_net(states)                                             # shape: (batch_size, num_actions)
        current_q_values = state_q_values[torch.arange(self.batch_size), actions.int()]           # shape: (batch_size,)
        loss = self.criterion(current_q_values, target_q_values)                                        # shape: (1,)
        loss.backward()
        # We should be only updating self.actor NO self.target yet
        self.optimizer.step()
        # This step is to provide more stability to the learning phase
        self._update_target_network()