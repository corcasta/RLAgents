import numpy as np

# These algorithms DISCOVERS the best way to REACT through EXPERIMENTATION and EXPLORATION
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


# These algorithms SOLVES and MDP by knowing the STATE TRANSITIONS PROBABILITIES.
# For VI and PI you need to know the state probabilities to used them in a ENV.
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