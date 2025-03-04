import os
import copy
import torch
import ale_py
import numpy as np
import gymnasium as gym
from datetime import datetime
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter 
from agents import DQNAgent, state_preprocessor, reward_clipping



RUNS_PATH          = os.path.dirname(__file__) + "/runs"
DEVICE             ="cuda" if torch.cuda.is_available() else "cpu"
EPOCHS             = 500 #100
EPISODES           = 10_000
FIRST_N_FRAMES     = 500_000 #1_000_000
MEMORY_BUFFER_SIZE = 100_000
MAX_STEPS          = 100_000
SKIPPED_FRAMES     = 4
FRAME_BUFFER_SIZE  = 4
IMAGE_SHAPE        = (84, 84)
INPUT_SHAPE        = (FRAME_BUFFER_SIZE, *IMAGE_SHAPE)


def main():
    # Environment SETUP
    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout-v5", max_episode_steps=MAX_STEPS, obs_type="grayscale")#, render_mode="human")
   

    # Agent SETUP
    agent = DQNAgent(INPUT_SHAPE, 4, MEMORY_BUFFER_SIZE, 32, 1, device=DEVICE)

    # Tensorboard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(RUNS_PATH + f"/breakout_{timestamp}")
    
    # PROGRESS BARS
    epoch_pg_bar   = tqdm(range(EPOCHS), position=0, desc="Epoch", postfix="total_reward: 0", ncols=100)
    episode_pg_bar = tqdm(range(EPISODES), position=1, desc="Episode", postfix="total_reward: 0", ncols=150, leave=True)
  
  
    for epoch in range(EPOCHS):
        episode_pg_bar.n = 0
        epoch_total_reward = 0

        # Episodes loop 
        for episode in range(EPISODES):
            # At the start of every epoch the environment is reseted.
            episode_total_reward = 0
            dead = False
            frame_buffer = [] 
            obs, info = env.reset()
            processed_obs = state_preprocessor(obs, frame_buffer, FRAME_BUFFER_SIZE) * FRAME_BUFFER_SIZE
            
            while not dead:
                action = agent.get_action(processed_obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                episode_total_reward += reward
                dead = terminated or truncated
                frame_number = info["frame_number"]
                
                # Data preprocessing
                reward = reward_clipping(reward)
                processed_next_obs = state_preprocessor(next_obs, frame_buffer, FRAME_BUFFER_SIZE)  
                experience = (processed_obs, action, reward, processed_next_obs, dead)
                
                # Storing data in buffer and moving to next iter
                agent.store_experience(experience)
                processed_obs = copy.deepcopy(processed_next_obs)
                
                                
                # Adjusting epsilon from greedy strategy
                writer.add_scalar("Epsilon Evolution", agent.epsilon, frame_number)    
                if frame_number <= FIRST_N_FRAMES:
                    agent.epsilon -= (1-0.1)/FIRST_N_FRAMES
                else:
                    agent.epsilon = 0.05
                
            epoch_total_reward += episode_total_reward

            # Updating tensorboard log
            writer.add_scalar("Total_Reward/Episode", episode_total_reward, episode)
                
            # Update TQDM bar    
            episode_pg_bar.set_postfix({"total_reward": episode_total_reward})
            episode_pg_bar.update()
            
            # Update agent weights
            agent.update()
            
        # Updating tensorboard log
        writer.add_scalar("Total_Reward/Epoch", epoch_total_reward, epoch)
        writer.flush() 
        
        # Update TQDM bar
        epoch_pg_bar.set_description(desc=f"Epoch")
        epoch_pg_bar.set_postfix({"total_reward": epoch_total_reward})
        epoch_pg_bar.update()
        
    writer.close()    
    env.close()

if __name__ == '__main__':
    main()