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



RUNS_PATH = os.path.dirname(__file__) + "/runs"
DEVICE ="cuda" if torch.cuda.is_available() else "cpu"
EPOCHS           = 500 #100
WEIGTHS_UPDATES  = 10_000 # 50_000
#EPISODES = 1000
FIRST_N_FRAMES   = 500_000 #1_000_000
MAX_STEPS        = 100_000
SKIPPED_FRAMES   = 4
FRAME_BUFFER_SIZE = 4
IMAGE_SHAPE      = (84, 84)
INPUT_SHAPE      = (FRAME_BUFFER_SIZE, *IMAGE_SHAPE)
MEMORY_BUFFER_SIZE = 100_000


def main():
    # Environment SETUP
    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout-v5")#, render_mode="human")
   

    # Agent SETUP
    agent = DQNAgent(INPUT_SHAPE, 4, MEMORY_BUFFER_SIZE, 32, 1, device=DEVICE)

    # Tensorboard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(RUNS_PATH + f"/breakout_{timestamp}")
    
    # PROGRESS BARS
    epoch_pg_bar          = tqdm(range(EPOCHS), position=0, desc="Epoch: 0", ncols=100)
    weights_update_pg_bar = tqdm(range(WEIGTHS_UPDATES), position=1, desc="Update: 0", ncols=150, leave=True)
    #skipped_frames_pg_bar = tqdm(range(SKIPPED_FRAMES), position=2, desc="Frame: 0", ncols=150, leave=True)

  
  
    frame_counter = 0
    for epoch in range(EPOCHS):
        weights_update_pg_bar.n = 0
        steps = 0
        accum_reward = 0
        played_games = 0
        
        # At the start of every epoch the environment is reseted.
        frame_buffer = [] 
        obs, info = env.reset()
        processed_obs = state_preprocessor(obs, frame_buffer, FRAME_BUFFER_SIZE) * FRAME_BUFFER_SIZE
        
        # Episodes loop 
        for update_i in range(WEIGTHS_UPDATES):
            
            action = agent.get_action(processed_obs)
                
            for frame_i in range(SKIPPED_FRAMES): 
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Data preprocessing
                done = terminated or truncated
                reward = reward_clipping(reward)
                processed_next_obs = state_preprocessor(next_obs, frame_buffer, FRAME_BUFFER_SIZE)  
                experience = (processed_obs, int(action), reward, processed_next_obs, done)
                
                # Storing data in buffer and moving to next iter
                agent.store_experience(experience)
                processed_obs = copy.deepcopy(processed_next_obs)
                
                accum_reward += reward
                steps += 1
                
                
                # Adjusting epsilon from greedy strategy
                writer.add_scalar("Epsilon Evolution", agent.epsilon, frame_counter)
                frame_counter += 1    
                if frame_counter <= FIRST_N_FRAMES:
                    agent.epsilon -= (1-0.1)/FIRST_N_FRAMES
                else:
                    agent.epsilon = 0.05
                
                # Logic when terminating env or reaching max steps
                if terminated or truncated or steps == MAX_STEPS:
                    obs, info = env.reset()
                    frame_buffer = [] 
                    processed_obs = state_preprocessor(obs, frame_buffer, FRAME_BUFFER_SIZE) * FRAME_BUFFER_SIZE
                    steps = 0
                    played_games += 1
                    break
                
            # Update TQDM bar    
            weights_update_pg_bar.set_description(desc=f"Update") 
            weights_update_pg_bar.set_postfix({"accum_reward": accum_reward, "games": played_games})
            weights_update_pg_bar.update()
            
            # Update agent weights
            agent.update()
        
        avg_reward = accum_reward / played_games
        writer.add_scalars("Training Metrics", 
                           {"AVG_Reward": avg_reward,
                            "PLAYED_GAMES": played_games,}, 
                           epoch)
        writer.add_scalar("Epsilon Evolution", agent.epsilon, frame_counter)
        writer.flush() 
        
        # Update TQDM bar
        epoch_pg_bar.set_description(desc=f"Epoch: {epoch}")
        epoch_pg_bar.set_postfix({"avg_reward": avg_reward, "games": played_games})
        epoch_pg_bar.update()
        
        
    env.close()

if __name__ == '__main__':
    main()