import os
import torch
import ale_py
import numpy as np
import gymnasium as gym
from datetime import datetime
from tqdm.auto import tqdm
from agents import DQNAgent, state_preprocessor
from torch.utils.tensorboard import SummaryWriter 



RUNS_PATH = os.path.dirname(__file__) + "/runs"
DEVICE ="cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
WEIGTHS_UPDATES = 50000
#EPISODES = 1000
FIRST_N_FRAMES = 1000000
MAX_STEPS = 10000
SKIPPED_FRAMES = 4
FRAME_BUFFER_LEN = 4
IMAGE_SHAPE = (210, 160)
INPUT_SHAPE = (FRAME_BUFFER_LEN, *IMAGE_SHAPE)



def main():
    # Environment SETUP
    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout-v5", render_mode="human")
   

    # Agent SETUP
    agent = DQNAgent(INPUT_SHAPE, 4, 4, 32, 1, device="cpu")

    # Tensorboard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(RUNS_PATH + f"/breakout_{timestamp}")
    
    # PROGRESS BARS
    epoch_pg_bar          = tqdm(range(EPOCHS), position=0, desc="Epoch: 0", ncols=100)
    weights_update_pg_bar = tqdm(range(WEIGTHS_UPDATES), position=1, desc="Update: 0", ncols=150, leave=True)
    #skipped_frames_pg_bar = tqdm(range(SKIPPED_FRAMES), position=2, desc="Frame: 0", ncols=150, leave=True)

  
  
    frame_counter = 0
    for epoch in range(EPOCHS):
        steps = 0
        accum_reward = 0
        played_games = 0
        
        # At the start of every epoch the environment is reseted.
        frame_buffer = [] 
        obs, info = env.reset()
        processed_obs = state_preprocessor(obs, frame_buffer, FRAME_BUFFER_LEN) * FRAME_BUFFER_LEN
        
        # Episodes loop 
        for update_i in range(WEIGTHS_UPDATES):
            
            action = agent.get_action(processed_obs)
            
            for frame_i in range(SKIPPED_FRAMES): 
                next_obs, reward, terminated, truncated, info = env.step(action)
                processed_next_obs = state_preprocessor(next_obs, frame_buffer, FRAME_BUFFER_LEN)  
                agent.store_experience((processed_obs, reward, int(action), processed_next_obs))
                processed_obs = processed_next_obs
                
                accum_reward += reward
                steps += 1
                
                
                # Adjusting epsilon from greedy strategy
                frame_counter += 1    
                if frame_counter <= FIRST_N_FRAMES:
                    agent.epsilon -= (1-0.1)/FIRST_N_FRAMES
                else:
                    agent.epsilon = 0.05
                
                # Logic when terminating env or reaching max steps
                if terminated or steps == MAX_STEPS:
                    obs, info = env.reset()
                    frame_buffer = [] 
                    processed_obs = state_preprocessor(obs, frame_buffer, FRAME_BUFFER_LEN) * FRAME_BUFFER_LEN
                    steps = 0
                    played_games += 1
                    break
                
            # Update TQDM bar    
            weights_update_pg_bar.set_description(desc=f"Update: {update_i}") 
            weights_update_pg_bar.set_postfix({"accum_reward": accum_reward, "games": played_games})
            weights_update_pg_bar.update()
            
            # Update agent weights
            agent.update()
        
        avg_reward = accum_reward / WEIGTHS_UPDATES
        writer.add_scalars("Training Metrics", 
                           {"AVG_Reward": avg_reward,
                            "PLAYED_GAMES": played_games,}, 
                           epoch)
        
        # Update TQDM bar
        epoch_pg_bar.set_description(desc=f"Epoch: {epoch}")
        epoch_pg_bar.update()
        
        
    env.close()


if __name__ == '__main__':
    main()