import time
import os
import sys
import numpy as np
import gym
import robosuite as suite
from robosuite.wrappers import GymWrapper
from torch.utils.tensorboard import SummaryWriter
from td3_torch import Agent

if __name__ == "__main__":

    if not os.path.exists("tmp/td3"):
        os.makedirs("tmp/td3")

    env_name = "Door"

    env = suite.make(
        env_name,
        robots="Panda",
        controller_configs= suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        horizon=300,
        has_renderer=True,
        use_camera_obs=False,
        render_camera="frontview",
        has_offscreen_renderer=True,
        reward_shaping=True, 
        control_freq=20,
    )

    env = GymWrapper(env)
    actor_learning_rate = 1e-3
    critic_learning_rate = 1e-3
    tau = 0.005
    gamma = 0.99
    update_actor_interval = 2
    warmup = 1000
    n_actions = env.action_space.shape[0]
    max_size = 1000000
    layer1_size = 256
    layer2_size = 128
    batch_size = 128
    noise = 0.1

    agent = Agent(actor_learning_rate=actor_learning_rate, 
                  critic_learning_rate=critic_learning_rate, 
                  input_dims=env.observation_space.shape, 
                  tau=tau, 
                  env=env, 
                  gamma=gamma, 
                  update_actor_interval=update_actor_interval, 
                  warmup=warmup, 
                  n_actions=n_actions, 
                  max_size=max_size, 
                  layer1_size=layer1_size, 
                  layer2_size=layer2_size, 
                  batch_size=batch_size, 
                  noise=noise)
    
    n_games = 1000
    best_score = 0

    agent.load_models() # loads models from default tmp/td3 directory

    for i in range(n_games):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(obs, eval=True)
            obs_, reward, done, info = env.step(action)
            env.render()
            score += reward
            obs = obs_
            time.sleep(0.05)
        
        print(f"episode: {i} score: {score}")

        


    