import time
import os
import sys
import numpy as np
import gym
import robosuite as suite
from robosuite.wrappers import GymWrapper
from torch.utils.tensorboard import SummaryWriter
from td3_torch import Agent
from buffer import ReplayBuffer

if __name__ == "__main__":

    if not os.path.exists("tmp/td3"):
        os.makedirs("tmp/td3")

    env_name = "Door"

    env = suite.make(
        env_name,
        robots="Panda",
        controller_configs= suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        horizon=300,
        has_renderer=False,
        use_camera_obs=False,
        reward_shaping=True, #
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

    writer = SummaryWriter(f"runs/{env_name}")
    n_games = 1000
    best_score = 0
    episode_identifier = f"actor_{actor_learning_rate}_critic_{critic_learning_rate}_gamma_{gamma}_update_actor_interval_{update_actor_interval}_warmup_{warmup}_n_actions_{n_actions}_layer1_size_{layer1_size}_layer2_size_{layer2_size}_batch_size_{batch_size}_noise_{noise}"

    agent.load_models()
    for i in range(n_games):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward
            agent.remember(obs, action, reward, obs_, done)
            agent.learn()
            obs = obs_

        writer.add_scalar(f"score-{episode_identifier}", score, global_step= i)
        
        if (i%10 == 0):
            agent.save_models(episode_identifier)
        
        print(f"episode: {i} score: {score}")

        if score > best_score:
            best_score = score
            agent.save_models(episode_identifier)
        

