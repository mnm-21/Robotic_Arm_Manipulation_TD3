import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork

class Agent():

    def __init__(self, actor_learning_rate, critic_learning_rate, input_dims, tau, env=None, gamma=0.99, update_actor_interval=2, warmup = 1000, n_actions=2, max_size=1000000, layer1_size=256, layer2_size=128, batch_size=100, noise=0.1):

        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high # dim = 2
        self.min_action = env.action_space.low  # dim = 2
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.time_step = 0
        self.n_actions = n_actions
        self.warmup = warmup
        self.update_actor_interval = update_actor_interval

        self.actor = ActorNetwork(input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size,
                                   n_actions=n_actions, name='actor', chkpt_dir='tmp/td3', learning_rate=actor_learning_rate)
        
        self.critic_1 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size,
                                      n_actions=n_actions, name='critic_1', chkpt_dir='tmp/td3', learning_rate=critic_learning_rate)
        
        self.critic_2 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size,
                                      n_actions=n_actions, name='critic_2', chkpt_dir='tmp/td3', learning_rate=critic_learning_rate)
        
        self.target_actor = ActorNetwork(input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size,
                                         n_actions=n_actions, name='target_actor', chkpt_dir='tmp/td3', learning_rate=actor_learning_rate)
        
        self.target_critic_1 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size,
                                             n_actions=n_actions, name='target_critic_1', chkpt_dir='tmp/td3', learning_rate=critic_learning_rate)
        
        self.target_critic_2 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size,
                                             n_actions=n_actions, name='target_critic_2', chkpt_dir='tmp/td3', learning_rate=critic_learning_rate)
        
        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, eval=False):
        if self.time_step < self.warmup and eval is False:
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to(self.actor.device)
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,), dtype=np.float)).to(self.actor.device)
        mu_prime = T.clamp(mu_prime, min=T.tensor(self.min_action).to(self.actor.device),
                                     max=T.tensor(self.max_action).to(self.actor.device))
        self.time_step += 1
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        pass

    def learn(self):
        if self.memory.mem_cntr < self.batch_size * 10:
            return

        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)
        
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        dones = T.tensor(dones).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        next_states = T.tensor(next_states, dtype=T.float).to(self.actor.device)

        target_actions = self.target_actor.forward(next_states)
        target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale=self.noise, size=(self.batch_size, self.n_actions))), min=-0.5, max=0.5)
        target_actions = T.clamp(target_actions, min=T.tensor(self.min_action).to(self.actor.device),
                                                 max=T.tensor(self.max_action).to(self.actor.device))

        target_q_1 = self.target_critic_1.forward(next_states, target_actions)
        target_q_2 = self.target_critic_2.forward(next_states, target_actions)

        current_q_1 = self.critic_1.forward(states, actions)
        current_q_2 = self.critic_2.forward(states, actions)

        target_q_1 = target_q_1.view(-1)
        target_q_2 = target_q_2.view(-1)

        target_q_1[dones] = 0.0
        target_q_2[dones] = 0.0

        target_q = T.min(target_q_1, target_q_2)
        target_q = rewards + self.gamma * target_q
        target_q = target_q.view(self.batch_size, 1) 

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target_q, current_q_1)
        q2_loss = F.mse_loss(target_q, current_q_2)
    
        q_loss = q1_loss + q2_loss
        q_loss.backward() 

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_actor_interval != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic_1.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()


    def update_network_parameters(self, tau=None):
        if tau==None:
            tau = self.tau
        
        actor_params = self.actor.named_parameters()  
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        for param in critic_1_state_dict:
            critic_1_state_dict[param] = tau*critic_1_state_dict[param].clone() + (1-tau)*target_critic_1_state_dict[param].clone()

        for param in critic_2_state_dict:
            critic_2_state_dict[param] = tau*critic_2_state_dict[param].clone() + (1-tau)*target_critic_2_state_dict[param].clone()

        for param in actor_state_dict:
            actor_state_dict[param] = tau*actor_state_dict[param].clone() + (1-tau)*target_actor_state_dict[param].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)   
        self.target_actor.load_state_dict(actor_state_dict)        

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()
        print('...saved models...')

    def load_models(self):

        try:
            self.actor.load_checkpoint()
            self.target_actor.load_checkpoint()
            self.critic_1.load_checkpoint()
            self.critic_2.load_checkpoint()
            self.target_critic_1.load_checkpoint()
            self.target_critic_2.load_checkpoint()
            print('...loaded models...')
        except:
            print('...no models found...')
            print('...creating new models...')