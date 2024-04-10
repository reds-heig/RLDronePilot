from collections import deque
from DDPGNetworks import ActorNetwork, CriticNetwork
from Environment import Environment
import multiprocessing as mp
import numpy as np
import torch as T
import torch.nn.functional as F
from utils import init_neptune

import os

class OUActionNoise:
    def __init__(self, mu, sigma=0.05, theta=.2, dt=5e-3, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    
    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    
    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class Agent(mp.Process):
    def __init__(self, replay_buffer, actor_lr, input_dims, n_actions, 
                 layer1_size, layer2_size, batch_size, allow_x_movement, weights_shared):
        super().__init__()
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.allow_x_movement = allow_x_movement

        self.actor_weights_shared = weights_shared
        self.actor = ActorNetwork(actor_lr, input_dims, layer1_size,
                                  layer2_size, n_actions=n_actions, 
                                  allow_x_movement=allow_x_movement,
                                  name='Actor')


class LearningAgent(Agent):
    def __init__(self, replay_buffer, new_weights_events, load_weights, actor_lr, critic_lr, 
                 input_dims, tau, weights_shared, gamma=0.99, n_actions=2, layer1_size=400, 
                 layer2_size=300, batch_size=64, allow_x_movement=True):
        super().__init__(
            replay_buffer=replay_buffer, actor_lr=actor_lr, input_dims=input_dims,
            weights_shared=weights_shared, n_actions=n_actions, layer1_size=layer1_size, 
            layer2_size=layer2_size, batch_size=batch_size, allow_x_movement=allow_x_movement,
        )
        self.tau = tau
        self.gamma = gamma
        self.new_weights_events = new_weights_events

        self.critic = CriticNetwork(critic_lr, input_dims, layer1_size,
                                    layer2_size, n_actions=n_actions,
                                    name='Critic')
        self.target_actor = ActorNetwork(actor_lr, input_dims, layer1_size,
                                         layer2_size, n_actions=n_actions,
                                         allow_x_movement=allow_x_movement,
                                         name='TargetActor')
        self.target_critic = CriticNetwork(critic_lr, input_dims, layer1_size,
                                           layer2_size, n_actions=n_actions,
                                           name='TargetCritic')
        if load_weights:
            self.load_models()
        self.update_network_parameters(tau=1)
        
    
    def run(self):
        update_cntr = 0
        while True:
            self.replay_buffer.wait_for_transition()
            self.learn()

            if update_cntr % 100 == 0:
                self.save_models()
            update_cntr += 1
        
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

        # upload the actor weights to the shared memory space so that other processes can access them
        cpu_weights = {key: value.cpu() for key, value in self.actor.state_dict().items()}
        self.actor_weights_shared.update(cpu_weights)
        # notify explorer agents that new networks weights are available
        self.send_new_weights_notif()
        
        
    def learn(self):
        if not self.replay_buffer.is_available(self.batch_size):
            return
        state, action, reward, new_state, done = self.replay_buffer.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()


    def send_new_weights_notif(self):
        for ev in self.new_weights_events:
            ev.set()

    
    def check_actor_params(self):
        current_actor_params = self.actor.named_parameters()
        current_actor_dict = dict(current_actor_params)
        original_actor_dict = dict(self.original_actor.named_parameters())
        original_critic_dict = dict(self.original_critic.named_parameters())
        current_critic_params = self.critic.named_parameters()
        current_critic_dict = dict(current_critic_params)
        print('Checking Actor parameters')

        for param in current_actor_dict:
            print(param, T.equal(original_actor_dict[param], current_actor_dict[param]))
        print('Checking critic parameters')
        for param in current_critic_dict:
            print(param, T.equal(original_critic_dict[param], current_critic_dict[param]))
        input()

    
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    
    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
        
        
class ExplorationAgent(Agent):
    def __init__(self, replay_buffer, new_weights_event, actor_lr, input_dims, render_dict, is_plot_process,
                 noise_sigma, noise_theta, noise_dt, pbar_queue, log_lock, weights_shared, environment_params, neptune_params, 
                 n_actions=2, layer1_size=400, layer2_size=300, batch_size=64, memory_size=10, allow_x_movement=True):
        super().__init__(
            replay_buffer=replay_buffer, actor_lr=actor_lr, input_dims=input_dims, n_actions=n_actions, 
            layer1_size=layer1_size, layer2_size=layer2_size, batch_size=batch_size, allow_x_movement=allow_x_movement, weights_shared=weights_shared,
        )
        self.is_plot_process = is_plot_process
        self.render_dict = render_dict
        self.pbar_queue = pbar_queue
        self.log_lock = log_lock
        self.neptune = None
        self.neptune_params = neptune_params
        self.new_weights_event = new_weights_event
        self.env = Environment(**environment_params)
        self.noise = OUActionNoise(mu=np.zeros(n_actions), sigma=noise_sigma, theta=noise_theta, dt=noise_dt)

        self.memory_size = memory_size
        self.states_memory = deque(maxlen=self.memory_size+1)
        self.reset_states_memory()

    
    def run(self):
        # init Neptune if this Process is the logger one
        if self.neptune_params is not None:
            self.neptune = init_neptune(self.neptune_params)
            self.env.set_neptune(self.neptune)
        
        while True:
            episode_speedz, episode_speedx, episode_speeda, episode_score = 0., 0., 0., 0.
            if self.new_weights_event.is_set(): # load updated networks weights if available
                self.actor.load_state_dict(dict(self.actor_weights_shared))
                self.new_weights_event.clear()
            self.reset_states_memory()
            state = self.env.reset()
            terminated, truncated, done = False, False, False

            while not done: # while episode is not over
                action = self.choose_action(state)
                new_state, reward, terminated, truncated = self.env.step(action)
                done = terminated or truncated
                self.remember(state, action, reward, new_state, int(done))
                state = new_state

                if self.is_plot_process:
                    self.render_dict.update(self.env.get_render_dict())

                if not self.allow_x_movement:
                    action = [action[0], 0., action[1]]
                speed_z, speed_x, speed_a = self.env.drone.pilot_speeds_to_drones(*action)
                episode_speedz += speed_z
                episode_speedx += speed_x
                episode_speeda += speed_a
                episode_score += reward

            with self.log_lock: # sync between processes before updating progress bar and Neptune
                if self.neptune is not None:
                    self.neptune['train/score'].log(np.mean(episode_score))
                    self.neptune['train/avg_speed_z'].log(np.mean(episode_speedz))
                    self.neptune['train/avg_speed_x'].log(np.mean(episode_speedx))
                    self.neptune['train/avg_speed_a'].log(np.mean(episode_speeda))
                    self.neptune['train/terminated'].log(terminated)
                    self.neptune['train/truncated'].log(truncated)
                self.pbar_queue.put(1)
        
        
    def choose_action(self, observation):
        self.actor.eval()
        # concat observation & previous actions (if memory_size > 0)
        actor_input = np.concatenate([observation, np.array(self.states_memory)[1:].flatten()])
        actor_input = T.tensor(actor_input, dtype=T.float).to(self.actor.device)
        # get action
        mu = self.actor.forward(actor_input).to(self.actor.device)
        # add noise
        noise = self.noise()
        mu_prime = mu + T.tensor(noise, dtype=T.float).to(self.actor.device)
        # select action with expected best reward
        self.actor.train()
        action = mu_prime.cpu().detach().numpy()
        # save observation to memory
        self.states_memory.append(list(observation))
        # clip speeds to make sure they are in the correct ranges (not guaranteed after the noise addition)
        action[0]  = np.clip(action[0],   0., 1.)
        action[1:] = np.clip(action[1:], -1., 1.)
        return action


    def remember(self, state, action, reward, new_state, done):
        prev_actor_input = np.concatenate([state, np.array(self.states_memory)[0:-1].flatten()])
        new_actor_input = np.concatenate([new_state, np.array(self.states_memory)[1:].flatten()])
        self.replay_buffer.store_transition(prev_actor_input, action, reward, new_actor_input, done)


    def reset_states_memory(self):
        for _ in range(self.memory_size+1): # fill memory with default action values
            if self.allow_x_movement:
                self.states_memory.append([
                    0., # z-speed: minimum speed
                    0., # x-speed: neutral
                    0., # a-speed: neutral
                ])
            else:
                self.states_memory.append([
                    0., # z-speed: minimum speed
                    0., # a-speed: neutral
                ])