import argparse
import os
import sys
import random
import numpy as np
import scipy

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data

from params import Params

import pickle
import time as t

from model import ActorCriticNet, Shared_obs_stats

import statistics
import matplotlib.pyplot as plt
from operator import add, sub


import pickle
import threading
import torch.multiprocessing as mp
import queue
from random import randint

from cassie_env.cassieRLEnvMirror import cassieRLEnvMirror
from cassie_env.cassieRLEnvMirrorIKTraj import cassieRLEnvMirrorIKTraj
from utils import TrafficLight
from utils import Counter



class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, events):
        for event in zip(*events):
            self.memory.append(event)
            if len(self.memory)>self.capacity:
                del self.memory[0]

    def push_half(self, events):
        temp_memory = []
        for event in zip(*events):
            temp_memory.append(event)
        self.memory = self.memory + temp_memory[0:len(temp_memory)//2]

        while len(self.memory)>self.capacity:
            del self.memory[0]

    def clear(self):
        self.memory = []

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        #print(map(lambda x: np.concatenate(x, 0), samples))
        return map(lambda x: np.concatenate(x, 0), samples)

def normal(x, mu, log_std):
    a = (x - mu)/(log_std.exp())
    a = -0.5 * a.pow(2)
    a = torch.sum(a, dim=1)
    b = torch.sum(log_std, dim=1)
    #print(result)
    return a-b

class RL(object):
    def __init__(self, env, hidden_layer=[64, 64]):
        self.env = env
        #self.env.env.disableViewer = False
        self.num_inputs = env.observation_space.shape[0]
        self.num_outputs = env.action_space.shape[0]
        self.hidden_layer = hidden_layer

        self.params = Params()

        self.model = ActorCriticNet(self.num_inputs, self.num_outputs,self.hidden_layer)
        self.model.share_memory()
        self.shared_obs_stats = Shared_obs_stats(self.num_inputs)
        self.best_model = ActorCriticNet(self.num_inputs, self.num_outputs,self.hidden_layer)
        self.memory = ReplayMemory(self.params.num_steps * 10000)
        self.test_mean = []
        self.test_std = []

        self.noisy_test_mean = []
        self.noisy_test_std = []
        self.fig = plt.figure()
        #self.fig2 = plt.figure()
        self.lr = self.params.lr
        plt.show(block=False)

        self.test_list = []
        self.noisy_test_list = []
        self.queue = mp.Queue()

        self.mpdone = [mp.Event(), mp.Event(), mp.Event(), mp.Event()]

        self.process = []
        self.traffic_light = TrafficLight()
        self.counter = Counter()

        self.best_trajectory = ReplayMemory(300)
        self.best_score_queue = mp.Queue()
        self.best_score = mp.Value("f", 0)

        self.expert_trajectory = ReplayMemory(600000)

        self.validation_trajectory = ReplayMemory(6000*9)

        self.best_validation = 1.0
        self.current_best_validation = 1.0

    def normalize_data(self, num_iter=50000, file='shared_obs_stats.pkl'):
        state = self.env.reset_for_normalization()
        state = Variable(torch.Tensor(state).unsqueeze(0))
        model_old = ActorCriticNet(self.num_inputs, self.num_outputs,self.hidden_layer)
        model_old.load_state_dict(self.model.state_dict())
        for i in range(num_iter):
            self.shared_obs_stats.observes(state)
            state = self.shared_obs_stats.normalize(state)
            mu, log_std, v = model_old(state)
            eps = torch.randn(mu.size())
            action = (mu + log_std.exp()*Variable(eps))
            env_action = action.data.squeeze().numpy()
            state, reward, done, _ = self.env.step(env_action)

            if done:
                state = self.env.reset()

            state = Variable(torch.Tensor(state).unsqueeze(0))

        with open(file, 'wb') as output:
            pickle.dump(self.shared_obs_stats, output, pickle.HIGHEST_PROTOCOL)

    def run_test(self, num_test=1):
        state = self.env.reset_for_test()
        state = Variable(torch.Tensor(state).unsqueeze(0))
        model_old = ActorCriticNet(self.num_inputs, self.num_outputs,self.hidden_layer)
        model_old.load_state_dict(self.model.state_dict())
        ave_test_reward = 0

        total_rewards = []
        '''self.fig2.clear()
        circle1 = plt.Circle((0, 0), 0.5, edgecolor='r', facecolor='none')
        circle2 = plt.Circle((0, 0), 0.01, edgecolor='r', facecolor='none')
        plt.axis('equal')'''
        
        for i in range(num_test):
            total_reward = 0
            while True:
                state = self.shared_obs_stats.normalize(state)
                mu, log_std, v = self.model(state)
                action = mu.data.squeeze().numpy()
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                #print(state)
                #print("done", done, "state", state)

                if done:
                    state = self.env.reset_for_test()
                    #print(self.env.position)
                    #print(self.env.time)
                    state = Variable(torch.Tensor(state).unsqueeze(0))
                    ave_test_reward += total_reward / num_test
                    total_rewards.append(total_reward)
                    break
                state = Variable(torch.Tensor(state).unsqueeze(0))
        #print("avg test reward is", ave_test_reward)

        reward_mean = statistics.mean(total_rewards)
        reward_std = statistics.stdev(total_rewards)
        self.test_mean.append(reward_mean)
        self.test_std.append(reward_std)
        self.test_list.append((reward_mean, reward_std))
        #print(self.model.state_dict())

    def run_test_with_noise(self, num_test=10):
        state = self.env.reset_for_test()
        state = Variable(torch.Tensor(state).unsqueeze(0))
        model_old = ActorCriticNet(self.num_inputs, self.num_outputs,self.hidden_layer)
        model_old.load_state_dict(self.model.state_dict())
        ave_test_reward = 0

        total_rewards = []
        '''self.fig2.clear()
        circle1 = plt.Circle((0, 0), 0.5, edgecolor='r', facecolor='none')
        circle2 = plt.Circle((0, 0), 0.01, edgecolor='r', facecolor='none')
        plt.axis('equal')'''
        
        for i in range(num_test):
            total_reward = 0
            while True:
                state = self.shared_obs_stats.normalize(state)
                mu, log_std, v = self.model(state)
                eps = torch.randn(mu.size())
                action = (mu + 0.1*Variable(eps))
                action = action.data.squeeze().numpy()
                state, reward, done, _ = self.env.step(action)
                total_reward += reward

                if done:
                    state = self.env.reset_for_test()
                    state = Variable(torch.Tensor(state).unsqueeze(0))
                    ave_test_reward += total_reward / num_test
                    total_rewards.append(total_reward)
                    break
                state = Variable(torch.Tensor(state).unsqueeze(0))
        #print("avg test reward is", ave_test_reward)

        reward_mean = statistics.mean(total_rewards)
        reward_std = statistics.stdev(total_rewards)
        self.noisy_test_mean.append(reward_mean)
        self.noisy_test_std.append(reward_std)
        self.noisy_test_list.append((reward_mean, reward_std))

    def plot_statistics(self):
        
        ax = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        low = []
        high = []
        index = []
        noisy_low = []
        noisy_high = []
        for i in range(len(self.test_mean)):
            low.append(self.test_mean[i] - self.test_std[i])
            high.append(self.test_mean[i] + self.test_std[i])
            noisy_low.append(self.noisy_test_mean[i]-self.noisy_test_std[i])
            noisy_high.append(self.noisy_test_mean[i]+self.noisy_test_std[i])
            index.append(i)
        #ax.set_xlim([0,1000])
        #ax.set_ylim([0,300])
        plt.xlabel('iterations')
        plt.ylabel('average rewards')
        ax.plot(self.test_mean, 'b')
        ax2.plot(self.noisy_test_mean, 'g')
        ax.fill_between(index, low, high, color='cyan')
        ax2.fill_between(index, noisy_low, noisy_high, color='r')
        #ax.plot(map(sub, test_mean, test_std))
        self.fig.canvas.draw()

        #plt.draw()
        #plt.errorbar(self.test_mean)

    def collect_samples(self, num_samples, start_state=None, noise=-2.0, env_index=0, random_seed=1):

        random.seed(random_seed)
        torch.manual_seed(random_seed+1)
        np.random.seed(random_seed+2)

        if start_state == None:
            start_state = self.env.reset()
        samples = 0
        done = False
        states = []
        next_states = []
        actions = []
        rewards = []
        values = []
        q_values = []
        self.model.set_noise(noise)
        #print("soemthing 1")
        model_old = ActorCriticNet(self.num_inputs, self.num_outputs, self.hidden_layer)
        model_old.load_state_dict(self.model.state_dict())
        #print("something 2")
        model_old.set_noise(noise)


        state = start_state
        state = Variable(torch.Tensor(state).unsqueeze(0))
        total_reward = 0
        #q_value = Variable(torch.zeros(1, 1))
        while True:
            self.model.set_noise(-2.0)
            model_old.set_noise(-2.0)
            signal_init = self.traffic_light.get()
            score = 0
            while samples < num_samples and not done:
                state = self.shared_obs_stats.normalize(state)
                #print(state)
                states.append(state.data.numpy())
                mu, log_std, v = model_old(state)
                eps = torch.randn(mu.size())
                #print(log_std.exp())
                action = (mu + log_std.exp()*Variable(eps))
                actions.append(action.data.numpy())
                values.append(v.data.numpy())
                env_action = action.data.squeeze().numpy()
                state, reward, done, _ = self.env.step(env_action)
                score += reward
                rewards.append(Variable(reward * torch.ones(1)).data.numpy())
                #q_value = self.gamma * q_value + Variable(reward * torch.ones(1))
                state = Variable(torch.Tensor(state).unsqueeze(0))

                next_state = self.shared_obs_stats.normalize(state)
                next_states.append(next_state.data.numpy())

                samples += 1
  

            state = self.shared_obs_stats.normalize(state)
            #print(state)
            _,_,v = model_old(state)
            if done:
                R = torch.zeros(1, 1)
            else:
                R = v.data
            R = Variable(R)
            for i in reversed(range(len(rewards))):
                R = self.params.gamma * R + Variable(torch.from_numpy(rewards[i]))
                q_values.insert(0, R.data.numpy())

            #self.memory.push([states, actions, next_states, rewards, q_values])
            #return [states, actions, next_states, rewards, q_values]
            self.queue.put([states, actions, next_states, rewards, q_values])
            #print(score)
            '''if score >= self.best_score.value:
                self.best_score.value = score
                print("best score", self.best_score.value)
                self.best_score_queue.put([states, actions, next_states, rewards, q_values])'''
            self.counter.increment()
            self.env.reset()
            while self.traffic_light.get() == signal_init:
                pass
            start_state = self.env.reset()
            state = start_state
            state = Variable(torch.Tensor(state).unsqueeze(0))
            total_reward = 0
            samples = 0
            done = False
            states = []
            next_states = []
            actions = []
            rewards = []
            values = []
            q_values = []
            model_old = ActorCriticNet(self.num_inputs, self.num_outputs, self.hidden_layer)
            model_old.load_state_dict(self.model.state_dict())
            model_old.set_noise(-2.0)

    def collect_expert_samples(self, num_samples, filename, noise=-2.0, speed=0, y_speed=0, validation=False):
        expert_env = cassieRLEnvMirror()
        start_state = expert_env.reset_by_speed(speed, y_speed)
        samples = 0
        done = False
        states = []
        next_states = []
        actions = []
        rewards = []
        values = []
        q_values = []
        self.model.set_noise(noise)
        model_expert = ActorCriticNet(85, 10, [256, 256])
        

        model_expert.load_state_dict(torch.load(filename))
        model_expert.set_noise(noise)

        with open('torch_model/cassie3dMirror2kHz_shared_obs_stats.pkl', 'rb') as input:
            expert_shared_obs_stats = pickle.load(input)


        state = start_state
        virtual_state = np.concatenate([np.copy(state[0:46]), np.zeros(39)])
        state = Variable(torch.Tensor(state).unsqueeze(0))
        virtual_state = Variable(torch.Tensor(virtual_state).unsqueeze(0))
        total_reward = 0
        total_sample = 0
        #q_value = Variable(torch.zeros(1, 1))
        if validation:
            max_sample = 300
        else:
            max_sample = 3000
        while total_sample < max_sample:
            model_expert.set_noise(-2.0)
            score = 0
            while samples < num_samples and not done:
                state = expert_shared_obs_stats.normalize(state)
                virtual_state = expert_shared_obs_stats.normalize(virtual_state)

                states.append(state.data.numpy())
                mu, log_std, v = model_expert(state)
                mu_residual, _, _ = residual_model(state)
                #print(log_std.exp())
                action = (mu + mu_residual * 0)
                pos_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
                vel_index = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
                ref_pos, ref_vel = expert_env.get_kin_next_state()
                saved_action = action.data.numpy() + ref_pos[pos_index]


                #actions.append(action.data.numpy())
                actions.append(action.data.numpy())
                values.append(v.data.numpy())
                eps = torch.randn(mu.size())
                if validation:
                    weight = 0.1
                else:
                    weight = 0.1
                mu = (action + weight*Variable(eps))
                env_action = mu.data.squeeze().numpy()
                
                state, reward, done, _ = expert_env.step(env_action)
                reward = 1
                rewards.append(Variable(reward * torch.ones(1)).data.numpy())
                #q_value = self.gamma * q_value + Variable(reward * torch.ones(1))
                virtual_state = np.concatenate([np.copy(state[0:46]), np.zeros(39)])
                virtual_state = Variable(torch.Tensor(virtual_state).unsqueeze(0))
                state = Variable(torch.Tensor(state).unsqueeze(0))

                next_state = expert_shared_obs_stats.normalize(state)
                next_states.append(next_state.data.numpy())

                samples += 1
                #total_sample += 1
                score += reward
            print("expert score", score)
  

            state = expert_shared_obs_stats.normalize(state)
            #print(state)
            _,_,v = model_expert(state)
            if done:
                R = torch.zeros(1, 1)
            else:
                R = v.data
                R = torch.ones(1, 1) * 100
            R = Variable(R)
            for i in reversed(range(len(rewards))):
                R = self.params.gamma * R + Variable(torch.from_numpy(rewards[i]))
                q_values.insert(0, R.data.numpy())
            
            if not validation and score >= 299:
                self.expert_trajectory.push([states, actions, next_states, rewards, q_values])
                total_sample += 300
            elif score >= 299:
                self.validation_trajectory.push([states, actions, next_states, rewards, q_values])
            expert_env.reset_by_speed(speed, y_speed)
            start_state = expert_env.reset_by_speed(speed, y_speed)
            state = start_state
            state = Variable(torch.Tensor(state).unsqueeze(0))
            total_reward = 0
            samples = 0
            done = False
            states = []
            next_states = []
            actions = []
            rewards = []
            values = []
            q_values = []

    def update_critic(self, batch_size, num_epoch):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr*10)
        model_old = ActorCriticNet(self.num_inputs, self.num_outputs, self.hidden_layer)
        model_old.load_state_dict(self.model.state_dict())
        for k in range(num_epoch):
            batch_states, batch_actions, batch_next_states, batch_rewards, batch_q_values = self.memory.sample(batch_size)
            batch_states = Variable(torch.Tensor(batch_states))
            batch_q_values = Variable(torch.Tensor(batch_q_values))
            batch_next_states = Variable(torch.Tensor(batch_next_states))
            _, _, v_pred_next = model_old(batch_next_states)
            _, _, v_pred = self.model(batch_states)
            loss_value = (v_pred - batch_q_values)**2
            #loss_value = (v_pred_next * self.params.gamma + batch_rewards - v_pred)**2
            loss_value = 0.5*torch.mean(loss_value)
            optimizer.zero_grad()
            loss_value.backward(retain_graph=True)
            optimizer.step()
            #av_value_loss = loss_value.data[0]
            #model_old.load_state_dict(model.state_dict())
        #print("value loss ", av_value_loss)

    def update_actor(self, batch_size, num_epoch, supervised=False):
        model_old = ActorCriticNet(self.num_inputs, self.num_outputs, self.hidden_layer)
        model_old.load_state_dict(self.model.state_dict())
        model_old.set_noise(self.model.noise)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        for k in range(num_epoch):
            batch_states, batch_actions, batch_next_states, batch_rewards, batch_q_values = self.memory.sample(batch_size)

            batch_states = Variable(torch.Tensor(batch_states))
            batch_q_values = Variable(torch.Tensor(batch_q_values))
            batch_actions = Variable(torch.Tensor(batch_actions))
            mu_old, log_std_old, v_pred_old = model_old(batch_states)
            #mu_old_next, log_std_old_next, v_pred_old_next = model_old(batch_next_states)
            mu, log_std, v_pred = self.model(batch_states)
            batch_advantages = batch_q_values - v_pred_old
            probs_old = normal(batch_actions, mu_old, log_std_old)
            probs = normal(batch_actions, mu, log_std)
            ratio = (probs - (probs_old)).exp()
            ratio = ratio.unsqueeze(1)
            #print(model_old.noise)
            #print(ratio)
            batch_advantages = batch_q_values - v_pred_old
            surr1 = ratio * batch_advantages
            surr2 = ratio.clamp(1-self.params.clip, 1+self.params.clip) * batch_advantages
            loss_clip = -torch.mean(torch.min(surr1, surr2))

            #expert loss
            if supervised is True:
                if k % 1000 == 999:
                    batch_expert_states, batch_expert_actions, _, _, _ = self.expert_trajectory.sample(len(self.expert_trajectory.memory))
                else:
                    batch_expert_states, batch_expert_actions, _, _, _ = self.expert_trajectory.sample(batch_size)
                batch_expert_states = Variable(torch.Tensor(batch_expert_states))
                batch_expert_actions = Variable(torch.Tensor(batch_expert_actions))
                mu_expert, _, _ = self.model(batch_expert_states)
                mu_expert_old, _, _ = model_old(batch_expert_states)
                loss_expert1 = torch.mean((batch_expert_actions-mu_expert)**2)
                clip_expert_action = torch.max(torch.min(mu_expert, mu_expert_old + 0.1), mu_expert_old-0.1)
                loss_expert2 = torch.mean((clip_expert_action-batch_expert_actions)**2)
                loss_expert = loss_expert1#torch.min(loss_expert1, loss_expert2)
            else:
                loss_expert = 0

            total_loss = 0.0*loss_clip + self.weight*loss_expert
            print(k, loss_expert)
            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            #print(torch.nn.utils.clip_grad_norm(self.model.parameters(),1))
            optimizer.step()
        if self.lr > 1e-4:
            self.lr *= 0.99
        if self.weight > 10:
            self.weight *= 0.99
        if self.weight < 10:
            self.weight = 10.0

    def validation(self):
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_q_values = self.validation_trajectory.sample(300)
        model_old = ActorCriticNet(self.num_inputs, self.num_outputs, self.hidden_layer)
        model_old.load_state_dict(self.model.state_dict())
        batch_states = Variable(torch.Tensor(batch_states))
        batch_q_values = Variable(torch.Tensor(batch_q_values))
        batch_actions = Variable(torch.Tensor(batch_actions))
        mu_old, log_std_old, v_pred_old = model_old(batch_states)
        loss = torch.mean((batch_actions-mu_old)**2)
        if loss.data < self.current_best_validation:
            self.current_best_validation = loss.data
        print("validation error", self.current_best_validation)

    def clear_memory(self):
        self.memory.clear()

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def save_shared_obs_stas(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self.shared_obs_stats, output, pickle.HIGHEST_PROTOCOL)

    def save_statistics(self, filename):
        statistics = [self.noisy_test_mean, self.noisy_test_std]
        with open(filename, 'wb') as output:
            pickle.dump(statistics, output, pickle.HIGHEST_PROTOCOL)

    def collect_samples_multithread(self):
        #queue = Queue.Queue()
        self.lr = 1e-4
        self.weight = 10
        num_threads = 10
        seeds = [
            np.random.randint(0, 4294967296) for _ in range(num_threads)
        ]

        ts = [
            mp.Process(target=self.collect_samples,args=(300,), kwargs={'noise':-2.0, 'random_seed':seed})
            for seed in seeds
        ]
        for t in ts:
            t.start()
            #print("started")
        self.model.set_noise(-2.0)
        while True:
            self.save_model("torch_model/corl_demo.pt")
            while len(self.memory.memory) < 30000:
                #print(len(self.memory.memory))
                if self.counter.get() == num_threads:
                    for i in range(num_threads):
                        self.memory.push(self.queue.get())
                    self.counter.increment()
                if len(self.memory.memory) < 30000 and self.counter.get() == num_threads + 1:
                    self.counter.reset()
                    self.traffic_light.switch()

            self.update_critic(128, 640)
            self.update_actor(128, 640, supervised=False)
            self.clear_memory()
            self.run_test(num_test=10)
            self.run_test_with_noise(num_test=10)
            #self.validation()
            self.plot_statistics()
            self.traffic_light.switch()
            self.counter.reset()

    def add_env(self, env):
        self.env_list.append(env)

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

if __name__ == '__main__':
    env = cassieRLEnvMirrorIKTraj()
    env.delay = False
    env.noisy = False
    ppo = RL(env, [256, 256])
    '''for speed in range(-5, 0):
        ppo.collect_expert_samples(300, "expert_model/SoftFootStablePelvisIKBackwardJune19Size256X256.pt", speed=speed/10.0, y_speed=0)
    for speed in range(0 ,11):
        ppo.collect_expert_samples(300, "expert_model/SoftFootForwardBackwardJune10Size256X256.pt", speed=speed/10.0, y_speed=0)'''
    with open('torch_model/cassie3dMirror2kHz_shared_obs_stats.pkl', 'rb') as input:
        ppo.shared_obs_stats = pickle.load(input)
    ppo.collect_samples_multithread()

    start = t.time()

    noise = -2.0