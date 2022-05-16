from cassiemujoco import *
from cassie_env.cassieRLEnvMirror import cassieRLEnvMirror
from cassie_env.cassieRLEnvMirrorIKTraj import cassieRLEnvMirrorIKTraj

import time as t

import statistics

import argparse
import os
import sys
import gym
from gym import wrappers
import random
import numpy as np

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data

import pickle
from model import ActorCriticNet, Shared_obs_stats

env = cassieRLEnvMirror()

u = pd_in_t()
u.leftLeg.motorPd.torque[3] = 0 # Feedforward torque
u.leftLeg.motorPd.pTarget[3] = -2
u.leftLeg.motorPd.pGain[3] = 1000
u.leftLeg.motorPd.dTarget[3] = -2
u.leftLeg.motorPd.dGain[3] = 100
u.rightLeg.motorPd = u.leftLeg.motorPd

num_inputs = env.observation_space.shape[0]
num_outputs = env.action_space.shape[0]

model = ActorCriticNet(num_inputs, num_outputs,[256, 256])
model.load_state_dict(torch.load("torch_model/StablePelvisForwardBackward256X256Jan25.pt"))
model.cuda()
# model.load_state_dict(torch.load("torch_model/corl_demo.pt"))
with open('torch_model/cassie3dMirror2kHz_shared_obs_stats.pkl', 'rb') as input:
	shared_obs_stats = pickle.load(input)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
	shared_obs_stats.to_cuda()

state_list = []
env.visualize = True

def run_test():
	t.sleep(1)
	state = env.reset()
	total_reward = 0
	done = False
	total_10_reward = 0
	current_scale = 0
	clock = 0
	done = False
	current_phase = 0
	reward_list = []

	for i in range(10):
		state = env.reset()
		#env.speed = -1.0
		total_reward = 0
		counter = 0
		interpolate = 1
		while counter < 100000 and not done:
			start = t.time()
			for j in range(1):
				counter += 1
				clock += 1
				state = torch.Tensor(state).unsqueeze(0).to(device)
				state = shared_obs_stats.normalize(state)
				mu, log_std, v = model(state)
				eps = torch.randn(mu.size())
				env_action = mu.cpu().data.squeeze().numpy() + eps.cpu().data.squeeze().numpy() * 0.1
				state, reward, done, _ = env.step(env_action)
				env.vis.draw(env.sim)
				total_reward += reward
				force = np.zeros(12)
				pos = np.zeros(6)
			while True:
				stop = t.time()
				if stop - start > 0.03 * env.control_rate / 60:
					break
		done = False
		counter = 0
		reward_list.append(total_reward)
		total_10_reward += total_reward
		print("total rewards", total_reward)
	print(total_10_reward)
	print(statistics.mean(reward_list))
	print(statistics.stdev(reward_list))

def play_kin():
	env.vis.draw(env.sim)
	env.phase = 0
	env.reset()
	env.phase = 0
	t.sleep(1)
	while True:
		start = t.time()
		while True:
			stop = t.time()
			if stop - start > 0.033:
				break
		pos, vel = env.get_kin_next_state()
		if env.phase == 6:
			print(pos[[7, 8, 9, 14, 20, 21, 22, 23, 28, 34]])
			print(vel)
			print(env.phase)
		env.phase += 1
		#print(env.speed)
		if env.phase >= env.max_phase:
			env.phase = 0
			env.counter += 1
			#break
		env.set_state(pos, vel)
		y = env.sim.step_pd(u)
		env.vis.draw(env.sim)

run_test()
#play_kin()