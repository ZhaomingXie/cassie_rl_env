from cassiemujoco import *
from .loadstep import CassieTrajectory
import numpy as np
import os
import random
import pickle

class cassieRLEnv:
	def __init__(self):
		self.sim = CassieSim()
		self.vis = CassieVis()
		self.observation_space = np.zeros(80)
		self.action_space = np.zeros(10)
		self.trajectory = CassieTrajectory("trajectory/stepdata.bin")
		self.P = np.array([100, 100, 88, 96, 50, 100, 100, 88, 96, 50])
		self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0, 10.0, 10.0, 8.0, 9.6, 5.0])
		self.u = pd_in_t()
		self.time = 0
		self.phase = 0
		self.counter = 0
		self.time_limit = 400

	def get_state(self):
		qpos = np.copy(self.sim.qpos())
		qvel = np.copy(self.sim.qvel())

		ref_pos, ref_vel = self.get_kin_next_state()

		'''
		Position [1], [2] 				-> Pelvis y, z
				 [3], [4], [5], [6] 	-> Pelvis Orientation qw, qx, qy, qz
				 [7], [8], [9]			-> Left Hip Roll (Motor[0]), Yaw (Motor[1]), Pitch (Motor[2])
				 [14]					-> Left Knee   	(Motor[3])
				 [15]					-> Left Shin   	(Joint[0])
				 [16]					-> Left Tarsus 	(Joint[1])
				 [20]					-> Left Foot   	(Motor[4], Joint[2])
				 [21], [22], [23]		-> Right Hip Roll (Motor[5]), Yaw (Motor[6]), Pitch (Motor[7])
				 [28]					-> Rigt Knee   	(Motor[8])
				 [29]					-> Rigt Shin   	(Joint[3])
				 [30]					-> Rigt Tarsus 	(Joint[4])
				 [34]					-> Rigt Foot   	(Motor[9], Joint[5])
		'''
		pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])

		'''
		Velocity [0], [1], [2] 			-> Pelvis x, y, z
				 [3], [4], [5]		 	-> Pelvis Orientation wx, wy, wz
				 [6], [7], [8]			-> Left Hip Roll (Motor[0]), Yaw (Motor[1]), Pitch (Motor[2])
				 [12]					-> Left Knee   	(Motor[3])
				 [13]					-> Left Shin   	(Joint[0])
				 [14]					-> Left Tarsus 	(Joint[1])
				 [18]					-> Left Foot   	(Motor[4], Joint[2])
				 [19], [20], [21]		-> Right Hip Roll (Motor[5]), Yaw (Motor[6]), Pitch (Motor[7])
				 [25]					-> Rigt Knee   	(Motor[8])
				 [26]					-> Rigt Shin   	(Joint[3])
				 [27]					-> Rigt Tarsus 	(Joint[4])
				 [31]					-> Rigt Foot   	(Motor[9], Joint[5])
		'''
		vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])

		return np.concatenate([qpos[pos_index], qvel[vel_index], ref_pos[pos_index], ref_vel[vel_index]])

	def step_simulation(self, action):
		qpos = np.copy(self.sim.qpos())
		qvel = np.copy(self.sim.qvel())

		'''
		Position [7], [8], [9]			-> Left Hip Roll (Motor[0]), Yaw (Motor[1]), Pitch (Motor[2])
				 [14]					-> Left Knee   	(Motor[3])
				 [20]					-> Left Foot   	(Motor[4], Joint[2])
				 [21], [22], [23]		-> Right Hip Roll (Motor[5]), Yaw (Motor[6]), Pitch (Motor[7])
				 [28]					-> Rigt Knee   	(Motor[8])
				 [34]					-> Rigt Foot   	(Motor[9], Joint[5])
		'''
		pos_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]

		'''
		Velocity [6], [7], [8]			-> Left Hip Roll (Motor[0]), Yaw (Motor[1]), Pitch (Motor[2])
				 [12]					-> Left Knee   	(Motor[3])
				 [18]					-> Left Foot   	(Motor[4], Joint[2])
				 [19], [20], [21]		-> Right Hip Roll (Motor[5]), Yaw (Motor[6]), Pitch (Motor[7])
				 [25]					-> Rigt Knee   	(Motor[8])
				 [31]					-> Rigt Foot   	(Motor[9], Joint[5])
		'''
		vel_index = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

		ref_pos, ref_vel = self.get_kin_next_state()
		target = action + ref_pos[pos_index]
		#control = self.P*(target-self.sim.data.qpos[index])-self.D*self.sim.data.qvel[vel_index]

		self.u = pd_in_t()
		for i in range(5):
			self.u.leftLeg.motorPd.torque[i] = 0 # Feedforward torque
			self.u.leftLeg.motorPd.pTarget[i] = target[i]
			self.u.leftLeg.motorPd.pGain[i] = self.P[i]
			self.u.leftLeg.motorPd.dTarget[i] = 0
			self.u.leftLeg.motorPd.dGain[i] = self.D[i]
			self.u.rightLeg.motorPd.torque[i] = 0 # Feedforward torque
			self.u.rightLeg.motorPd.pTarget[i] = target[i+5]
			self.u.rightLeg.motorPd.pGain[i] = self.P[i+5]
			self.u.rightLeg.motorPd.dTarget[i] = 0
			self.u.rightLeg.motorPd.dGain[i] = self.D[i+5]
		self.sim.step_pd(self.u)

	def ref_action(self):
		ref_pos, ref_vel = self.get_kin_next_state()
		pos_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
		return ref_pos[pos_index]

	def step(self, action):
		for _ in range(60):
			self.step_simulation(action)

		height = self.sim.qpos()[2]
		self.time += 1
		self.phase += 1

		if self.phase >= 28:
			self.phase = 0
			self.counter +=1

		done = not(height > 0.4 and height < 3.0) or self.time >= self.time_limit

		reward = self.compute_reward()
		if reward < 0.3:
			done = True

		return self.get_state(), reward, done, {}

	def get_kin_state(self):
		pose = np.copy(self.trajectory.qpos[self.phase*2*30])
		pose[0] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter
		pose[1] = 0
		vel = np.copy(self.trajectory.qvel[self.phase*2*30])
		return pose, vel

	def get_kin_next_state(self):
		phase = self.phase + 1
		if phase >= 28:
			phase = 0
		pose = np.copy(self.trajectory.qpos[phase*2*30])
		vel = np.copy(self.trajectory.qvel[phase*2*30])
		pose[0] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter
		pose[1] = 0
		return pose, vel

	def compute_reward(self):
		ref_pos, ref_vel = self.get_kin_state()
		weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]
		joint_penalty = 0


		joint_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
		vel_index = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

		for i in range(10):
			error = weight[i] * (ref_pos[joint_index[i]]-self.sim.qpos()[joint_index[i]])**2
			joint_penalty += error*30

		com_penalty = (ref_pos[0] - self.sim.qpos()[0])**2 + (self.sim.qpos()[1])**2 + (self.sim.qpos()[2]-ref_pos[2])**2

		orientation_penalty = (self.sim.qpos()[4])**2+(self.sim.qpos()[5])**2+(self.sim.qpos()[6])**2

		spring_penalty = (self.sim.qpos()[15])**2+(self.sim.qpos()[29])**2
		spring_penalty *= 1000

		total_reward = 0.5*np.exp(-joint_penalty)+0.3*np.exp(-com_penalty)+0.1*np.exp(-orientation_penalty)+0.1*np.exp(-spring_penalty)

		return total_reward

	def reset(self):
		self.phase = random.randint(0, 27)
		self.time = 0
		self.counter = 0
		qpos, qvel = self.get_kin_state()
		self.sim.set_qpos(qpos)
		self.sim.set_qvel(qvel)
		return self.get_state()

	def reset_for_normalization(self):
		return self.reset()

	def reset_for_test(self):
		return self.reset()

	def set_state(self, qpos, qvel):
		self.sim.set_qpos(qpos)
		self.sim.set_qvel(qvel)

	def record_state(self, state_list):
		state_list.append((self.sim.qpos(), self.sim.qvel()))

class cassieRLEnvDelay(cassieRLEnv):
	def __init__(self):
		self.sim = CassieSim()
		self.vis = CassieVis()
		self.observation_space = np.zeros(80)
		self.action_space = np.zeros(10)
		self.trajectory = CassieTrajectory("trajectory/stepdata.bin")
		self.P = np.array([100, 100, 88, 96, 50, 100, 100, 88, 96, 50])
		self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0, 10.0, 10.0, 8.0, 9.6, 5.0])
		self.u = pd_in_t()
		self.time = 0
		self.phase = 0
		self.counter = 0
		self.time_limit = 400
		self.state_buffer = []
		self.delay = True
		self.buffer_size = 120
		self.noisy = True

	def step_simulation(self, action):
		qpos = np.copy(self.sim.qpos())
		qvel = np.copy(self.sim.qvel())

		self.state_buffer.append((qpos, qvel))
		if len(self.state_buffer) > self.buffer_size:
			self.state_buffer.pop(0)

		pos_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
		vel_index = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

		ref_pos, ref_vel = self.get_kin_next_state()
		target = action + ref_pos[pos_index]

		self.u = pd_in_t()
		for i in range(5):
			self.u.leftLeg.motorPd.torque[i] = 0 # Feedforward torque
			self.u.leftLeg.motorPd.pTarget[i] = target[i]
			self.u.leftLeg.motorPd.pGain[i] = self.P[i]
			self.u.leftLeg.motorPd.dTarget[i] = 0
			self.u.leftLeg.motorPd.dGain[i] = self.D[i]
			self.u.rightLeg.motorPd.torque[i] = 0 # Feedforward torque
			self.u.rightLeg.motorPd.pTarget[i] = target[i+5]
			self.u.rightLeg.motorPd.pGain[i] = self.P[i+5]
			self.u.rightLeg.motorPd.dTarget[i] = 0
			self.u.rightLeg.motorPd.dGain[i] = self.D[i+5]
		self.sim.step_pd(self.u)

	def get_state(self):
		if len(self.state_buffer) >= self.buffer_size and self.delay:
			random_index = random.randint(0, 20)
			state = self.state_buffer[random_index]
			qpos = np.copy(state[0])
			qvel = np.copy(state[1])
			#print(random_index)
		else:
			qpos = np.copy(self.sim.qpos())
			qvel = np.copy(self.sim.qvel())

		if self.noisy:
			qpos += np.random.normal(size=35)*0.01
			qvel += np.random.normal(size=32)*0.01

		ref_pos, ref_vel = self.get_kin_next_state()

		pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
		vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])

		return np.concatenate([qpos[pos_index], qvel[vel_index], ref_pos[pos_index], ref_vel[vel_index]])


class cassieRLEnvStepInPlace(cassieRLEnvDelay):
	def __init__(self):
		self.sim = CassieSim()
		self.vis = CassieVis()
		self.observation_space = np.zeros(80)
		self.action_space = np.zeros(10)
		#self.trajectory = CassieTrajectory("trajectory/stepdata.bin")
		#self.trajectory = "step_in_place_trajectory"
		with open ("step_in_place_trajectory", 'rb') as fp:
			self.trajectory = pickle.load(fp)
		self.P = np.array([100, 100, 88, 96, 50, 100, 100, 88, 96, 50])
		self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0, 10.0, 10.0, 8.0, 9.6, 5.0])
		self.u = pd_in_t()
		self.time = 0
		self.phase = 0
		self.counter = 0
		self.time_limit = 400
		self.state_buffer = []
		self.buffer_size = 150
		self.delay = True

	def step_simulation(self, action):
		qpos = np.copy(self.sim.qpos())
		qvel = np.copy(self.sim.qvel())

		self.state_buffer.append((qpos, qvel))
		if len(self.state_buffer) > self.buffer_size:
			self.state_buffer.pop(0)

		pos_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
		vel_index = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

		ref_pos, ref_vel = self.get_kin_next_state()
		target = action + ref_pos[pos_index]

		self.u = pd_in_t()
		for i in range(5):
			self.u.leftLeg.motorPd.torque[i] = 0 # Feedforward torque
			self.u.leftLeg.motorPd.pTarget[i] = target[i]
			self.u.leftLeg.motorPd.pGain[i] = self.P[i]
			self.u.leftLeg.motorPd.dTarget[i] = 0
			self.u.leftLeg.motorPd.dGain[i] = self.D[i]
			self.u.rightLeg.motorPd.torque[i] = 0 # Feedforward torque
			self.u.rightLeg.motorPd.pTarget[i] = target[i+5]
			self.u.rightLeg.motorPd.pGain[i] = self.P[i+5]
			self.u.rightLeg.motorPd.dTarget[i] = 0
			self.u.rightLeg.motorPd.dGain[i] = self.D[i+5]
		self.sim.step_pd(self.u)

	def get_state(self):
		if len(self.state_buffer) >= 80 and self.delay:
			random_index = random.randint(0, 20)
			state = self.state_buffer[random_index]
			qpos = np.copy(state[0])
			qvel = np.copy(state[1])
			#print(random_index)
		else:
			qpos = np.copy(self.sim.qpos())
			qvel = np.copy(self.sim.qvel())

		ref_pos, ref_vel = self.get_kin_next_state()

		pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
		vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])

		return np.concatenate([qpos[pos_index], qvel[vel_index], ref_pos[pos_index], ref_vel[vel_index]])

	def get_kin_state(self):
		pose = np.copy(self.trajectory[self.phase][0])
		vel = np.copy(self.trajectory[self.phase][1])
		return pose, vel

	def get_kin_next_state(self):
		phase = self.phase + 1
		if phase >= 28:
			phase = 0
		pose = np.copy(self.trajectory[self.phase][0])
		vel = np.copy(self.trajectory[self.phase][1])
		return pose, vel


class cassieRLEnvSpeed(cassieRLEnvDelay):
	def __init__(self):
		super().__init__()
		self.speed = 0
		self.observation_space = np.zeros(81)
		self.delay = True
		self.buffer_size = 10
		self.noisy = True

	def set_speed(self, speed):
		self.speed = speed

	def get_state(self):
		if len(self.state_buffer) >= self.buffer_size and self.delay:
			random_index = random.randint(0, 5)
			state = self.state_buffer[random_index]
			qpos = np.copy(state[0])
			qvel = np.copy(state[1])
			#print(random_index)
		else:
			qpos = np.copy(self.sim.qpos())
			qvel = np.copy(self.sim.qvel())

		if self.noisy:
			qpos += np.random.normal(size=35)*0.01
			qvel += np.random.normal(size=32)*0.01

		ref_pos, ref_vel = self.get_kin_next_state()

		pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
		vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])

		return np.concatenate([self.speed*np.ones(1), qpos[pos_index], qvel[vel_index], ref_pos[pos_index], ref_vel[vel_index]])

	def get_kin_state(self):
		pose = np.copy(self.trajectory.qpos[self.phase*2*30])
		pose[0] *= self.speed
		pose[0] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter * self.speed
		pose[1] = 0

		vel = np.copy(self.trajectory.qvel[self.phase*2*30])
		vel[0] * self.speed
		return pose, vel

	def get_kin_next_state(self):
		phase = self.phase + 1
		if phase >= 28:
			phase = 0
		pose = np.copy(self.trajectory.qpos[phase*2*30])
		pose[0] *= self.speed
		vel = np.copy(self.trajectory.qvel[phase*2*30])
		pose[0] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter * self.speed
		pose[1] = 0
		vel[0] *= self.speed
		return pose, vel

	def reset(self):
		self.phase = random.randint(0, 27)
		self.speed = random.randint(0, 10) / 10.0
		self.speed = 0
		self.time = 0
		self.counter = 0
		qpos, qvel = self.get_kin_state()
		self.sim.set_qpos(qpos)
		self.sim.set_qvel(qvel)
		return self.get_state()

	def reset_for_normalization(self):
		return self.reset()

	def reset_for_test(self):
		return self.reset()

	def compute_reward(self):
		ref_pos, ref_vel = self.get_kin_state()
		weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]
		joint_penalty = 0


		joint_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
		vel_index = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

		for i in range(10):
			error = weight[i] * (ref_pos[joint_index[i]]-self.sim.qpos()[joint_index[i]])**2
			if i == 7 or i == 8 or i == 21 or i == 22:
				joint_penalty += error*100
			else:
				joint_penalty += error*30

		com_penalty = (ref_pos[0] - self.sim.qpos()[0])**2 + (self.sim.qpos()[1])**2 + (self.sim.qpos()[2]-ref_pos[2])**2

		orientation_penalty = (self.sim.qpos()[4])**2+(self.sim.qpos()[5])**2+(self.sim.qpos()[6])**2

		spring_penalty = (self.sim.qpos()[15])**2+(self.sim.qpos()[29])**2
		spring_penalty *= 1000

		total_reward = 0.3*np.exp(-joint_penalty)+0.5*np.exp(-com_penalty)+0.1*np.exp(-orientation_penalty)+0.1*np.exp(-spring_penalty)

		return total_reward