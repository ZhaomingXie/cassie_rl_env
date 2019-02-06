from cassiemujoco_ctypes import *
import math
from .quaternion_function import *
from .cassieRLEnvMultiSkill import *

class cassieRLEnvMultiTraj(cassieRLEnvMultiSkill):
	def __init__(self):
		super().__init__()
		self.speed = 0
		self.observation_space = np.zeros(85)
		joint_index = [7, 8, 21, 22]
		vel_index = [6, 7, 19, 20]
		with open ("trajectory/stepping_trajectory_Nov", 'rb') as fp:
			self.step_in_place_trajectory = pickle.load(fp)
		with open("trajectory/backward_trajectory_Nov", "rb") as fp:
			self.backward_trajectory = pickle.load(fp)

		for i in range(1682):
			self.step_in_place_trajectory[i][0][0] = 0
			self.step_in_place_trajectory[i][0][1] = 0
			self.step_in_place_trajectory[i][0][2] = 1.05
			self.step_in_place_trajectory[i][0][3] = 1
			self.step_in_place_trajectory[i][0][4] = 0
			self.step_in_place_trajectory[i][0][5] = 0
			self.step_in_place_trajectory[i][0][6] = 0
			self.step_in_place_trajectory[i][0][7] = 0
			self.step_in_place_trajectory[i][0][8] = 0
			self.step_in_place_trajectory[i][0][21] = 0
			self.step_in_place_trajectory[i][0][22] = 0
			self.step_in_place_trajectory[i][1][6] = 0
			self.step_in_place_trajectory[i][1][7] = 0
			self.step_in_place_trajectory[i][1][19] = 0
			self.step_in_place_trajectory[i][1][20] = 0
			self.backward_trajectory[i][1][3] = 0
			self.backward_trajectory[i][1][4] = 0
			self.backward_trajectory[i][1][5] = 0
			self.backward_trajectory[i][0][0] = self.trajectory.qpos[i][0]*-1
			self.backward_trajectory[i][0][1] = 0
			self.backward_trajectory[i][0][2] = 1.05
			self.backward_trajectory[i][0][3] = 1
			self.backward_trajectory[i][0][4] = 0
			self.backward_trajectory[i][0][5] = 0
			self.backward_trajectory[i][0][6] = 0
			self.backward_trajectory[i][0][7] = 0
			self.backward_trajectory[i][0][8] = 0
			self.backward_trajectory[i][0][21] = 0
			self.backward_trajectory[i][0][22] = 0
			self.backward_trajectory[i][1][6] = 0
			self.backward_trajectory[i][1][7] = 0
			self.backward_trajectory[i][1][19] = 0
			self.backward_trajectory[i][1][20] = 0
			self.backward_trajectory[i][1][3] = 0
			self.backward_trajectory[i][1][4] = 0
			self.backward_trajectory[i][1][5] = 0
			self.trajectory.qpos[i][2] = 1.05

		for i in range(841):
			self.backward_trajectory[i][0][7:21] = np.copy(self.backward_trajectory[i+841][0][21:35])
			self.backward_trajectory[i][0][12] = -self.backward_trajectory[i][0][12]
			self.backward_trajectory[i][0][21:35] = np.copy(self.backward_trajectory[i+841][0][7:21])
			self.backward_trajectory[i][0][26] = -self.backward_trajectory[i][0][26]

			self.step_in_place_trajectory[i][0][7:21] = np.copy(self.step_in_place_trajectory[i+841][0][21:35])
			self.step_in_place_trajectory[i][0][12] = -self.step_in_place_trajectory[i][0][12]
			self.step_in_place_trajectory[i][0][21:35] = np.copy(self.step_in_place_trajectory[i+841][0][7:21])
			self.step_in_place_trajectory[i][0][26] = -self.step_in_place_trajectory[i][0][26]

			self.trajectory.qpos[i][7:21] = np.copy(self.trajectory.qpos[(i+841)][21:35])
			self.trajectory.qpos[i][12] = -self.trajectory.qpos[i][12]
			self.trajectory.qpos[i][21:35] = np.copy(self.trajectory.qpos[(i+841)][7:21])
			self.trajectory.qpos[i][26] = -self.trajectory.qpos[i][26]
	def get_kin_state(self):
		if self.speed < 0:
			interpolate = self.speed * -1
			phase = self.phase
			pose = np.copy(self.backward_trajectory[phase*self.control_rate][0]) * interpolate + (1 - interpolate) * np.copy(self.step_in_place_trajectory[phase*self.control_rate][0])
			pose[0] += (self.backward_trajectory[(self.max_phase-1)*self.control_rate][0][0]- self.backward_trajectory[0][0][0])* self.counter * (-self.speed)
			pose[1] = 0
			vel = np.copy(self.backward_trajectory[phase*self.control_rate][1])
			vel[0] *= (-self.speed)
		elif self.speed <= 1.0:
			interpolate = self.speed * 1
			pose = np.copy(self.trajectory.qpos[self.phase*self.control_rate]) * interpolate + (1 - interpolate) * np.copy(self.step_in_place_trajectory[self.phase*self.control_rate][0])
			pose[0] = self.trajectory.qpos[self.phase*self.control_rate, 0]
			pose[0] *= self.speed
			pose[0] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter * self.speed
			pose[1] = 0
			vel = np.copy(self.trajectory.qvel[self.phase*self.control_rate])
			vel[0] *= self.speed
		else:
			pose = np.copy(self.trajectory.qpos[self.phase*self.control_rate])
			pose[0] = self.trajectory.qpos[self.phase*self.control_rate, 0]
			pose[0] *= self.speed
			pose[0] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter * self.speed
			pose[1] = 0
			vel = np.copy(self.trajectory.qvel[self.phase*self.control_rate])
			vel[0] *= self.speed
		#print("vel", vel[0])
		pose[3] = 1
		pose[4:7] = 0
		pose[7] = 0
		pose[8] = 0
		pose[21] = 0
		pose[22] = 0
		vel[6] = 0
		vel[7] = 0
		vel[19] = 0
		vel[20] = 0
		return pose, vel
	def get_kin_next_state(self):
		if self.speed < 0:
			phase = self.phase + 1
			if phase >= self.max_phase:
				phase = 0
			phase = self.max_phase - phase
			pose = np.copy(self.trajectory.qpos[phase*self.control_rate])
			vel = np.copy(self.trajectory.qvel[phase*self.control_rate])
			pose[0] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter * self.speed
			#print(pose[0])
			pose[1] = 0
			vel[0] *= self.speed
		else:
			phase = self.phase + 1
			if phase >= self.max_phase:
				phase = 0
			pose = np.copy(self.trajectory.qpos[phase*self.control_rate])
			pose[0] *= self.speed
			vel = np.copy(self.trajectory.qvel[phase*self.control_rate])
			pose[0] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter * self.speed
			pose[1] = 0
			vel[0] *= self.speed
		pose[3] = 1
		pose[4:7] = 0
		pose[7] = 0
		pose[8] = 0
		pose[21] = 0
		pose[22] = 0
		vel[6] = 0
		vel[7] = 0
		vel[19] = 0
		vel[20] = 0
		return pose, vel
	def reset(self):
		self.orientation = 0
		self.speed = (random.randint(-1, 1)) / 1
		orientation = self.orientation + random.randint(-20, 20) * np.pi / 100
		quaternion = euler2quat(z=orientation, y=0, x=0)
		self.phase = random.randint(0, self.max_phase-1)
		self.time = 0
		self.counter = 0
		cassie_sim_free(self.sim.c)
		self.sim.c = cassie_sim_init()
		qpos, qvel = self.get_kin_state()
		qpos[3:7] = quaternion
		self.sim.set_qpos(qpos)
		self.sim.set_qvel(qvel)
		self.cassie_state = self.sim.step_pd(self.u)
		return self.get_state()

	def reset_for_normalization(self):
		return self.reset()

	def reset_for_test(self):
		return self.reset()