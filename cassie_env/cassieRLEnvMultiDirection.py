from cassiemujoco_ctypes import *
import math
from quaternion_function import *
from cassieRLEnvMultiSkill import *

class cassieRLEnvMultiDirection(cassieRLEnvMultiSkill):
	def __init__(self):
		super().__init__()
		self.speed = 0
		self.y_speed = 1
		self.observation_space = np.zeros(85)
		self.control_rate = 60
		self.max_phase = 28
		self.time_limit = 400
		#self.trajectory = CassieTrajectory("trajectory/more-poses-trial.bin")
		#print(self.sim.mjdata())
	def get_kin_state(self):
		joint_index = [7, 8, 21, 22]
		vel_index = [6, 7, 19, 20]
		if self.speed < 0:
			phase = self.max_phase - self.phase
			pose = np.copy(self.trajectory.qpos[phase*self.control_rate])
			pose[0] = self.trajectory.qpos[self.phase*self.control_rate, 0]
			pose[1] = pose[0] * self.y_speed
			pose[0] = pose[0] * self.speed
			pose[0] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter * self.speed
			pose[1] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter * self.y_speed
			vel = np.copy(self.trajectory.qvel[phase*2*30])
			vel[1] = self.y_speed * vel[0]
			vel[0] *= self.speed
		else:
			pose = np.copy(self.trajectory.qpos[self.phase*self.control_rate])
			pose[1] = pose[0] * self.y_speed
			pose[0] *= self.speed
			pose[0] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter * self.speed
			pose[1] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter * self.y_speed
			vel = np.copy(self.trajectory.qvel[self.phase*self.control_rate])
			vel[1] = self.y_speed * vel[0]
			vel[0] *= self.speed
		pose[joint_index] = 0
		vel[vel_index] = 0
		return pose, vel
	def get_kin_next_state(self):
		joint_index = [7, 8, 21, 22]
		vel_index = [6, 7, 19, 20]
		if self.speed < 0:
			phase = self.phase + 1
			phase = self.max_phase - self.phase
			if phase >= self.max_phase:
				phase = 0
			pose = np.copy(self.trajectory.qpos[phase*self.control_rate])
			pose[0] = self.trajectory.qpos[self.phase*self.control_rate, 0]
			pose[1] = pose[0] * self.y_speed
			pose[0] = pose[0] * self.speed
			pose[0] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter * self.speed
			pose[1] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter * self.y_speed
			vel = np.copy(self.trajectory.qvel[phase*self.control_rate])
			vel[1] = self.y_speed * vel[0]
			vel[0] *= self.speed
		else:
			phase = self.phase + 1
			if phase >= self.max_phase:
				phase = 0
			pose = np.copy(self.trajectory.qpos[phase*self.control_rate])
			pose[1] = pose[0] * self.y_speed
			pose[0] *= self.speed
			vel = np.copy(self.trajectory.qvel[phase*self.control_rate])
			pose[0] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter * self.speed
			pose[1] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter * self.y_speed
			pose[1] = 0
			vel[1] = self.y_speed * vel[0]
			vel[0] *= self.speed
		pose[joint_index] = 0
		vel[vel_index] = 0
		return pose, vel
	def reset(self):
		self.orientation = 0
		self.speed = (random.randint(-10, 10)) / 10.0
		self.y_speed = 0#(random.randint(0, 1)) / 2
		#print(self.speed, self.y_speed)
		orientation = self.orientation + random.randint(-10, 10) * np.pi / 100
		quaternion = euler2quat(z=orientation, y=0, x=0)
		self.phase = random.randint(0, self.max_phase - 1)
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

	def reset_by_speed(self, speed, y_speed):
		self.orientation = 0
		self.speed = speed#(random.randint(-10, 10)) / 10.0
		self.y_speed = y_speed
		orientation = self.orientation + random.randint(-10, 10) * np.pi / 100
		quaternion = euler2quat(z=orientation, y=0, x=0)
		self.phase = random.randint(0, 27)
		self.time = 0
		self.counter = 0
		cassie_sim_free(self.sim.c)
		self.sim.c = cassie_sim_init()
		qpos, qvel = self.get_kin_state()
		qpos[3:7] = quaternion
		self.sim.set_qpos(qpos)
		self.sim.set_qvel(qvel)
		self.cassie_state = self.sim.step_pd(self.u)
		#self.sim.perturb_mass()
		#self.sim.get_mass()
		#self.sim.perturb_inertia()
		return self.get_state()
