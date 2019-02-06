from .cassieRLEnvWithMoreState import *
from cassiemujoco_ctypes import *
import math
from .quaternion_function import *

class cassieRLEnvMultiSkill(cassieRLEnvWithFootForces):
	def __init__(self):
		super().__init__()
		self.speed = 0
		self.observation_space = np.zeros(85)
	def get_kin_state(self):
		if self.speed < 0:
			phase = self.max_phase - self.phase
			pose = np.copy(self.trajectory.qpos[phase*self.control_rate])
			pose[0] = self.trajectory.qpos[self.phase*self.control_rate, 0]
			pose[0] = pose[0] * self.speed
			pose[0] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter * self.speed
			pose[1] = 0
			vel = np.copy(self.trajectory.qvel[phase*self.control_rate])
			vel[0] *= self.speed
		else:
			pose = np.copy(self.trajectory.qpos[self.phase*self.control_rate])
			pose[0] *= self.speed
			pose[0] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter * self.speed
			pose[1] = 0
			vel = np.copy(self.trajectory.qvel[self.phase*self.control_rate])
			vel[0] *= self.speed
		return pose, vel
	def get_kin_next_state(self):
		if self.speed < 0:
			phase = self.phase + 1
			if phase >= self.max_phase:
				phase = 0
			phase = self.max_phase - phase
			pose = np.copy(self.trajectory.qpos[phase*2*30])
			vel = np.copy(self.trajectory.qvel[phase*2*30])
			pose[0] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter * self.speed
			#print(pose[0])
			pose[1] = 0
			vel[0] *= self.speed
		else:
			phase = self.phase + 1
			if phase >= self.max_phase:
				phase = 0
			pose = np.copy(self.trajectory.qpos[phase*2*30])
			pose[0] *= self.speed
			vel = np.copy(self.trajectory.qvel[phase*2*30])
			pose[0] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter * self.speed
			pose[1] = 0
			vel[0] *= self.speed
		return pose, vel
	def reset(self):
		self.orientation = 0
		self.speed = (random.randint(0, 10)) / 10
		orientation = self.orientation# + random.randint(-10, 10) * np.pi / 100
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
		self.speed = (random.randint(0, 10)) / 10
		self.orientation = 0
		orientation = self.orientation# + random.randint(-10, 10) * np.pi / 100
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

	def get_state(self):
		if len(self.state_buffer) > 0:
			random_index = random.randint(len(self.state_buffer)-10, len(self.state_buffer)-1)
			state = self.state_buffer[random_index]
		else:
			state = self.cassie_state

		ref_pos, ref_vel = self.get_kin_next_state()

		pos_index = np.array([2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
		vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])
		quaternion = euler2quat(z=self.orientation, y=0, x=0)
		iquaternion = inverse_quaternion(quaternion)
		new_orientation = quaternion_product(iquaternion, state.pelvis.orientation[:])
		#print(new_orientation)
		new_translationalVelocity = rotate_by_quaternion(state.pelvis.translationalVelocity[:], iquaternion)
		#print(new_translationalVelocity)
		new_translationalAcceleration = rotate_by_quaternion(state.pelvis.translationalAcceleration[:], iquaternion)
		new_rotationalVelocity = rotate_by_quaternion(state.pelvis.rotationalVelocity[:], quaternion)
		useful_state = np.copy(np.concatenate([[state.pelvis.position[2] - state.terrain.height], new_orientation[:], state.motor.position[:], new_translationalVelocity[:], state.pelvis.rotationalVelocity[:], state.motor.velocity[:], new_translationalAcceleration[:], state.leftFoot.toeForce[:], state.leftFoot.heelForce[:], state.rightFoot.toeForce[:], state.rightFoot.heelForce[:]]))
		return np.concatenate([useful_state, ref_pos[pos_index], ref_vel[vel_index]])
	def compute_reward(self):
		ref_pos, ref_vel = self.get_kin_state()
		weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]
		joint_penalty = 0
		joint_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
		vel_index = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

		for i in range(10):
			error = weight[i] * (ref_pos[joint_index[i]]-self.sim.qpos()[joint_index[i]])**2
			joint_penalty += error*30

		pelvis_pos = np.copy(self.cassie_state.pelvis.position[:])
		com_penalty = (pelvis_pos[0]-ref_pos[0])**2 + (pelvis_pos[1]-ref_pos[1])**2 + (self.sim.qvel()[2])**2

		yaw = quat2yaw(self.sim.qpos()[3:7])

		orientation_penalty = (self.sim.qpos()[4])**2+(self.sim.qpos()[5])**2+(yaw - self.orientation)**2

		spring_penalty = (self.sim.qpos()[15])**2+(self.sim.qpos()[29])**2
		spring_penalty *= 1000

		speed_penalty = (self.sim.qvel()[0] - ref_vel[0])**2 + (self.sim.qvel()[1] - ref_vel[1])**2

		total_reward = 0.5*np.exp(-joint_penalty)+0.3*np.exp(-com_penalty)+0.2*np.exp(-10*orientation_penalty)#+0.1*np.exp(-spring_penalty)

		return total_reward