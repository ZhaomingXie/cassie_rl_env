from .cassieRLEnv import *
from cassiemujoco_ctypes import *
import math
from .quaternion_function import *

class cassieRLEnvStepInPlaceWithFootForces(cassieRLEnvDelay):
	def __init__(self):
		super().__init__()
		self.observation_space = np.zeros(87)
		self.action_space = np.zeros(10)
		self.trajectory = CassieTrajectory("trajectory/stepdata.bin")
		self.delay = True
		self.buffer_size = 20
		self.noisy = True
		self.cassie_state = state_out_t()
		self.orientation = 0
		self.foot_forces = np.ones(2) * 500
		self.max_phase = 28
		self.control_rate = 60
		self.time_limit = 400 * 60 / self.control_rate

	def step_simulation(self, action):
		qpos = np.copy(self.sim.qpos())
		qvel = np.copy(self.sim.qvel())

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

		self.state_buffer.append(self.sim.step_pd(self.u))
		if len(self.state_buffer) > self.buffer_size:
			self.state_buffer.pop(0)
		self.cassie_state = self.state_buffer[len(self.state_buffer) - 1]

	def step(self, action):
		for _ in range(self.control_rate):
			self.step_simulation(action)

		height = self.sim.qpos()[2]
		self.time += 1
		self.phase += 1

		if self.phase >= self.max_phase:
			self.phase = 0
			self.counter +=1
		#print("height", height)

		done = not(height > 0.4 and height < 100.0) or self.time >= self.time_limit
		yaw = quat2yaw(self.sim.qpos()[3:7])

		reward = self.compute_reward()
		#print(reward)
		if reward < 0.3:
			done = True

		return self.get_state(), reward, done, {}

	def get_state(self):
		state = self.cassie_state

		ref_pos, ref_vel = self.get_kin_next_state()

		pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
		vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])

		quaternion = euler2quat(z=self.orientation, y=0, x=0)
		iquaternion = inverse_quaternion(quaternion)
		new_orientation = quaternion_product(iquaternion, state.pelvis.orientation[:])
		#print(new_orientation)
		new_translationalVelocity = rotate_by_quaternion(state.pelvis.translationalVelocity[:], iquaternion)
		#print(new_translationalVelocity)
		new_translationalAcceleration = rotate_by_quaternion(state.pelvis.translationalAcceleration[:], iquaternion)
		new_rotationalVelocity = rotate_by_quaternion(state.pelvis.rotationalVelocity[:], quaternion)

		useful_state = np.copy(np.concatenate([0 * np.ones(1), [state.pelvis.position[2] - state.terrain.height], new_orientation[:], state.motor.position[:], new_translationalVelocity[:], state.pelvis.rotationalVelocity[:], state.motor.velocity[:], new_translationalAcceleration[:], state.leftFoot.toeForce[:], state.leftFoot.heelForce[:], state.rightFoot.toeForce[:], state.rightFoot.heelForce[:]]))

		return np.concatenate([useful_state, ref_pos[pos_index], ref_vel[vel_index]])

	def get_kin_state(self):
		pose = np.copy(self.trajectory.qpos[self.phase*self.control_rate])
		pose[0] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter
		pose[1] = 0
		pose[0] = 0
		vel = np.copy(self.trajectory.qvel[self.phase*self.control_rate])
		vel[0] = 0
		return pose, vel

	def get_kin_next_state(self):
		phase = self.phase + 1
		if phase >= self.max_phase:
			phase = 0
		pose = np.copy(self.trajectory.qpos[phase*self.control_rate])
		vel = np.copy(self.trajectory.qvel[phase*self.control_rate])
		pose[0] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter
		pose[1] = 0
		pose[0] = 0
		vel[0] = 0
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

		pelvis_pos = self.cassie_state.pelvis.position[:]

		com_penalty = (pelvis_pos[0])**2 + (pelvis_pos[1])**2 + (pelvis_pos[2]-ref_pos[2])**2

		yaw = quat2yaw(self.sim.qpos()[3:7])

		orientation_penalty = (self.sim.qpos()[4]*20)**2+(self.sim.qpos()[5]*20)**2+(yaw - self.orientation)**2

		spring_penalty = (self.sim.qpos()[15])**2+(self.sim.qpos()[29])**2
		spring_penalty *= 1000

		total_reward = 0.5*np.exp(-joint_penalty)+0.3*np.exp(-com_penalty)+0.1*np.exp(-orientation_penalty)+0.1*np.exp(-spring_penalty)

		return total_reward

	def reset(self):
		self.orientation = 0 * np.pi
		orientation = self.orientation + random.randint(-20, 20) * np.pi / 100
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
		return self.get_state()

	def reset_for_normalization(self):
		return self.reset()

	def reset_for_test(self):
		self.orientation = 0.0*np.pi
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

	def set_state(self, qpos, qvel):
		self.sim.set_qpos(qpos)
		self.sim.set_qvel(qvel)

class cassieRLEnvWithFootForces(cassieRLEnvStepInPlaceWithFootForces):
	def __init__(self):
		super().__init__()
		self.speed = 1.0
		#self.observation_space = np.zeros(87)

	def get_kin_state(self):
		pose = np.copy(self.trajectory.qpos[self.phase*self.control_rate])
		pose[0] *= self.speed
		pose[0] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter * self.speed
		pose[1] = 0
		vel = np.copy(self.trajectory.qvel[self.phase*self.control_rate])
		vel[0] *= self.speed
		return pose, vel

	def get_kin_next_state(self):
		phase = self.phase + 1
		if phase >= 28:
			phase = 0
		pose = np.copy(self.trajectory.qpos[phase*self.control_rate])
		pose[0] *= self.speed
		vel = np.copy(self.trajectory.qvel[phase*self.control_rate])
		pose[0] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter * self.speed
		pose[1] = 0
		vel[0] *= self.speed
		return pose, vel

	def reset(self):
		self.orientation = 0
		self.speed = random.randint(0, 10) / 10
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
		return self.get_state()

	def reset_for_normalization(self):
		return self.reset()

	def reset_for_test(self):
		self.speed = random.randint(0, 10) / 10
		self.orientation = 0
		orientation = self.orientation + random.randint(-10, 10) * np.pi / 100
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

	def compute_reward(self):
		ref_pos, ref_vel = self.get_kin_state()
		weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]
		joint_penalty = 0


		joint_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
		vel_index = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

		for i in range(10):
			error = weight[i] * (ref_pos[joint_index[i]]-self.sim.qpos()[joint_index[i]])**2
			joint_penalty += error*30

		pelvis_pos = self.cassie_state.pelvis.position[:]

		desired_x = ref_pos[0]*np.cos(self.orientation)
		desired_y = ref_pos[1]*np.sin(self.orientation)

		com_penalty = (pelvis_pos[0]-desired_x)**2 + (pelvis_pos[1]-desired_y)**2 + (pelvis_pos[2]-ref_pos[2])**2
		#print("x", pelvis_pos[0]-desired_x, "y", pelvis_pos[1]-desired_y)

		yaw = quat2yaw(self.sim.qpos()[3:7])

		orientation_penalty = (self.sim.qpos()[4])**2+(self.sim.qpos()[5])**2+(yaw - self.orientation)**2

		spring_penalty = (self.sim.qpos()[15])**2+(self.sim.qpos()[29])**2
		spring_penalty *= 1000

		total_reward = 0.5*np.exp(-joint_penalty)+0.3*np.exp(-com_penalty)+0.1*np.exp(-30*orientation_penalty)+0.1*np.exp(-force_penalty)#0.1*np.exp(-force_penalty)

		return total_reward