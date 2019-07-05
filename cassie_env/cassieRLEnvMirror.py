import math
import pickle
from .quaternion_function import *
from .cassieRLEnvMultiTraj import *

class cassieRLEnvMirror(cassieRLEnvMultiTraj):
	def __init__(self):
		super().__init__()
		self.record_state = False
		self.recorded = False
		self.recorded_state = []
		self.max_phase = 28
		self.control_rate = 60
		self.time_limit = 400
		self.P = np.array([100, 100, 88, 96, 50, 100, 100, 88, 96, 50])
		self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0, 10.0, 10.0, 8.0, 9.6, 5.0])

	def get_state(self):
		if len(self.state_buffer) > 0:
			random_index = random.randint(0, len(self.state_buffer)-1)
			state = self.state_buffer[random_index]
		else:
			state = self.cassie_state
		ref_pos, ref_vel = np.copy(self.get_kin_next_state())

		if self.phase < 14:
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
			useful_state = np.copy(np.concatenate([[state.pelvis.position[2] - state.terrain.height], new_orientation[:], state.motor.position[:], new_translationalVelocity[:], state.pelvis.rotationalVelocity[:], state.motor.velocity[:], new_translationalAcceleration[:], state.joint.position[:], state.joint.velocity[:]]))
			return np.concatenate([useful_state, ref_pos[pos_index], ref_vel[vel_index]])
		else:
			pos_index = np.array([2,3,4,5,6,21,22,23,28,29,30,34,7,8,9,14,15,16,20])
			vel_index = np.array([0,1,2,3,4,5,19,20,21,25,26,27,31,6,7,8,12,13,14,18])
			ref_vel[1] = -ref_vel[1]
			euler = quaternion2euler(ref_pos[3:7])
			euler[0] = -euler[0]
			euler[2] = -euler[2]
			ref_pos[3:7] = euler2quat(z=euler[2],y=euler[1],x=euler[0])
			quaternion = euler2quat(z=-self.orientation, y=0, x=0)
			iquaternion = inverse_quaternion(quaternion)

			pelvis_euler = quaternion2euler(np.copy(state.pelvis.orientation[:]))
			pelvis_euler[0] = -pelvis_euler[0]
			pelvis_euler[2] = -pelvis_euler[2]
			pelvis_orientation = euler2quat(z=pelvis_euler[2],y=pelvis_euler[1],x=pelvis_euler[0])

			translational_velocity = np.copy(state.pelvis.translationalVelocity[:])
			translational_velocity[1] = -translational_velocity[1]

			translational_acceleration = np.copy(state.pelvis.translationalAcceleration[:])
			translational_acceleration[1] = -translational_acceleration[1]

			rotational_velocity = np.copy(state.pelvis.rotationalVelocity)
			rotational_velocity[0] = -rotational_velocity[0]
			rotational_velocity[2] = -rotational_velocity[2]

			motor_position = np.zeros(10)
			motor_position[0:5] = np.copy(state.motor.position[5:10])
			motor_position[5:10] = np.copy(state.motor.position[0:5])
			motor_position[0] = -motor_position[0]
			motor_position[1] = -motor_position[1]
			motor_position[5] = -motor_position[5]
			motor_position[6] = -motor_position[6]

			motor_velocity = np.zeros(10)
			motor_velocity[0:5] = np.copy(state.motor.velocity[5:10])
			motor_velocity[5:10] = np.copy(state.motor.velocity[0:5])
			motor_velocity[0] = -motor_velocity[0]
			motor_velocity[1] = -motor_velocity[1]
			motor_velocity[5] = -motor_velocity[5]
			motor_velocity[6] = -motor_velocity[6]

			joint_position = np.zeros(6)
			joint_position[0:3] = np.copy(state.joint.position[3:6])
			joint_position[3:6] = np.copy(state.joint.position[0:3])

			joint_velocity = np.zeros(6)
			joint_velocity[0:3] = np.copy(state.joint.velocity[3:6])
			joint_velocity[3:6] = np.copy(state.joint.velocity[0:3])

			left_toeForce = np.copy(state.rightFoot.toeForce[:])
			left_toeForce[1] = -left_toeForce[1]
			left_heelForce = np.copy(state.rightFoot.heelForce[:])
			left_heelForce[1] = -left_heelForce[1]

			right_toeForce = np.copy(state.leftFoot.toeForce[:])
			right_toeForce[1] = -right_toeForce[1]
			right_heelForce = np.copy(state.leftFoot.heelForce[:])
			right_heelForce[1] = -right_heelForce[1]
			
			new_orientation = quaternion_product(iquaternion, pelvis_orientation)
			new_translationalVelocity = rotate_by_quaternion(translational_velocity, iquaternion)
			new_translationalAcceleration = rotate_by_quaternion(translational_acceleration, iquaternion)
			new_rotationalVelocity = rotate_by_quaternion(rotational_velocity, quaternion)

			useful_state = np.copy(np.concatenate([[state.pelvis.position[2] - state.terrain.height], new_orientation[:], motor_position, new_translationalVelocity[:], rotational_velocity, motor_velocity, new_translationalAcceleration[:], joint_position, joint_velocity]))
			return np.concatenate([useful_state, ref_pos[pos_index], ref_vel[vel_index]])
	def step_simulation(self, action):
		#self.sim.perturb_mass()
		#self.sim.get_mass()
		qpos = np.copy(self.sim.qpos())
		qvel = np.copy(self.sim.qvel())

		pos_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
		vel_index = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

		ref_pos, ref_vel = self.get_kin_next_state()
		if self.phase < 14:
			target = action + ref_pos[pos_index]
		else:
			mirror_action = np.zeros(10)
			mirror_action[0:5] = np.copy(action[5:10])
			mirror_action[5:10] = np.copy(action[0:5])
			mirror_action[0] = -mirror_action[0]
			mirror_action[1] = -mirror_action[1]
			mirror_action[5] = -mirror_action[5]
			mirror_action[6] = -mirror_action[6]
			target = mirror_action + ref_pos[pos_index]

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

	def reset(self):
		self.orientation = 0
		self.speed = 0#(random.randint(-10, 10)) / 10.0
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

	def reset_by_speed(self, speed, y_speed=0, phase=None):
		self.orientation = 0
		self.speed = speed#(random.randint(-10, 10)) / 10.0
		self.y_speed = 0
		orientation = self.orientation + (random.randint(0, 1) * 2 - 1) * np.pi / 10
		quaternion = euler2quat(z=orientation, y=0, x=0)
		if phase is None:
			self.phase = random.randint(0, 27)
		else:
			self.phase = phase
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

	def reset_by_phase(self, phase):
		self.orientation = 0
		self.speed = (random.randint(-10, 10)) / 10
		orientation = 0#self.orientation + random.randint(-20, 20) * np.pi / 100
		quaternion = euler2quat(z=orientation, y=0, x=0)
		self.phase = phase
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
