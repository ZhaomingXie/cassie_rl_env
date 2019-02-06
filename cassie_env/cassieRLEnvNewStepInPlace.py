from cassieRLEnvWithMoreState import *
from cassiemujoco_ctypes import *
import math
from quaternion_function import *


class cassieRLEnvNewStepInPlace(cassieRLEnvWithFootForces):
    def __init__(self):
        super().__init__()
        self.trajectory = CassieTrajectory("trajectory/more-poses-trial.bin")
        self.policy_freq = 58
        self.max_phase = 29


    def step_simulation(self, action):
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel())

        pos_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
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

        self.state_buffer.append(self.sim.step_pd(self.u))
        if len(self.state_buffer) > self.buffer_size:
            self.state_buffer.pop(0)
            #print(self.state_buffer[0].pelvis.translationalAcceleration[:])
            #print(self.state_buffer[1].pelvis.translationalAcceleration[:])

        self.cassie_state = self.state_buffer[len(self.state_buffer) - 1]

    def step(self, action):
        for _ in range(self.policy_freq):
            self.step_simulation(action)

        height = self.sim.qpos()[2]
        self.time += 1
        self.phase += 1

        if self.phase >= self.max_phase:
            self.phase = 0
            self.counter +=1
        #print("height", height)

        done = not(height > 0.4 and height < 3.0) or self.time >= self.time_limit
        yaw = quat2yaw(self.sim.qpos()[3:7])
        #print(yaw - self.orientation)
        #if (yaw-self.orientation)**2 > 0.15:
        #       done = True

        reward = self.compute_reward()
        #print(reward)
        if reward < 0.3:
            done = True

        return self.get_state(), reward, done, {}

    def get_state(self):
        #if len(self.state_buffer) >= self.buffer_size and self.delay:
        #       random_index = random.randint(0, self.buffer_size - 1)
        #       state = self.state_buffer[random_index]
            #print(random_index)
        #else:
        state = self.cassie_state

        ref_pos, ref_vel = self.get_kin_next_state()

        pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
        vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])


        #state.pelvis.orientation[3] -= np.sin(self.orientation/2)
        quaternion = euler2quat(z=self.orientation, y=0, x=0)
        iquaternion = inverse_quaternion(quaternion)
        new_orientation = quaternion_product(iquaternion, state.pelvis.orientation[:])
        #print(new_orientation)
        new_translationalVelocity = rotate_by_quaternion(state.pelvis.translationalVelocity[:], iquaternion)
        #print(new_translationalVelocity)
        new_translationalAcceleration = rotate_by_quaternion(state.pelvis.translationalAcceleration[:], iquaternion)
        new_rotationalVelocity = rotate_by_quaternion(state.pelvis.rotationalVelocity[:], quaternion)
        #print(state.pelvis.rotationalVelocity[:])
        #print(new_rotationalVelocity, state.pelvis.rotationalVelocity[:])
        #print(state.rightFoot.toeForce[:])
        '''new_lefttoeForce = rotate_by_quaternion(state.leftFoot.toeForce[:], iquaternion)
        new_leftFootheelForce = rotate_by_quaternion(state.leftFoot.heelForce[:], iquaternion)
        new_rightFootheelForce = rotate_by_quaternion(state.rightFoot.heelForce[:], iquaternion)
        new_rightFoottoeForce = rotate_by_quaternion(state.rightFoot.toeForce[:], iquaternion)'''

        useful_state = np.copy(np.concatenate([0 * np.ones(1), [state.pelvis.position[2] - state.terrain.height], new_orientation[:], state.motor.position[:], new_translationalVelocity[:], state.pelvis.rotationalVelocity[:], state.motor.velocity[:], new_translationalAcceleration[:], state.leftFoot.toeForce[:], state.leftFoot.heelForce[:], state.rightFoot.toeForce[:], state.rightFoot.heelForce[:]]))
        #print(useful_state[1])

        return np.concatenate([useful_state, ref_pos[pos_index], ref_vel[vel_index]])

    def get_mirror_state(self, state):
        orientation = np.copy(state[0:1])
        height = np.copy(state[1:2])
        pelvis_orientation = np.copy(state[2:6])
        euler = quaternion2euler(pelvis_orientation)
        euler[0] = -euler[0]
        euler[2] = -euler[2]
        pelvis_orientation = euler2quat(z=euler[2],y=euler[1],x=euler[0])
        #pelvis_orientation[1] = -pelvis_orientation[1]
        #pelvis_orientation[3] = -pelvis_orientation[3]


        left_motor_p = np.copy(state[6:11])
        right_motor_p = np.copy(state[11:16])
        pelvis_vel = np.copy(state[16:19])
        pelvis_vel[1] = -pelvis_vel[1]
        pelvis_rot_vel =np.copy(state[19:22])
        pelvis_rot_vel[0] = -pelvis_rot_vel[0]
        pelvis_rot_vel[2] = -pelvis_rot_vel[2]
        left_motor_v = np.copy(state[22:27])
        right_motor_v = np.copy(state[27:32])
        pelvis_acc = np.copy(state[32:35])
        pelvis_acc[1] = -pelvis_acc[1]
        left_toe_force = np.copy(state[35:38])
        left_toe_force[1] = -left_toe_force[1]
        left_heel_force = np.copy(state[38:41])
        left_heel_force[1] = -left_heel_force[1]
        right_toe_force = np.copy(state[41:44])
        right_toe_force[1] = -right_toe_force[1]
        right_heel_force = np.copy(state[44:47])
        right_heel_force[1] = -right_heel_force[1]
        ref_pelvis_p = np.copy(state[47:53])
        ref_pelvis_p[0] = -ref_pelvis_p[0]
        ref_pelvis_p[3] = -ref_pelvis_p[3]
        ref_pelvis_p[5] = -ref_pelvis_p[5]
        ref_left_motor_p = np.copy(state[53:60])
        ref_right_motor_p = np.copy(state[60:67])
        ref_vel_p = np.copy(state[67:73])
        ref_vel_p[1] = -ref_vel_p[1]
        ref_vel_p[3] = -ref_vel_p[3]
        ref_vel_p[5] = -ref_vel_p[5]
        ref_left_motor_v = np.copy(state[73:80])
        ref_right_motor_v = np.copy(state[80:87])
        return np.concatenate([np.copy(orientation), np.copy(height), np.copy(pelvis_orientation), np.copy(right_motor_p), np.copy(left_motor_p), np.copy(pelvis_vel), np.copy(pelvis_rot_vel), np.copy(right_motor_v), np.copy(left_motor_v), np.copy(pelvis_acc), np.copy(right_toe_force), np.copy(right_heel_force), np.copy(left_toe_force), np.copy(left_heel_force), np.copy(ref_pelvis_p), np.copy(ref_right_motor_p), np.copy(ref_left_motor_p), np.copy(ref_vel_p), np.copy(ref_right_motor_v), np.copy(ref_left_motor_v)])

    def get_mirror_action(self, action):
        left_action = np.copy(action[0:5])
        right_action = np.copy(action[5:10])
        return np.concatenate([right_action, left_action])

    def get_kin_state(self):
        pose = np.copy(self.trajectory.qpos[self.phase*self.policy_freq])
        pose[0] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter
        pose[1] = 0
        pose[0] = 0
        vel = np.copy(self.trajectory.qvel[self.phase*self.policy_freq])
        vel[0] = 0
        return pose, vel

    def get_kin_next_state(self):
        phase = self.phase + 1
        if phase >= self.max_phase:
            phase = 0
        pose = np.copy(self.trajectory.qpos[phase*self.policy_freq])
        vel = np.copy(self.trajectory.qvel[phase*self.policy_freq])
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
        pelvis_pos[2] -= self.cassie_state.terrain.height

        com_penalty = (pelvis_pos[0])**2 + (pelvis_pos[1])**2 + (pelvis_pos[2]-ref_pos[2])**2

        yaw = quat2yaw(self.sim.qpos()[3:7])

        orientation_penalty = (self.sim.qpos()[4]*20)**2+(self.sim.qpos()[5]*20)**2+(yaw - self.orientation)**2

        spring_penalty = (self.sim.qpos()[15])**2+(self.sim.qpos()[29])**2
        spring_penalty *= 1000

        foot_forces = self.get_foot_forces()
        #print(foot_forces)
        force_penalty = 0
        if self.foot_forces[0] < 10 and foot_forces[2] > 10:
            force_penalty += (foot_forces[2]-150)**2
        if self.foot_forces[1] < 10 and foot_forces[8] > 10:
            force_penalty += (foot_forces[8]-150)**2
        self.foot_forces[0] = foot_forces[2]
        self.foot_forces[1] = foot_forces[8]

        total_reward = 0.5*np.exp(-joint_penalty)+0.3*np.exp(-com_penalty)+0.1*np.exp(-orientation_penalty)+0.1*np.exp(-force_penalty)

        return total_reward

    def reset(self):
        self.orientation = 0 * np.pi#0.5*np.pi#0.5*np.pi#random.randint(-4, 4) * np.pi / 4.0
        #self.orientation = -0.712 * np.pi
        orientation = self.orientation + random.randint(-20, 20) * np.pi / 100
        quaternion = euler2quat(z=orientation, y=0, x=0)
        #yaw = quat2yaw(quaternion)
        #print(yaw - self.orientation)
        #print(np.arcsin(quaternion[3])-self.orientation/2)
        self.phase = 0#random.randint(0, 27)
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
        self.orientation = 0.0*np.pi#random.randint(-10, 10) * np.pi / 10.0
        #self.orientation = -0.712 * np.pi
        orientation = self.orientation + random.randint(-20, 20) * np.pi / 100
        quaternion = euler2quat(z=orientation, y=0, x=0)
        #print(quaternion[3]-np.sin(self.orientation/2))
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

    def set_state(self, qpos, qvel):
        self.sim.set_qpos(qpos)
        self.sim.set_qvel(qvel)
