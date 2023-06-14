"""Environment for Position reach only
Observations: End-Effector Positions, and Joint Angles """

import os
import math
import time
from typing import Any, SupportsFloat
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
# import gym
# from gym import spaces
# from gym.utils import seeding

MIN_GOAL_COORDS = np.array([-0.8, -.2, 0.05])
MAX_GOAL_COORDS = np.array([-0.2, .2, .5])
MIN_END_EFF_COORDS = np.array([-.8, -.2, 0.05])
MAX_END_EFF_COORDS = np.array([-.2, .2, .5])
FIXED_GOAL_COORDS_SPHERE = np.array([-.7, -0.1, 0.35])
# RESET_VALUES=[0.00, -2.464, 1.486, 1.405, 1.393, 1.515, -1.747, 0.842, 0.0, 0.000000, -0.00000]
RESET_VALUES=[-2.464, 1.486, 1.405, 1.393, 1.515, -1.747, 0.842]
RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class iiwaEnvPos(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.useSimulation = 1
        self.useInverseKinematics = 1
        self.useNullSpace = 1
        self.useOrientation = 1
        self.eef_index = 6
        self.observations = []
        self.cam_dist = 1.3
        self.cam_yaw = 180
        self.cam_pitch = -40
        self.maxForce = 200
        self.maxVelocity=0.35
        actionRepeat=1
        renders=True
        maxSteps=100000
        self._maxSteps = maxSteps
        self._actionRepeat = actionRepeat
        self._renders = renders
        self.terminated = 0
        self._timestep = 1. / 240.
        self.l_jointlimit = [-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054]
        self.u_jointlimit = [2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054]
        #lower limits for null space
        self.lowerlimits = [-2.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        #upper limits for null space
        self.upperlimits = [2.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        #joint ranges for null space
        self.jointranges = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        #restposes for null space
        self.resetposes = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        #joint damping coefficents
        self.jointdampings = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]
        self.step_count = 0

        #****** RL reach parameters ******#
        self.endeffector_pos = np.zeros(3)
        self.old_endeffector_pos = np.zeros(3)
        self.end_goal_pos = np.zeros(3)
        self.joint_positions = np.zeros(7)
        self.new_joint_positions = np.zeros(7)
        self.goal_pos = np.zeros(3)
        self.target_object_orient = np.zeros(3)
        self.reward = None
        self.observation = None
        self.action = np.zeros(7)
        self.pybullet_action = np.zeros(7)
        self.pybullet_action_coeff = 1
        self.pybullet_action_min = np.array([-0.05, -0.05, -0.05, -0.05, -0.05, -0.05 -0.05]) * self.pybullet_action_coeff
        self.pybullet_action_max = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]) * self.pybullet_action_coeff
        self.action_min = [-1, -1, -1, -1, -1, -1, -1]
        self.action_max = [1, 1, 1, 1, 1, 1, 1]
        self.dist = 0
        self.old_dist = 0
        self.delta_pos = 0
        self.delta_dist = 0
        self.reward_coeff = 1.0

        # render settings
        self.renderer = p.ER_TINY_RENDERER  # p.ER_BULLET_HARDWARE_OPENGL
        self._width = 224
        self._height = 224
        self._cam_dist = 0.8
        self._cam_yaw = 0
        self._cam_pitch = -30
        self._cam_roll = 0
        self.camera_target_pos = [0.2, 0, 0.1]
        self._screen_width = 3840 #1920
        self._screen_height = 2160 #1080
        #    # For joint space action
        # self.action_space = spaces.Box(
        #         low=np.float32(self.action_min),
        #         high=np.float32(self.action_max),
        #         dtype=np.float32)
        #   For Cartesian space action
        self.action_space = spaces.Box(np.array([-1]*3), np.array([1]*3))

        self.obs_space_low = np.float32(
                np.concatenate((MIN_END_EFF_COORDS, self.l_jointlimit), axis=0))
        self.obs_space_high = np.float32(
                np.concatenate((MAX_END_EFF_COORDS, self.u_jointlimit), axis=0))
        
        self.observation_space = spaces.Box(
                    low=self.obs_space_low,
                    high=self.obs_space_high,
                    dtype=np.float64)
        #****** RL reach parameters ******#
        self.physics_client=p.connect(p.DIRECT)
        p.setTimeStep(self._timestep)
        self.world = self.createWorld()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        # self.state = self.init_state()

    # def init_state(self):
        # p.resetSimulation()
        # p.stepSimulation()
        # self.observation = self.getObservation()
        # return np.array(self.observation)

        # eef_pose = p.getLinkState(self.iiwaId, 10)[0]
        # obs = np.array([eef_pose]).flatten
        # return observations
    
    def reset(self, seed=None, options=None):
    # def reset(self):
        super().reset(seed=seed)
        # self.reward=0
        self.terminated = 0
        self.step_count=0
        self.goal_pos = FIXED_GOAL_COORDS_SPHERE
        for jointIndex in range(6):
            p.resetJointState(self.iiwaId, jointIndex, RESET_VALUES[jointIndex])

        # p.stepSimulation()
        self.observation = self.getObservation()
        info = self._get_info()
        return np.array(self.observation), info
    
    def _get_info(self):
        return {"distance": np.linalg.norm(self.endeffector_pos-self.goal_pos, ord=1)}
    
    def seed(self, seed = None):
        self.np_random, seed  = seeding.np_random(seed)
        return [seed]
    
    def step(self,action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw]).
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv

        # get distance and end effector position before taking the action
        self.endeffector_pos = p.getLinkState(self.iiwaId,self.eef_index)[0]
        self.old_dist = np.linalg.norm(self.endeffector_pos - self.goal_pos) #defin goal pos
        self.old_endeffector_pos = self.endeffector_pos
        #  take action
        # self.action = np.array(action, dtype=np.float32)
        new_pose = [self.endeffector_pos[0]+dx,
                    self.endeffector_pos[1]+dy,
                    self.endeffector_pos[2]+dz]
        # print("new_pose")
        # print(new_pose)
        jointPoses = p.calculateInverseKinematics(self.iiwaId,
                                                self.eef_index,
                                                new_pose,
                                                lowerLimits=self.lowerlimits,
                                                upperLimits=self.upperlimits,
                                                jointRanges=self.jointranges,
                                                restPoses=self.resetposes)
        
        for joinitIndex in range(6):
            # p.resetJointState(self.iiwaId, jointIndex, RESET_VALUES[jointIndex])
            p.setJointMotorControl2(bodyIndex=self.iiwaId,
                                jointIndex=joinitIndex,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses[joinitIndex],
                                targetVelocity=0,
                                force=500,
                                positionGain=0.03,
                                velocityGain=1)
        p.stepSimulation()
        # self.step_count+=1
        actionCost = np.linalg.norm(action) * 10
        reward = self._reward() - actionCost
        terminated  = self._termination()
        
        # if self.new_distance < 0.0005:
        #      terminated = True
        # print("reward")
        # print(reward)
        
        truncated = False
        info = {"is_success": terminated}
        return np.array(self.observation), reward, terminated, truncated, info
   
    def render(self, mode='human'):
        """ Render Pybullet simulation """

        p.disconnect(self.physics_client)
        self.physics_client = p.connect(p.GUI, options='--background_color_red=0.8 --background_color_green=0.9 --background_color_blue=1.0 --width=%d --height=%d' % (self._screen_width, self._screen_height))
        self.createWorld()

        # Initialise debug camera angle
        p.resetDebugVisualizerCamera(
            cameraDistance=self._cam_dist,
            cameraYaw=self._cam_yaw,
            cameraPitch=self._cam_pitch,
            cameraTargetPosition=self.camera_target_pos,
            physicsClientId=self.physics_client)    

    def close(self):
        p.disconnect()
    
    ####################################################################################
    ######################## user Defined functions ####################################
    ####################################################################################

    def createWorld(self):
            self.goal_pos = FIXED_GOAL_COORDS_SPHERE
            p.setGravity(0,0,-9.81)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.iiwaId = p.loadURDF("kuka_iiwa/model.urdf",[0,0,0])
            p.resetBasePositionAndOrientation(self.iiwaId, [0.00000, 0.000000, 0.00000],
                                        [0.000000, 0.000000, 0.000000, 1.000000])
            path = "/home/terabotics/gym_ws/my_py_bullet/kuka_env/kuka_env_example/envs"
            # self.iiwaId = p.loadURDF(os.path.join(path,'agents','assets','iiwa','urdf','iiwa14_rs_scanner_v1.urdf'),
            #                         basePosition=[0,0,0],
            #                         baseOrientation=[0,0,0,1],
            #                         useFixedBase=True,
            #                         flags=p.URDF_USE_SELF_COLLISION)
            self.target_object = p.loadURDF(os.path.join(path,'agents','assets','iiwa','urdf', "sphere.urdf"),
                                        useFixedBase=True)
            p.resetBasePositionAndOrientation(
                self.target_object, self.goal_pos, p.getQuaternionFromEuler(self.target_object_orient))
            p.loadURDF("plane.urdf",[0,0,0],[0,0,0,1])
            self.numJoints = p.getNumJoints(self.iiwaId)
            for jointIndex in range(self.numJoints):
                p.resetJointState(self.iiwaId, jointIndex, RESET_VALUES[jointIndex])
                # p.setJointMotorControl2(self.iiwaId,
                #                     jointIndex,
                #                     p.POSITION_CONTROL,
                #                     targetPosition=RESET_VALUES[jointIndex],
                #                     force=200)
            p.stepSimulation()

    def _normalize_scalar(self, var, old_min, old_max, new_min, new_max):
        """ Normalize scalar var from one range to another """
        return ((new_max - new_min) * (var - old_min) / (old_max - old_min)) + new_min

    def _scale_action_pybullet(self):
        """ Scale action to Pybullet action range """
        for i in range(6):
            self.pybullet_action[i] = self._normalize_scalar(
                self.action[i],
                self.action_min[i],
                self.action_max[i],
                self.pybullet_action_min[i],
                self.pybullet_action_max[i])

    def getObservation(self):
        self.observation=[]
        linkstate = p.getLinkState(self.iiwaId, self.eef_index)
        pos = linkstate[0]
        orn = linkstate[1]
        euler = p.getEulerFromQuaternion(orn)
        j_pos = np.zeros(7)
        count=0
        # for joint_number in range(1,self.numJoints-3):
        for joint_number in range(6):
            j_pos[count] = p.getJointState(self.iiwaId,joint_number)[0]
            count+=1
        # j_pos_ = tuple(j_pos)
        self.observation = np.concatenate((pos, j_pos),axis=0)
        # self.observation.extend(list(pos))
        # self.observation.extend(list(euler))

        return self.observation
    
    def _termination(self):
        #print (self._kuka.endEffectorPos[2])
        state = p.getLinkState(self.iiwaId, self.eef_index)
        actualEndEffectorPos = state[0]

        #print("self._envStepCounter")
        #print(self._envStepCounter)
        # if (self.step_count > self._maxSteps):
        #     self.observation = self.getObservation()
        #     print("*********************terminated observation*********************")
        #     return True
        
        if (np.linalg.norm(actualEndEffectorPos-self.goal_pos) <0.001):
            self._observation = self.getObservation()
            return True
        # print(actualEndEffectorPos)
        if (actualEndEffectorPos[2] < 0.25):
            return True
        
        return False

    def _reward(self):
        state = p.getLinkState(self.iiwaId, self.eef_index)
        tool_pos = state[0]
        self.dist = np.linalg.norm(tool_pos-self.goal_pos)
        # print(self.dist)
        reward = - self.dist ** 2 
        reward *= self.reward_coeff
        
        # if (np.linalg.norm(tool_pos-self.goal_pos) > 0.1):
        #     reward += -10
        if (tool_pos[2]<0.25):
            reward += -10
        if (np.linalg.norm(tool_pos-self.goal_pos) < 0.05):
            reward = reward+10
        # print(reward)
        return reward
        
# def step(self,action):
    #     # get distance and end effector position before taking the action
    #     self.endeffector_pos = p.getLinkState(self.iiwaId,self.eef_index)[0]
    #     self.old_dist = np.linalg.norm(self.endeffector_pos - self.goal_pos) #defin goal pos
    #     self.old_endeffector_pos = self.endeffector_pos
    #     #  take action
    #     self.action = np.array(action, dtype=np.float32)
    #     self._scale_action_pybullet()
    #     j_pos=np.zeros(7)
    #     count=0
    #     for joint_number in range(1,self.numJoints-3):
    #         j_pos[count] = p.getJointState(self.iiwaId,joint_number)[0]
    #         count+=1
    #     self.new_joint_positions = j_pos + self.pybullet_action
    #     """ Instantaneous reset of the joint angles (not position control) """
    #     j_1 = np.array([0.0])
    #     j_last = np.array([0.0,0.0,0.0])
    #     j_state_ = np.concatenate((j_1, self.new_joint_positions, j_last),axis=0)
    #     for jointIndex in range(self.numJoints):
    #         # p.resetJointState(self.iiwaId, jointIndex, RESET_VALUES[jointIndex])
    #         p.setJointMotorControl2(self.iiwaId,
    #                             jointIndex,
    #                             p.POSITION_CONTROL,
    #                             targetPosition=j_state_[jointIndex],
    #                             force=200)
    #     p.stepSimulation()
    #     self.step_count+=1
    #     terminated  = self._termination()
    
    #     reward = self._reward()
        
    #     # if self.new_distance < 0.0005:
    #     #      terminated = True
    #     # print("reward")
    #     # print(reward)
        
    #     truncated = False
    #     info = {"is_success": terminated}
    #     return np.array(self.observation), reward, terminated, truncated, info


        

