import pybullet as p
import pybullet_data
import time
import numpy as np
import math

client = p.connect(p.GUI)
p.setGravity(0,0,-9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf", [0, 0, 0.0])
robotId = p.loadURDF("kuka_iiwa/model.urdf", [0,0,0])
p.resetBasePositionAndOrientation(robotId, [0, 0, 0], [0, 0, 0, 1])
position, orientation = p.getBasePositionAndOrientation(robotId)
number_joints = p.getNumJoints(robotId)
# jointPositions = [
#         0.00, -2.464, 1.486, 1.405, 1.393, 1.515, -1.747, 0.842, 0.0, 0.000000, -0.00000 ]
# jointPositions = [
#         0.00, 2.464, 0.486, 0, -1.393, -1.515, 1.747, -0.842, 0.0, 0.000000, -0.00000 ]
#lower limits for null space
ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
#upper limits for null space
ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
#joint ranges for null space
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
#restposes for null space
rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
#joint damping coefficents
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
for jointIndex in range(6):
    p.resetJointState(robotId, jointIndex, rp[jointIndex])
p.setGravity(0,0,0)
eef_index=6
useNullSpace = 1
useOrientation = 1
#If we set useSimulation=0, it sets the arm pose to be the IK result directly without using dynamic control.
#This can be used to test the IK result accuracy.
useSimulation = 1
useRealTimeSimulation = 0
ikSolver = 0
p.setRealTimeSimulation(useRealTimeSimulation)
t=0.0
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
hasPrevPose = 0
trailDuration = 15
# print(eef_pose)
print(number_joints)
for joint_number in range(number_joints):
    info = p.getJointInfo(robotId,joint_number)
    print("joint",joint_number)
    print(info,end="\n")
eef_pose = p.getLinkState(robotId,eef_index)
print(eef_pose)

# j_state = np.array(7)
# j_state=np.zeros(7)
# print(j_state)
# count=0
# for joint_number in range(1,number_joints-3):
#     j_state[count] = p.getJointState(robotId,joint_number)[0]
#     print("joint",joint_number, j_state[count])
#     count+=1
i=0
while 1:
    i+=1
    t = t+0.01
    p.stepSimulation()
    for i in range(1):
        pos = [-0.4, 0.2 * math.cos(t), 0. + 0.2 * math.sin(t)]
        orn = p.getQuaternionFromEuler([0,-math.pi,0])
        if (useNullSpace == 1):
            if (useOrientation == 1):
                jointPoses = p.calculateInverseKinematics(robotId, eef_index, pos, orn, ll, ul,
                                                        jr, rp)
            else:
                jointPoses = p.calculateInverseKinematics(robotId,
                                                        eef_index,
                                                        pos,
                                                        lowerLimits=ll,
                                                        upperLimits=ul,
                                                        jointRanges=jr,
                                                        restPoses=rp)
        else:
            if (useOrientation == 1):
                jointPoses = p.calculateInverseKinematics(robotId,
                                                        eef_index,
                                                        pos,
                                                        orn,
                                                        jointDamping=jd,
                                                        solver=ikSolver,
                                                        maxNumIterations=100,
                                                        residualThreshold=.01)
            else:
                jointPoses = p.calculateInverseKinematics(robotId,
                                                        eef_index,
                                                        pos,
                                                        solver=ikSolver)

        if (useSimulation):
            for i in range(7):
                p.setJointMotorControl2(bodyIndex=robotId,
                                        jointIndex=i,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=jointPoses[i],
                                        targetVelocity=0,
                                        force=500,
                                        positionGain=0.03,
                                        velocityGain=1)
        else:
        #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
            for i in range(7):
                p.resetJointState(robotId, i, jointPoses[i])

    ls = p.getLinkState(robotId, eef_index)
    if (hasPrevPose):
        p.addUserDebugLine(prevPose, pos,[0,0,0.3],1,trailDuration)
        p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, trailDuration)
    prevPose = pos
    prevPose1 = ls[4]
    hasPrevPose = 1
p.disconnect()
        





# %% 
# j_1 = np.array([0.0])
# j_last = np.array([0.0,0.0,0.0])
# j_state_ = np.concatenate((j_1, -j_state, j_last),axis=0)
# # print(j_state_)
# for i in range(numJoints):
#             p.resetJointState(
#                 robotId,
#                 i,
#                 j_state_[i]
#            )
