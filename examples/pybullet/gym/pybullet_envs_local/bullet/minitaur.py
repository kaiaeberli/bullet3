"""This file implements the functionalities of a minitaur using pybullet.

"""
import copy
import math
import numpy as np
from . import motor
import os

INIT_POSITION = [0, 0, .2]
INIT_ORIENTATION = [0, 0, 0, 1] # quaternion
KNEE_CONSTRAINT_POINT_RIGHT = [0, 0.005, 0.2]
KNEE_CONSTRAINT_POINT_LEFT = [0, 0.01, 0.2]
OVERHEAT_SHUTDOWN_TORQUE = 2.45 # N m
OVERHEAT_SHUTDOWN_TIME = 1.0
LEG_POSITION = ["front_left", "back_left", "front_right", "back_right"]
MOTOR_NAMES = [
    "motor_front_leftL_joint", "motor_front_leftR_joint", "motor_back_leftL_joint",
    "motor_back_leftR_joint", "motor_front_rightL_joint", "motor_front_rightR_joint",
    "motor_back_rightL_joint", "motor_back_rightR_joint"
] # first the right side then the left side of the robot
LEG_LINK_ID = [2, 3, 5, 6, 8, 9, 11, 12, 15, 16, 18, 19, 21, 22, 24, 25]
MOTOR_LINK_ID = [1, 4, 7, 10, 14, 17, 20, 23]
FOOT_LINK_ID = [3, 6, 9, 12, 16, 19, 22, 25]
BASE_LINK_ID = -1


class Minitaur(object):
  """The minitaur class that simulates a quadruped robot from Ghost Robotics.

  """

  def __init__(self,
               pybullet_client,
               urdf_root=os.path.join(os.path.dirname(__file__), "../data"),
               time_step=0.01, #seconds
               self_collision_enabled=False,
               motor_velocity_limit=np.inf,
               pd_control_enabled=False,
               accurate_motor_model_enabled=False,
               motor_kp=1.0,
               motor_kd=0.02,
               torque_control_enabled=False,
               motor_overheat_protection=False,
               on_rack=False,
               kd_for_pd_controllers=0.3):
    """Constructs a minitaur and reset it to the initial states.

    Args:
      pybullet_client: The instance of BulletClient to manage different
        simulations.
      urdf_root: The path to the urdf folder.
      time_step: The time step of the simulation.
      self_collision_enabled: Whether to enable self collision.
      motor_velocity_limit: The upper limit of the motor velocity.
      pd_control_enabled: Whether to use PD control for the motors.
      accurate_motor_model_enabled: Whether to use the accurate DC motor model.
      motor_kp: proportional gain for the accurate motor model
      motor_kd: derivative gain for the acurate motor model
      torque_control_enabled: Whether to use the torque control, if set to
        False, pose control will be used.
      motor_overheat_protection: Whether to shutdown the motor that has exerted
        large torque (OVERHEAT_SHUTDOWN_TORQUE) for an extended amount of time
        (OVERHEAT_SHUTDOWN_TIME). See ApplyAction() in minitaur.py for more
        details.
      on_rack: Whether to place the minitaur on rack. This is only used to debug
        the walking gait. In this mode, the minitaur's base is hanged midair so
        that its walking gait is clearer to visualize.
      kd_for_pd_controllers: kd value for the pd controllers of the motors.
    """
    self.num_motors = 8
    self.num_legs = int(self.num_motors / 2) # coaxial drive like Doggo?
    self._pybullet_client = pybullet_client
    self._urdf_root = urdf_root
    self._self_collision_enabled = self_collision_enabled
    self._motor_velocity_limit = motor_velocity_limit
    self._pd_control_enabled = pd_control_enabled
    self._motor_direction = [-1, -1, -1, -1, 1, 1, 1, 1] # coaxial motors, mounted on each hip joint, driving in opposite directions
    self._observed_motor_torques = np.zeros(self.num_motors)
    self._applied_motor_torques = np.zeros(self.num_motors)
    self._max_force = 3.5
    self._accurate_motor_model_enabled = accurate_motor_model_enabled
    self._torque_control_enabled = torque_control_enabled
    self._motor_overheat_protection = motor_overheat_protection
    self._on_rack = on_rack

    if self._accurate_motor_model_enabled:
      self._kp = motor_kp
      self._kd = motor_kd
      self._motor_model = motor.MotorModel(torque_control_enabled=self._torque_control_enabled,
                                           kp=self._kp,
                                           kd=self._kd)
    elif self._pd_control_enabled:
      self._kp = 8
      self._kd = kd_for_pd_controllers
    else:
      self._kp = 1
      self._kd = 1
    self.time_step = time_step
    self.Reset() # load robot urdf

  def _RecordMassInfoFromURDF(self):
    self._base_mass_urdf = self._pybullet_client.getDynamicsInfo(self.quadruped, BASE_LINK_ID)[0]
    self._leg_masses_urdf = []
    self._leg_masses_urdf.append(
        self._pybullet_client.getDynamicsInfo(self.quadruped, LEG_LINK_ID[0])[0])
    self._leg_masses_urdf.append(
        self._pybullet_client.getDynamicsInfo(self.quadruped, MOTOR_LINK_ID[0])[0])

  def _BuildJointNameToIdDict(self):
    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    self._joint_name_to_id = {}
    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
      self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

  def _BuildMotorIdList(self):
    self._motor_id_list = [self._joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES]

  def _BuildOrderedMotorNameList(self):
    self._motor_name_id_list = {motor_name: self._joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES}

  def GetOrderedMotorNameList(self):
    return  [[motor_name, self._joint_name_to_id[motor_name]] for motor_name in MOTOR_NAMES]

  def Reset(self, reload_urdf=True):
    """Reset the minitaur to its initial states.

    Args:
      reload_urdf: Whether to reload the urdf file. If not, Reset() just place
        the minitaur back to its starting position.
    """
    if reload_urdf:
      if self._self_collision_enabled:
        self.quadruped = self._pybullet_client.loadURDF(
            "%s/quadruped/minitaur.urdf" % self._urdf_root,
            INIT_POSITION,
            flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
      else:
        self.quadruped = self._pybullet_client.loadURDF(
            "%s/quadruped/minitaur.urdf" % self._urdf_root, INIT_POSITION)
      self._BuildJointNameToIdDict() # build joint name:id dict
      self._BuildMotorIdList() # get ids of motors
      self._BuildOrderedMotorNameList()
      self._RecordMassInfoFromURDF() # 
      self.ResetPose(add_constraint=True)
      if self._on_rack:
        self._pybullet_client.createConstraint(self.quadruped, -1, -1, -1,
                                               self._pybullet_client.JOINT_FIXED, [0, 0, 0],
                                               [0, 0, 0], [0, 0, 1])
    
    # reset position, orientation and velocity to 0
    else:
      self._pybullet_client.resetBasePositionAndOrientation(self.quadruped, INIT_POSITION,
                                                            INIT_ORIENTATION)
      self._pybullet_client.resetBaseVelocity(self.quadruped, [0, 0, 0], [0, 0, 0])
      self.ResetPose(add_constraint=False)

    self._overheat_counter = np.zeros(self.num_motors)
    self._motor_enabled_list = [True] * self.num_motors

  def _SetMotorTorqueById(self, motor_id, torque):
    self._pybullet_client.setJointMotorControl2(bodyIndex=self.quadruped,
                                                jointIndex=motor_id,
                                                controlMode=self._pybullet_client.TORQUE_CONTROL,
                                                force=torque) # the force that causes motor to turn in angular direction

  def _SetDesiredMotorAngleById(self, motor_id, desired_angle):
    self._pybullet_client.setJointMotorControl2(bodyIndex=self.quadruped,
                                                jointIndex=motor_id,
                                                controlMode=self._pybullet_client.POSITION_CONTROL,
                                                targetPosition=desired_angle, # motor will move to this angle
                                                positionGain=self._kp,
                                                velocityGain=self._kd,
                                                force=self._max_force)

  def _SetDesiredMotorAngleByName(self, motor_name, desired_angle):
    self._SetDesiredMotorAngleById(self._joint_name_to_id[motor_name], desired_angle)

  def ResetPose(self, add_constraint):
    """Reset the pose of the minitaur.

    Args:
      add_constraint: Whether to add a constraint at the joints of two feet.
    """
    for i in range(self.num_legs):
      self._ResetPoseForLeg(i, add_constraint)

  def _ResetPoseForLeg(self, leg_id, add_constraint):
    """Reset the initial pose for the leg. 
    Both hip motors on each leg are set to 90 degrees, both knees motors to 125 degrees.




    Args:
      leg_id: It should be 0, 1, 2, or 3, which represents the leg at
        front_left, back_left, front_right and back_right. Each leg has 4 motors, two for each subleg.
        Each subleg has a hip motor and a knee motor.

      add_constraint: Whether to add a constraint at the joints of two feet.
    """

    # default position of motor is point upwards along z axis, this is 0 degree rotation.

    knee_friction_force = 0
    half_pi = math.pi / 2.0 # default position of leg is pi/2 radians or 90 degrees
    knee_angle = -2.1834 # knee motor angle set to 125 degrees

    leg_position = LEG_POSITION[leg_id] # get location of leg on body

    # joint: hip bone (femur), link: shinbone (tibia)

    # can display urdf here: https://mymodelrobot.appspot.com/5629499534213120

    # set right leg

    # hip motor
    self._pybullet_client.resetJointState(self.quadruped,
                                          self._joint_name_to_id["motor_" + leg_position +
                                                                 "R_joint"],
                                          self._motor_direction[2 * leg_id + 1] * half_pi,
                                          targetVelocity=0)
    
    # knee motor
    self._pybullet_client.resetJointState(self.quadruped,
                                          self._joint_name_to_id["knee_" + leg_position +
                                                                 "R_link"],
                                          self._motor_direction[2 * leg_id + 1] * knee_angle,
                                          targetVelocity=0)


    # set left leg

    # hip motor
    self._pybullet_client.resetJointState(self.quadruped,
                                          self._joint_name_to_id["motor_" + leg_position +
                                                                 "L_joint"], # get id of joint by name
                                          self._motor_direction[2 * leg_id] * half_pi, # set to 90 degree
                                          targetVelocity=0)
    
    # knee motor
    self._pybullet_client.resetJointState(self.quadruped,
                                          self._joint_name_to_id["knee_" + leg_position +
                                                                 "L_link"],
                                          self._motor_direction[2 * leg_id] * knee_angle, # set to 125 degree
                                          targetVelocity=0)
    
   
    

    # connect knees together so that they move together - urdfs can't do this as tree structure
    if add_constraint: 
      self._pybullet_client.createConstraint(
          self.quadruped, self._joint_name_to_id["knee_" + leg_position + "R_link"],
          self.quadruped, self._joint_name_to_id["knee_" + leg_position + "L_link"],
          self._pybullet_client.JOINT_POINT2POINT, [0, 0, 0], KNEE_CONSTRAINT_POINT_RIGHT,
          KNEE_CONSTRAINT_POINT_LEFT)



    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      # Disable the default motor in pybullet.
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(self._joint_name_to_id["motor_" + leg_position + "L_joint"]),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0, # velocity = 0 means motor is not turning
          force=knee_friction_force)
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(self._joint_name_to_id["motor_" + leg_position + "R_joint"]),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=knee_friction_force)

    else:
      self._SetDesiredMotorAngleByName("motor_" + leg_position + "L_joint",
                                       self._motor_direction[2 * leg_id] * half_pi)
      self._SetDesiredMotorAngleByName("motor_" + leg_position + "R_joint",
                                       self._motor_direction[2 * leg_id + 1] * half_pi)

    # set knee motors of the leg to 0 velocity.
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=(self._joint_name_to_id["knee_" + leg_position + "L_link"]),
        controlMode=self._pybullet_client.VELOCITY_CONTROL,
        targetVelocity=0,
        force=knee_friction_force)
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=(self._joint_name_to_id["knee_" + leg_position + "R_link"]),
        controlMode=self._pybullet_client.VELOCITY_CONTROL,
        targetVelocity=0,
        force=knee_friction_force)

  def GetBasePosition(self):
    """Get the position of minitaur's base.

    Returns:
      The position of minitaur's base.
    """
    position, _ = (self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
    return position

  def GetBaseOrientation(self):
    """Get the orientation of minitaur's base, represented as quaternion.

    Returns:
      The orientation of minitaur's base.
    """
    _, orientation = (self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
    return orientation

  def GetActionDimension(self):
    """Get the length of the action list.

    Returns:
      The length of the action list.
    """
    return self.num_motors

  def GetObservationUpperBound(self):
    """Get the upper bound of the observation.

    Returns:
      The upper bound of an observation. See GetObservation() for the details
        of each element of an observation.
    """
    upper_bound = np.array([0.0] * self.GetObservationDimension())
    upper_bound[0:self.num_motors] = math.pi  # Joint angle, maximum pi radians
    upper_bound[self.num_motors:2 * self.num_motors] = (motor.MOTOR_SPEED_LIMIT)  # Joint velocity.
    upper_bound[2 * self.num_motors:3 * self.num_motors] = (motor.OBSERVED_TORQUE_LIMIT
                                                           )  # Joint torque.
    upper_bound[3 * self.num_motors:] = 1.0  # Quaternion of base orientation.
    return upper_bound

  def GetObservationLowerBound(self):
    """Get the lower bound of the observation."""
    return -self.GetObservationUpperBound()

  def GetObservationDimension(self):
    """Get the length of the observation list.

    Returns:
      The length of the observation list.
    """
    return len(self.GetObservation())

  def GetObservation(self):
    """Get the observations of minitaur.

    It includes the angles, velocities, torques and the orientation of the base.

    Returns:
      The observation list. observation[0:8] are motor angles. observation[8:16]
      are motor velocities, observation[16:24] are motor torques.
      observation[24:28] is the orientation of the base, in quaternion form.
    """
    observation = []
    observation.extend(self.GetMotorAngles().tolist())
    observation.extend(self.GetMotorVelocities().tolist())
    observation.extend(self.GetMotorTorques().tolist())
    observation.extend(list(self.GetBaseOrientation()))
    return observation

  def ApplyAction(self, motor_commands):
    """Set the desired motor angles to the motors of the minitaur.

    The desired motor angles are clipped based on the maximum allowed velocity.
    If the pd_control_enabled is True, a torque is calculated according to
    the difference between current and desired joint angle, as well as the joint
    velocity. This torque is exerted to the motor. For more information about
    PD control, please refer to: https://en.wikipedia.org/wiki/PID_controller.

    Args:
      motor_commands: The eight desired motor angles.
    """

    # clip motor angle commands at min/max velocity for 1 timestep - the desired angle must be reachable within 1 timestep at max velocity
    if self._motor_velocity_limit < np.inf:
      current_motor_angle = self.GetMotorAngles()
      motor_commands_max = (current_motor_angle + self.time_step * self._motor_velocity_limit) # constraint to move 1 time step at max velocity
      motor_commands_min = (current_motor_angle - self.time_step * self._motor_velocity_limit) # constraint to move 1 time step at min velocity
      motor_commands = np.clip(motor_commands, motor_commands_min, motor_commands_max)

    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      q = self.GetMotorAngles() # q = position
      qdot = self.GetMotorVelocities() # qdot = velocity

      if self._accurate_motor_model_enabled:
        actual_torque, observed_torque = self._motor_model.convert_to_torque(
            motor_commands, q, qdot)
        
        # shutdown motor if too much torque is applied for too long
        if self._motor_overheat_protection:
          for i in range(self.num_motors):
            if abs(actual_torque[i]) > OVERHEAT_SHUTDOWN_TORQUE:
              self._overheat_counter[i] += 1
            else:
              self._overheat_counter[i] = 0
            if (self._overheat_counter[i] > OVERHEAT_SHUTDOWN_TIME / self.time_step):
              self._motor_enabled_list[i] = False

        # The torque is already in the observation space because we use
        # GetMotorAngles and GetMotorVelocities.
        self._observed_motor_torques = observed_torque

        # Transform into the motor space when applying the torque.
        self._applied_motor_torque = np.multiply(actual_torque, self._motor_direction)

        for motor_id, motor_torque, motor_enabled in zip(self._motor_id_list,
                                                         self._applied_motor_torque,
                                                         self._motor_enabled_list):
          if motor_enabled:
            self._SetMotorTorqueById(motor_id, motor_torque)
          else:
            self._SetMotorTorqueById(motor_id, 0)
      

      else:
        # calculate the required torque (force that causes motor to turn) from the current and desired motor angles, taking velocity into account
        # assume observed and actual torque are the same

        # -kp * (currentAngles - motor_commands) - kd * currentVelocities
        torque_commands = -self._kp * (q - motor_commands) - self._kd * qdot

        # The torque is already in the observation space because we use
        # GetMotorAngles and GetMotorVelocities.
        self._observed_motor_torques = torque_commands

        # Transform into the motor space when applying the torque.
        self._applied_motor_torques = np.multiply(self._observed_motor_torques,
                                                  self._motor_direction)

        for motor_id, motor_torque in zip(self._motor_id_list, self._applied_motor_torques):
          self._SetMotorTorqueById(motor_id, motor_torque)
    
    # simple controller: just turn motor to desired angle, in the right direction
    # clipping above ensures it turns no faster than max motor velocity
    else:
      # calculate signed angles
      motor_commands_with_direction = np.multiply(motor_commands, self._motor_direction) # motor angles x [-1,1,-1 etc]
      for motor_id, motor_command_with_direction in zip(self._motor_id_list,
                                                        motor_commands_with_direction):
        
        # turn motor to desired angle
        self._SetDesiredMotorAngleById(motor_id, motor_command_with_direction)

  def GetMotorAngles(self):
    """Get the eight motor angles at the current moment.

    Returns:
      Motor angles.
    """
    motor_angles = [
        self._pybullet_client.getJointState(self.quadruped, motor_id)[0]
        for motor_id in self._motor_id_list
    ]
    motor_angles = np.multiply(motor_angles, self._motor_direction)
    return motor_angles

  def GetMotorVelocities(self):
    """Get the velocity of all eight motors.

    Returns:
      Velocities of all eight motors.
    """
    motor_velocities = [
        self._pybullet_client.getJointState(self.quadruped, motor_id)[1]
        for motor_id in self._motor_id_list
    ]
    motor_velocities = np.multiply(motor_velocities, self._motor_direction)
    return motor_velocities

  def GetMotorTorques(self):
    """Get the amount of torques the motors are exerting.

    Returns:
      Motor torques of all eight motors.
    """
    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      return self._observed_motor_torques
    else:
      motor_torques = [
          self._pybullet_client.getJointState(self.quadruped, motor_id)[3]
          for motor_id in self._motor_id_list
      ]
      motor_torques = np.multiply(motor_torques, self._motor_direction)
    return motor_torques

  def ConvertFromLegModel(self, actions):
    """Convert the actions that use leg model to the real motor actions.

    # convert from leg angles to motor angles

    Args:
      actions: The theta, phi of the leg model.
    Returns:
      The eight desired motor angles that can be used in ApplyActions().
    """

    motor_angle = copy.deepcopy(actions) # desired motor angles
    scale_for_singularity = 1 # ?
    offset_for_singularity = 1.5
    half_num_motors = int(self.num_motors / 2)
    quarter_pi = math.pi / 4 # 45 degree
    
    for i in range(self.num_motors):
      
      # [0, 1, 2, 3, 4, 5, 6, 7] for i in range(8)
      # [0, 0, 1, 1, 2, 2, 3, 3] for i//2 in range(8)
      # [1, -1, 1, -1, 1, -1, 1, -1] for (-1)**i in range(8)
      
      # this only ever addresses left legs of robot - can i assume actions are in motor order?
      action_idx = i // 2
    
      forward_backward_component = (
          -scale_for_singularity * quarter_pi *
          (actions[action_idx + half_num_motors] + offset_for_singularity))
    
      extension_component = (-1)**i * quarter_pi * actions[action_idx]
      
      if i >= half_num_motors: # flip for front and back right legs
        extension_component = -extension_component

      # theta in radians = pi + s + e (see paper), where e is negative for theta 2, so for every second motor
      motor_angle[i] = (math.pi + forward_backward_component + extension_component)
      # leg motor angle: for each leg, there are two motors, 0 degree is up for motor 1, down for motor 2.

      #                180 degree + 

      """
      motor angle order: 

     'motor_front_leftL_joint', 0
     'motor_front_leftR_joint', 1

     'motor_back_leftL_joint', 2
     'motor_back_leftR_joint', 3

     'motor_front_rightL_joint', 4
     'motor_front_rightR_joint', 5

     'motor_back_rightL_joint', 6
     'motor_back_rightR_joint', 7
     """



    return motor_angle

  def GetBaseMassFromURDF(self):
    """Get the mass of the base from the URDF file."""
    return self._base_mass_urdf

  def GetLegMassesFromURDF(self):
    """Get the mass of the legs from the URDF file."""
    return self._leg_masses_urdf

  def SetBaseMass(self, base_mass):
    self._pybullet_client.changeDynamics(self.quadruped, BASE_LINK_ID, mass=base_mass)

  def SetLegMasses(self, leg_masses):
    """Set the mass of the legs.

    A leg includes leg_link and motor. All four leg_links have the same mass,
    which is leg_masses[0]. All four motors have the same mass, which is
    leg_masses[1].

    Args:
      leg_masses: The leg masses. leg_masses[0] is the mass of the leg link.
        leg_masses[1] is the mass of the motor.
    """
    for link_id in LEG_LINK_ID:
      self._pybullet_client.changeDynamics(self.quadruped, link_id, mass=leg_masses[0])
    for link_id in MOTOR_LINK_ID:
      self._pybullet_client.changeDynamics(self.quadruped, link_id, mass=leg_masses[1])

  def SetFootFriction(self, foot_friction):
    """Set the lateral (left-right) friction of the feet.

    Args:
      foot_friction: The lateral friction coefficient of the foot. This value is
        shared by all four feet.
    """
    for link_id in FOOT_LINK_ID:
      self._pybullet_client.changeDynamics(self.quadruped, link_id, lateralFriction=foot_friction)

  def SetBatteryVoltage(self, voltage):
    if self._accurate_motor_model_enabled:
      self._motor_model.set_voltage(voltage)

  def SetMotorViscousDamping(self, viscous_damping):
    if self._accurate_motor_model_enabled:
      self._motor_model.set_viscous_damping(viscous_damping)
