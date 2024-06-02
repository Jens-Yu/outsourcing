# PioneerNavigation Class Definition
# File: pioneer_nav1.py
# Date: 18th Oct 2022
# Description: Simple Navigation Class support (2022)
# Author: Terry Payne (trp@liv.ac.uk)

from controller import Supervisor, Node

import math
import pose
from enum import Enum

class MoveState(Enum):
    STOP = 0
    FORWARD = 1
    ARC = 2
    CONFIGURE = 3

class PioneerNavigation:
    """ A custom class to initialise and manage simple navigation on a Pioneer Adept robot """

    WHEEL_RADIUS = 0.0975   # in meters
    AXEL_LENGTH = 0.31      # in meters

    def __init__(self, robot):
        
        self.robot = robot                        # reference to the robot
        self.robot_node = self.robot.getSelf()    # reference to the robot node
        self.robot_pose = self.get_real_pose()   # the robots believed pose, based on real location
        self.state = MoveState.STOP
        self.configure_str="Configuring..."
    
        # enable motors
        self.left_motor = self.robot.getDevice('left wheel')
        self.right_motor = self.robot.getDevice('right wheel')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        # Initialise motor velocity
        self.left_motor.setVelocity(0.0);
        self.right_motor.setVelocity(0.0);   

    def get_real_pose(self):
        if self.robot_node is None:
            return pose.Pose(0, 0, 0)
            
        real_pos = self.robot_node.getPosition()
        rot = self.robot_node.getOrientation()
        theta = math.atan2(-rot[0], rot[3])
        halfpi = math.pi / 2
        theta2 = theta + halfpi
        if (theta > halfpi):
            theta2 = -(3 * halfpi) + theta
        
        return pose.Pose(real_pos[0], real_pos[1], theta2)
    

    def configure_initialise_parameters(self, icr_omega):
        # This rotates the robot about one wheel, and monitors the time and distance
        # taken to rotate around a circle
        
        # ADD CODE HERE
        self.state = MoveState.CONFIGURE

 
    def configure_check_parameters(self, timestep):
        # ADD CODE HERE
        return self.configure_str
    
    def stop(self):
    
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.state = MoveState.STOP
    