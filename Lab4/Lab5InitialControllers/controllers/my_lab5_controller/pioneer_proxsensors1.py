# PioneerProxSensors Class Definition
# File: pioneer_proxsensors.py
# Date: 24th Jan 2022
# Description: Pose Class support for COMP329 Programming Assignment (2022)
# Author: Terry Payne (trp@liv.ac.uk)

#import numpy as np
import math
import pose


class PioneerProxSensors:
    """ A custom class to manage the 16 proximity sensors on a Pioneer Adept robot """

    # define the display constants
    DARKGREY = 0x3C3C3C
    BLACK = 0x000000
    WHITE = 0xFFFFFF
    LABEL_OFFSET = 0.3
    MAX_NUM_SENSORS = 16

    def __init__(self, robot, display_name, robot_pose):
        self.robot = robot
        self.robot_pose = robot_pose
        
        timestep = int(robot.getBasicTimeStep())

        # Dimensions of the Robot
        # Note that the dimensions of the robot are not strictly circular, as 
        # according to the data sheet the length is 485mm, and width is 381mm
        # so we assume for now the aprox average of the two (i.e. 430mm), in meters
        self.radius = 0.215

        # Insert Constructor Code Here
 
    # ==================================================================================
    # Internal (Private) methods
    # ==================================================================================

    # helper methods for mapping to the display
    # Map the real coordinates to screen coordinates assuming
    #   the origin is in the center and y axis is inverted

    def scale(self, l):
        return int(l * self.scalefactor)
    def mapx(self, x):
        return int((self.device_width / 2.0) + self.scale(x))
    def mapy(self, y):
        return int((self.device_height / 2.0) - self.scale(y))
    def rotx(self, x, y, theta):
        return math.cos(theta) * x - math.sin(theta) * y
    def roty(self, x, y, theta):
        return math.sin(theta) * x + math.cos(theta) * y

    # ==================================================================================
    # External (Public) methods
    # ==================================================================================
  
    # Insert public methods here

    def set_pose(self, p):
        self.robot_pose.set_position(p.x, p.y, p.theta)

    def paint(self):
        if self.display is None:
            return
        
        # draw a background
        self.display.setColor(0xF0F0F0)
        self.display.fillRectangle(0, 0, self.device_width, self.device_height)

        theta = self.robot_pose.theta
        
        # draw robot body
        self.display.setColor(self.WHITE)
        self.display.fillOval(self.mapx(0.0),
                              self.mapy(0.0),
                              self.scale(self.radius),
                              self.scale(self.radius))
                              
        self.display.setColor(self.DARKGREY)
        self.display.drawOval(self.mapx(0.0),
                              self.mapy(0.0),
                              self.scale(self.radius),
                              self.scale(self.radius))
        # Need to indicate heading          
        self.display.drawLine(self.mapx(0.0),
                              self.mapy(0.0),
                              self.mapx(math.cos(self.robot_pose.theta) * self.radius),
                              self.mapy(math.sin(self.robot_pose.theta) * self.radius))                             

        # Insert View based code here
          
 