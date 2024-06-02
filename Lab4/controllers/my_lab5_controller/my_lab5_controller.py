"""my_lab4_controller controller."""

# my_lab4_controller Class Definition
# File: my_lab4_controller.py
# Date: 15th Nov 2022
# Description: Simple Controller based on 2021 version (2022)
# Author: Terry Payne (trp@liv.ac.uk)
#

from controller import Supervisor
import pioneer_nav2 as pn
import pioneer_proxsensors1 as pps
import math
import pose

from pioneer_nav2 import MoveState

  
def run_robot(robot):
        
    # get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())
    
    robot_pose = pose.Pose(0.0, 0.0, 0.0)

    
    camera = robot.getDevice('camera')
    if camera is not None:
        camera.enable(timestep)

    prox_sensors = pps.PioneerProxSensors(robot, "sensor_display", robot_pose);
    nav = pn.PioneerNavigation(robot, robot_pose, prox_sensors)
    time_elapsed = 0
    target_time = 0
    robot_velocity = 0.3

    # define schedule
    schedule = [ MoveState.FOLLOW_WALL ]
    schedule_index = -1 # we increment before selecting the current action
     
    while robot.step(timestep) != -1:
        # Testing out the front proximity sensors

        robot_pose = nav.get_real_pose()
        prox_sensors.set_pose(robot_pose)
        prox_sensors.paint()
                    
        state = nav.state       
        if (time_elapsed > target_time):
            time_elapsed = 0
            
            # select next action in schedule if not stopped
            schedule_index = (schedule_index +1) % len(schedule)
            nav.state = schedule[schedule_index]
            
            if (nav.state == MoveState.FORWARD):
                target_time = nav.forward(0.8, robot_velocity)
            elif (nav.state == MoveState.ARC):
                target_time = nav.arc(math.pi/2.0, 0.0, robot_velocity)
            elif (nav.state == MoveState.STOP):
                nav.stop()
                target_time = 60 * 1000 # This doesn't really stop, but pauses for 1 minute
            elif (nav.state == MoveState.FOLLOW_WALL):
                target_time = 0  # always refresh!
                nav.follow_wall(robot_velocity, 0.2, False)
        else:
            time_elapsed += timestep    # Increment by the time state

   
if __name__ == "__main__":
    # create the Supervised Robot instance.
    my_robot = Supervisor()
    run_robot(my_robot)
