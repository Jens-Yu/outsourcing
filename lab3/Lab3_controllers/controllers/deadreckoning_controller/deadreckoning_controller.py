"""deadreckoning_controller controller."""

from controller import Supervisor
import pioneer_nav1 as pn
import math
from pioneer_nav1 import MoveState

def run_robot(robot):
        
    # get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    nav = pn.PioneerNavigation(robot)
    time_elapsed = 0
    target_time = 0
    robot_velocity = 0.3
    
    # define schedule
    schedule = [ MoveState.CONFIGURE, MoveState.STOP ]
    schedule_index = -1 # we increment before selecting the current action
        
    display_action = ""

    if (schedule[0] == MoveState.CONFIGURE):
        schedule_index = schedule_index + 1
        nav.configure_initialise_parameters(2*math.pi)
      
    while robot.step(timestep) != -1:
        if (nav.state == MoveState.CONFIGURE):
            # Special case for checking robot parameters
            display_action = nav.configure_check_parameters(timestep)
            
        elif (nav.state == MoveState.STOP):
            nav.stop()
            display_action = "Stop"
            break

        true_pose = nav.get_real_pose()
        print(f"Action: {display_action} \tTrue Pose: {true_pose}")

   
if __name__ == "__main__":
    # create the Supervised Robot instance.
    my_robot = Supervisor()
    run_robot(my_robot)

# Aims
# Create a state machine that takes a schedule of commands and runs indefinitely
# Display current move and position on display