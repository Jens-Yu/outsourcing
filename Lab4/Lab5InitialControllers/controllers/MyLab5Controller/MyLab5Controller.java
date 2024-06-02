// MyLab5Controller.java
/*
 * MyLab5Controller Class Definition
 * File: MyLab5Controller.java
 * Date: 15th Oct 2022
 * Description: Simple Controller based on the Lab4 controller (2022)
 * Author: Terry Payne (trp@liv.ac.uk)
 */

import com.cyberbotics.webots.controller.Supervisor;
import com.cyberbotics.webots.controller.Camera;

public class MyLab5Controller {

  public static void main(String[] args) {

    Supervisor robot = new Supervisor();
    int timeStep = (int) Math.round(robot.getBasicTimeStep());
    
    Pose robot_pose = new Pose(0.0, 0.0, 0.0);
    PioneerNav2 nav = new PioneerNav2(robot, robot_pose);
    
    double time_elapsed = 0;
    double target_time = 0;
    double robot_velocity = 0.3;
    
    // define schedule
    PioneerNav2.MoveState[] schedule = { PioneerNav2.MoveState.ARC };
    int schedule_index = -1; // we increment before selecting the current action
    PioneerNav2.MoveState state; // current state

    while (robot.step(timeStep) != -1) {
      state = nav.getState();
      if (time_elapsed > target_time) {
        time_elapsed = 0;
            
        // select next action in schedule if not stopped
        schedule_index = (schedule_index + 1) % schedule.length;
        state = schedule[schedule_index];
        
        if (state == PioneerNav2.MoveState.FORWARD) {
          target_time = nav.forward(0.8, robot_velocity);
        } else 
        if (state == PioneerNav2.MoveState.ARC) {
          target_time = nav.arc(Math.PI/2.0, 0.0, robot_velocity);
        } else
        if (state == PioneerNav2.MoveState.STOP) {
          nav.stop();
          target_time = 60 * 1000; // This doesn't really stop, but pauses for 1 minute
        }
      } else
        time_elapsed += timeStep;    // Increment by the time state
      
    };
    // Enter here exit cleanup code.
  }
}
