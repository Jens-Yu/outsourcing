// DeadreckoningController.java
/*
 * DeadreckoningController Class Definition
 * File: DeadreckoningController.java
 * Date: 15th Oct 2022
 * Description: Simple Controller based on 2021 version (2022)
 * Author: Terry Payne (trp@liv.ac.uk)
 */

import com.cyberbotics.webots.controller.Supervisor;
import com.cyberbotics.webots.controller.Display;

public class DeadreckoningController {

  public static void main(String[] args) {

    Supervisor robot = new Supervisor();
    int timeStep = (int) Math.round(robot.getBasicTimeStep());
    
    PioneerNav1 nav = new PioneerNav1(robot);

    double time_elapsed = 0;
    double target_time = 0;
    double robot_velocity = 0.3;
    
    // define schedule
    //PioneerNav1.MoveState[] schedule = { PioneerNav1.MoveState.CONFIGURE };
    PioneerNav1.MoveState[] schedule = { PioneerNav1.MoveState.FORWARD, PioneerNav1.MoveState.ARC };
    int schedule_index = -1; // we increment before selecting the current action
    PioneerNav1.MoveState state; // current state

    // set up display
    Display odometry_display = robot.getDisplay("odometryDisplay");
    String display_action = "";

    if (schedule[0] == PioneerNav1.MoveState.CONFIGURE) {
      schedule_index++;
      nav.configure_initialise_parameters(2*Math.PI);
    }


    while (robot.step(timeStep) != -1) {
      state = nav.getState();
      if (state == PioneerNav1.MoveState.CONFIGURE) {
        // Special case for checking robot parameters
       display_action = nav.configure_check_parameters(timeStep);
      } else if (time_elapsed > target_time) {
        time_elapsed = 0;
            
        // select next action in schedule if not stopped
        schedule_index = (schedule_index + 1) % schedule.length;
        state = schedule[schedule_index];
        
        if (state == PioneerNav1.MoveState.FORWARD) {
          target_time = nav.forward(0.5, robot_velocity);
          display_action = "Forward Action: 0.5m";
        } else 
        if (state == PioneerNav1.MoveState.ARC) {
          target_time = nav.arc(Math.PI/2.0, 0.7, robot_velocity);
          display_action = "Arc Action around an ICR 0.7 away";
        } else
        if (state == PioneerNav1.MoveState.CONFIGURE) {
          display_action = "Determine Wheel / Axel Parameters";
        } else 
        if (state == PioneerNav1.MoveState.STOP) {
          nav.stop();
          display_action = "Stop for 1 minute";
          target_time = 60 * 1000; // This doesn't really stop, but pauses for 1 minute
        }
      } else
        time_elapsed += timeStep;    // Increment by the time state

      
      if (odometry_display != null) {
        odometry_display.setColor(0xFFFFFF);          // White
        odometry_display.fillRectangle(0,0,
                odometry_display.getWidth(),
                odometry_display.getHeight());
    
        odometry_display.setColor(0x000000);          // Black
        odometry_display.setFont("Arial", 18, true);   // font size = 18, with antialiasing
        odometry_display.drawText("Robot State", 1, 1);

        odometry_display.setFont("Arial", 12, true);
        if (display_action != "")
          odometry_display.drawText(display_action, 1, 30);
                    
        Pose true_pose = nav.get_real_pose();
        odometry_display.drawText("True Pose: "+true_pose, 1, 50);
      }
    };

    // Enter here exit cleanup code.
  }
}
