// File:          DeadreckoningController.java
// Date:
// Description:
// Author:
// Modifications:

// You may need to add other webots classes such as
//  import com.cyberbotics.webots.controller.DistanceSensor;
//  import com.cyberbotics.webots.controller.Motor;
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
    PioneerNav1.MoveState[] schedule = { PioneerNav1.MoveState.CONFIGURE, PioneerNav1.MoveState.STOP };
    int schedule_index = -1; // we increment before selecting the current action
    PioneerNav1.MoveState state; // current state

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
      } else if (state == PioneerNav1.MoveState.STOP) {
          nav.stop();
          display_action = "Stop";
      }
      
      Pose true_pose = nav.get_real_pose();
      System.out.println("Action: " + display_action + " \t" + "True Pose: "+true_pose);
    };

    // Enter here exit cleanup code.
  }
}
