// PioneerNav1.java
/*
 * PioneerNavigation Class Definition
 * File: pioneer_nav1.py
 * Date: 18th Oct 2022
 * Description: Simple Navigation Class support (2022)
 * Author: Terry Payne (trp@liv.ac.uk)
 */
 
import com.cyberbotics.webots.controller.Motor;
import com.cyberbotics.webots.controller.Supervisor;
import com.cyberbotics.webots.controller.Node;

public class PioneerNav1 {

  public static enum MoveState {
    STOP,
    FORWARD,
    ARC,
    CONFIGURE };

  private Supervisor robot;       // reference to the robot
  private Node robot_node;        // reference to the robot node
  private Pose robot_pose;        // the robots believed pose, based on real location
  private Motor left_motor;
  private Motor right_motor;
  private String configure_str; 
  private MoveState state;
  
  private double velocity;
  private double config_max_x;
  private double config_min_x;
  private double config_timer;
  private double config_prev_theta;


  private final double WHEEL_RADIUS = 0.0975;   // in meters
  private final double AXEL_LENGTH = 0.31;      // in meters


  // ==================================================================================
  // Constructor
  // ==================================================================================
  public PioneerNav1(Supervisor robot) {
    this.robot = robot;                       // reference to the robot
    this.robot_node = this.robot.getSelf();   // reference to the robot node
    this.robot_pose = this.get_real_pose();   // the robots believed pose, based on real location
    this.state = MoveState.STOP;
    this.configure_str="Configuring...";

    // enable motors
    this.left_motor = robot.getMotor("left wheel");
    this.right_motor = robot.getMotor("right wheel");
    this.left_motor.setPosition(Double.POSITIVE_INFINITY);
    this.right_motor.setPosition(Double.POSITIVE_INFINITY);

    // Initialise motor velocity
    this.left_motor.setVelocity(0.0);
    this.right_motor.setVelocity(0.0);   
  } 
  
  public Pose get_real_pose() {
    if (this.robot_node == null)
      return new Pose(0,0,0);
      
    double[] realPos = robot_node.getPosition();
    double[] rot = this.robot_node.getOrientation(); // 3x3 Rotation matrix as vector of length 9
    double theta1 = Math.atan2(-rot[0], rot[3]);
    double halfPi = Math.PI/2;
    double theta2 = theta1 + halfPi;
    if (theta1 > halfPi)
        theta2 = -(3*halfPi)+theta1;
    
    return new Pose(realPos[0], realPos[1], theta2);
  }

  public void configure_initialise_parameters(double icr_omega) {
    // This rotates the robot about one wheel, and monitors the time and distance
    // taken to rotate around a circle
    
    // ADD CODE HERE
    
    this.state = MoveState.CONFIGURE;
  }
 
  public String configure_check_parameters(double timestep) {

    // ADD CODE HERE
    
    return this.configure_str;
  }
  
  public void stop() {
    this.left_motor.setVelocity(0.0);
    this.right_motor.setVelocity(0.0);
    this.state = MoveState.STOP;
  }
  
  public MoveState getState() {
    return this.state;
  }
}    