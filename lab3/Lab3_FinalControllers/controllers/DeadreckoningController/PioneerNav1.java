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

  private final double WHEEL_RADIUS = 0.0957; // in meters - found using CONFIGURE 
  private final double AXEL_LENGTH = 0.323;   // in meters - found using CONFIGURE


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

  public void configure_initialise_parameters(double wheel_omega) {
    // This rotates the robot about one wheel, and monitors the time and distance
    // taken to rotate around a circle
    
    Pose pose = this.get_real_pose();
    this.velocity = wheel_omega;
    this.config_max_x = pose.getX();
    this.config_min_x = pose.getX();
    this.config_timer = 0;
    this.config_prev_theta = pose.getTheta();
        
    this.left_motor.setVelocity(0.0);
    this.right_motor.setVelocity(this.velocity);
    this.state = MoveState.CONFIGURE;
  }
  
  public String configure_check_parameters(double timestep) {
    Pose pose = this.get_real_pose();
    double axel_length = this.config_max_x - this.config_min_x; // represents a circle travelled by center of robot
        
    if ((this.config_prev_theta < 0) && (pose.getTheta() >= 0.0)) {
      // We have just passed a heading of 0
      double distance = axel_length * 2 * Math.PI; // i.e. circumference around icr
      double wheel_radius = distance / (this.velocity * (this.config_timer / 1000));
      this.configure_str = String.format("axel_length=%.3f wheel_radius=%.3f", axel_length, wheel_radius);
      this.configure_initialise_parameters(this.velocity);
      this.config_timer = 0;
    } else {    
      this.config_max_x = Math.max(this.config_max_x, pose.getX());
      this.config_min_x = Math.min(this.config_min_x, pose.getX());
    }
    this.config_timer += timestep;
    this.config_prev_theta = pose.getTheta();
    return this.configure_str;
  }
  
  public int forward(double target_dist, double robot_linearvelocity) {
    double wheel_av = (robot_linearvelocity/this.WHEEL_RADIUS);
    double target_time = target_dist/robot_linearvelocity;
    
    this.left_motor.setVelocity(wheel_av);
    this.right_motor.setVelocity(wheel_av);
    this.state = MoveState.FORWARD;
        
    // return target_time as millisecs          
    return (int) (1000.0*target_time);
  }

  public int arc(double icr_angle, double icr_r, double icr_omega) {
    double target_time = icr_angle / icr_omega;

    // Calculate each wheel velocity around ICR
    double vl = icr_omega * (icr_r - (this.AXEL_LENGTH / 2));
    double vr = icr_omega * (icr_r + (this.AXEL_LENGTH / 2));
        
    double leftwheel_av = (vl/this.WHEEL_RADIUS);
    double rightwheel_av = (vr/this.WHEEL_RADIUS);
        
    this.left_motor.setVelocity(leftwheel_av);
    this.right_motor.setVelocity(rightwheel_av);
    this.state = MoveState.ARC;

    // return target_time as millisecs          
    return (int) (1000.0*target_time);
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