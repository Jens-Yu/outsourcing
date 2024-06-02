// PioneerNav2.java
/*
 * PioneerNavigation Class Definition
 * File: PioneerNav2.java
 * Date: 18th Oct 2022
 * Description: Simple Navigation Class support (2022)
 * Author: Terry Payne (trp@liv.ac.uk)
 */
 
import com.cyberbotics.webots.controller.Motor;
import com.cyberbotics.webots.controller.Supervisor;
import com.cyberbotics.webots.controller.Node;

public class PioneerNav2 {

  public static enum MoveState {
    STOP,
    FORWARD,
    ARC };

  private Supervisor robot;   // reference to the robot
  private Node robot_node;    // reference to the robot node
  private Pose robot_pose;    // the robots percieved pose
  private Motor left_motor;
  private Motor right_motor;
  
  private MoveState state;

  private final double WHEEL_RADIUS = 0.0957; // in meters - found using CONFIGURE 
  private final double AXEL_LENGTH = 0.323;   // in meters - found using CONFIGURE


  // ==================================================================================
  // Constructor
  // ==================================================================================
  public PioneerNav2(Supervisor robot, Pose init_pose) {
    this.robot = robot;    
    this.robot_node = this.robot.getSelf();   // reference to the robot node
    this.state = MoveState.STOP;

    // enable motors
    this.left_motor = robot.getMotor("left wheel");
    this.right_motor = robot.getMotor("right wheel");
    this.left_motor.setPosition(Double.POSITIVE_INFINITY);
    this.right_motor.setPosition(Double.POSITIVE_INFINITY);
    
    int timeStep = (int) Math.round(robot.getBasicTimeStep());

    // set up pose
    this.robot_pose = new Pose(init_pose.getX(), init_pose.getY(), init_pose.getTheta());

    // Initialise motor velocity
    this.left_motor.setVelocity(0.0);
    this.right_motor.setVelocity(0.0);   
  } 
  

  // The following method only works if in supervisor mode
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