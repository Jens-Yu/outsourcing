// File: PioneerProxSensors1.java
// Date: 28th Oct 2021
// Description: Represent the local area of the adept robot given its sonar sensors
// Author: Terry Payne
// Modifications:
//    * Based on SensorView.java which was developed on 28th Oct 2021 and then 
//      Updated for the Programming Assignment 2021 (29th Nov 20201)
//    * Builds upon the Python version (pioneer_proxsensors.py - 24th Jan 2022)
//      that integrates the Pioneer sensors into the view for cleaner code
//

import com.cyberbotics.webots.controller.DistanceSensor;
import com.cyberbotics.webots.controller.Display;
import com.cyberbotics.webots.controller.Robot;

public class PioneerProxSensors1 {
  // --------------------------------
  // Robot state variables  
  private Robot robot;          // Reference to the Robot itself
  private Pose robot_pose;       // we only track the orientation, as the robot is always centered
  private double radius;
  
  // --------------------------------
  // Display state variables  
  private Display display;      // reference to the display device on the robot
  private int device_width;     // width of the display device
  private int device_height;    // height of the display device
  private double scaleFactor;   // Scale factor to scale rendered map to the maximal dimension on the display

  // --------------------------------
  // Distance Sensor state variables  
  private DistanceSensor[] ps;  // array of distance sensors attached to the robot
  private double maxRange;      // we'll get this from the lookup table of so0
  private double maxValue;      // we'll get this from the parameters of so0

  private Pose[] psPose;        // the pose of each sensor (assuming the robot is a round cylinder)
	
  // --------------------------------
  // Colours used by the display
  private final static int DARKGREY = 0x3C3C3C;
  private final static int BLACK = 0x000000;
  private final static int WHITE = 0xFFFFFF;
  
  private final static int MAX_NUM_SENSORS = 16;            // Number of sensors on the robot

  // ==================================================================================
  // Constructors
  // ==================================================================================
  public PioneerProxSensors1(Robot r, String display_name, Pose p) {
    this.robot = r;      
    this.robot_pose = p;
    
    // get the time step of the current world.
    int timeStep = (int) Math.round(this.robot.getBasicTimeStep());    

    // Note that the dimensions of the robot are not strictly circular, as 
    // according to the data sheet the length is 485mm, and width is 381mm
    // so we assume for now the aprox average of the two (i.e. 430mm), in meters
    this.radius = 0.215;         // in meters
  
    // Insert Constructor Code Here
    
  }
  
  // ==================================================================================
  // Internal (Private) methods
  // ==================================================================================
  // Map the real coordinates to screen coordinates assuming
  // the origin is in the center and y axis is inverted
  private int scale(double l) {
    return (int) (this.scaleFactor * l);
  }
  private int mapX(double x) {
    return (int) ((device_width/2.0) + scale(x));
  }
  private int mapY(double y) {
    return (int) ((device_height/2.0) - scale(y));
  }
  private double rotX(double x, double y, double theta) {
    return Math.cos(theta)*x - Math.sin(theta)*y;
  }
  private double rotY(double x, double y, double theta) {
    return Math.sin(theta)*x + Math.cos(theta)*y;
  }

  // ==================================================================================
  // External (Public) methods
  // ==================================================================================  
      
  // Insert public methods here

  public void set_pose(Pose p) {
    this.robot_pose.setPosition(p.getX(), p.getY(), p.getTheta());
  } 

  public void paint() {
    if (this.display == null)
      return;

    // ===================================================================================
    // draw a background
    this.display.setColor(0xF0F0F0);     // Off White
    this.display.fillRectangle(0, 0, this.device_width, this.device_height);
 
    // Draw Robot Body      
    this.display.setColor(WHITE);     // White
    this.display.fillOval(mapX(0.0),
                          mapY(0.0),
                          scale(this.radius),
                          scale(this.radius));
    
    this.display.setColor(DARKGREY);     // Dark Grey
    this.display.drawOval(mapX(0.0),
                          mapY(0.0),
                          scale(this.radius),
                          scale(this.radius));
                          
    // Need to indicate heading          
    this.display.drawLine(mapX(0.0),
                          mapY(0.0),
                          mapX(Math.cos(this.robot_pose.getTheta()) * this.radius),
                          mapY(Math.sin(this.robot_pose.getTheta()) * this.radius));

    this.display.setColor(BLACK);         // Black
    this.display.setFont("Arial", 8, true);  // font size = 8, with antialiasing
          
    // Insert View based code here
  
  }
}
