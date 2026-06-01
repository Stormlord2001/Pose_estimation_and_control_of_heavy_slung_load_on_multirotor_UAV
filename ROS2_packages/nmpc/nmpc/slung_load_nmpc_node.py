from nmpc.controllers.slung_load_nmpc import SlungLoadNMPC
from nmpc.models.slung_load_model import SlungLoadModel 

__author__ = "Kristoffer Jensen"
__contact__ = "krisj21@student.sdu.dk"

import time
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock

from mavros.base import SENSOR_QOS
from mavros_msgs.srv import CommandBool
from mavros_msgs.msg import AttitudeTarget, Thrust
from geometry_msgs.msg import PoseStamped, Vector3, Quaternion, TwistStamped, Twist, Vector3Stamped, Pose
from nav_msgs.msg import Path
from nmpc_interfaces.msg import State
from std_msgs.msg import Header

from tf_transformations import euler_from_quaternion, quaternion_from_euler

class NMPC(Node):
    def __init__(self):
        super().__init__('slung_load_node')

        #########################################################################
        # Choose trajectory type for the reference trajectory
        #   Options: 'circular', 'setpoint'
        #########################################################################
        #self.trajectory_type = 'circular' 
        #self.trajectory_type = 'setpoint'
        self.trajectory_type = 'setpoint_test'

        # Subscribe to estimated vehicle state from state estimator
        self.state_sub = self.create_subscription(
            State,
            'nmpc/estimated_state_ukf',
            self.state_callback,
            SENSOR_QOS
        )

        # Publisher to send bodyrate commands to the vehicle
        self.body_rate_publisher = self.create_publisher(
            AttitudeTarget,
            '/mavros/setpoint_raw/attitude',
            SENSOR_QOS
        )

        # Publisher to send predicted trajectory for visualization
        self.predicted_trajectory_publisher = self.create_publisher(
            Path,
            'nmpc/predicted_trajectory',
            SENSOR_QOS
        )

        # Publisher to send predicted control input bodyrates for visualization
        self.predicted_bodyrate_publisher = self.create_publisher(
            Path,
            'nmpc/predicted_bodyrates',
            SENSOR_QOS
        )

        # Publisher to send the predicted drone trajectory for visualization
        self.predicted_drone_trajectory_publisher = self.create_publisher(
            Path,
            'nmpc/predicted_drone_trajectory',
            SENSOR_QOS
        )

        self.setpoint_pose_pub = self.create_publisher(
            Pose,
            'nmpc/setpoint_pose',
            SENSOR_QOS
        )

        # Timer for the command loop
        timer_period = 0.02  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)

        # Create quadrotor bodyrate model
        self.model = SlungLoadModel()

        # Create NMPC controller
        self.nmpc = SlungLoadNMPC(self.model)

        # x = {xL, vL, q, w, qe}
        self.xL = np.array([0.0, 0.0, 0.0])
        self.vL = np.array([0.0, 0.0, 0.0])
        self.q = np.array([0.0, 0.0, -1.0])
        self.w = np.array([0.0, 0.0, 0.0])
        self.qe = np.array([1, 0.0, 0.0, 0.0])

        self.state = np.hstack((self.xL, self.vL, self.q, self.w, self.qe))

        self.setpoint_position = np.array([0.0, 0.0, 1.0]) # Desired load position

        # Get starttime for reference trajectory generation in seconds
        self.start_time = self.get_clock().now()

        self.frames = 0
        self.processing_time = 0

    def state_callback(self, msg):
        self.xL = np.array([msg.load_position])
        self.vL = np.array([msg.load_velocity])
        self.q = np.array([msg.cable_vector])
        self.q = self.q / np.linalg.norm(self.q) # Normalize vector
        self.w = np.array([msg.load_angular_velocity])
        self.qe = np.array([msg.drone_attitude])

        self.state = np.hstack((self.xL, self.vL, self.q, self.w, self.qe))
        #print(np.shape(self.state), self.state)

    def cmdloop_callback(self):
        start_time = time.time()
        # Create the initial state vector
        x0 = self.state.reshape(16,1)

        # Get current time in seconds for reference trajectory generation
        t_now = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        # Update reference trajectory based on chosen trajectory type
        if self.trajectory_type == 'circular':
            # Update the reference trajectory in the OCP solver
            self.nmpc.circular_reference_trajectory(radius=1, angular_velocity=0.6, t_now=t_now)
        elif self.trajectory_type == 'setpoint':
            # Solve the NMPC problem for the current state and setpoint
            self.nmpc.set_setpoint(self.setpoint_position)
            print("Setpoint: 0.0, 0.0, 1.0")
        elif self.trajectory_type == 'setpoint_test':
            # Change between two setpoints every 10 seconds to test the response of the controller
            if int(t_now) % 30 < 15:
                self.nmpc.set_setpoint(np.array([0.0, 2.5, 1]))
                pose = Pose()
                pose.position.x = 0.0
                pose.position.y = 2.5
                pose.position.z = 0.75
                self.setpoint_pose_pub.publish(pose)
                self.get_logger().info("UKF: Setpoint: [0.0, 2.5, 0.75]")
            else:
                self.nmpc.set_setpoint(np.array([0.0, -2.5, 0.75]))
                pose = Pose()
                pose.position.x = 0.0
                pose.position.y = -2.5
                pose.position.z = 0.75
                self.setpoint_pose_pub.publish(pose)
                self.get_logger().info("UKF: Setpoint: [0.0, -2.5, 0.75]")

        # Try to solve the NMPC problem
        try:
            u_pred, x_pred = self.nmpc.solve(x0)
        except Exception as e:
            self.get_logger().error(f"NMPC solver failed: {e}")
            return

        # Publish the predicted trajectory on a ros2 topic for visualization
        self.publish_predicted_trajectory(x_pred)
        
        # Get the first control input from the predicted control inputs
        u = u_pred[0, :]

        # Scale thrust to the range [0, 1]         
        ########################################################################
        # TODO: NEEDS TO BE CALIBRATED FOR THE SPECIFIC VEHICLE!
        # mass*g/throttle_at_hover
        ########################################################################
        thrust_command = (u[0] / self.model.max_thrust)
        ########################################################################

        # Create the bodyrate command message
        # To disable attitude control, we set the type_mask to IGNORE_ATTITUDE
        # https://docs.ros.org/en/noetic/api/mavros_msgs/html/msg/AttitudeTarget.html
        # The bodyrate is given in body frame with FLU convention
        bodyrate_command = AttitudeTarget()
        bodyrate_command.header.stamp = Clock().now().to_msg()
        bodyrate_command.type_mask = AttitudeTarget.IGNORE_ATTITUDE
        bodyrate_command.body_rate.x = float(u[1])
        bodyrate_command.body_rate.y = float(u[2])
        bodyrate_command.body_rate.z = float(u[3])
        bodyrate_command.thrust = float(thrust_command)

        # Publish the bodyrate command
        self.body_rate_publisher.publish(bodyrate_command)
        #self.get_logger().info(f"Yaw command: {u[3]}")
        #self.get_logger().info(f"Bodyrate command: {bodyrate_command}")
        self.get_logger().info(f"Bodyrate command: {u[1]:.4f}, {u[2]:.4f}, {u[3]:.4f}, {float(thrust_command)}")

        end_time = time.time()
        processing_time = end_time - start_time
        self.get_logger().info(f"Processing time: {processing_time:.3f} seconds")

        self.frames += 1
        self.processing_time += processing_time
        self.get_logger().info(f"average time: {self.processing_time/self.frames:.3f} seconds")

    def publish_predicted_trajectory(self, x_pred):
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map" 

        bodyrate_path_msg = Path()
        bodyrate_path_msg.header = Header()
        bodyrate_path_msg.header.stamp = self.get_clock().now().to_msg()
        bodyrate_path_msg.header.frame_id = "map" 

        drone_path_msg = Path()
        drone_path_msg.header = Header()
        drone_path_msg.header.stamp = self.get_clock().now().to_msg()
        drone_path_msg.header.frame_id = "map"

        for k in range(x_pred.shape[0]):
            pose = PoseStamped()
            pose.header = path_msg.header

            pose.pose.position.x = float(x_pred[k, 0])
            pose.pose.position.y = float(x_pred[k, 1])
            pose.pose.position.z = float(x_pred[k, 2])

            # If quaternion exists in state
            pose.pose.orientation.x = float(x_pred[k, 12])
            pose.pose.orientation.y = float(x_pred[k, 13])
            pose.pose.orientation.z = float(x_pred[k, 14])
            pose.pose.orientation.w = float(x_pred[k, 15])

            path_msg.poses.append(pose)

            bodyrate_pose = PoseStamped()
            bodyrate_pose.header = bodyrate_path_msg.header

            bodyrate_pose.pose.position.x = float(x_pred[k, 9])
            bodyrate_pose.pose.position.y = float(x_pred[k, 10])
            bodyrate_pose.pose.position.z = float(x_pred[k, 11])

            # Store bodyrates in orientation fields for visualization
            bodyrate_pose.pose.orientation.x = float(0.0)
            bodyrate_pose.pose.orientation.y = float(0.0)
            bodyrate_pose.pose.orientation.z = float(0.0)
            bodyrate_pose.pose.orientation.w = float(1.0)

            bodyrate_path_msg.poses.append(bodyrate_pose)

            # Calculate drone position from load position and cable vector
            drone_pose = PoseStamped()
            drone_pose.header = drone_path_msg.header
            drone_pose.pose.position.x = float(x_pred[k, 0] - self.model.L * x_pred[k, 6])
            drone_pose.pose.position.y = float(x_pred[k, 1] - self.model.L * x_pred[k, 7])
            drone_pose.pose.position.z = float(x_pred[k, 2] - self.model.L * x_pred[k, 8])
            drone_pose.pose.orientation.x = float(x_pred[k, 12])
            drone_pose.pose.orientation.y = float(x_pred[k, 13])
            drone_pose.pose.orientation.z = float(x_pred[k, 14])
            drone_pose.pose.orientation.w = float(x_pred[k, 15])
            drone_path_msg.poses.append(drone_pose)

        self.predicted_trajectory_publisher.publish(path_msg)
        self.predicted_bodyrate_publisher.publish(bodyrate_path_msg)
        self.predicted_drone_trajectory_publisher.publish(drone_path_msg)

def main(args=None):    
    rclpy.init(args=args)

    nmpc_node = NMPC()

    rclpy.spin(nmpc_node)
    
    nmpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
