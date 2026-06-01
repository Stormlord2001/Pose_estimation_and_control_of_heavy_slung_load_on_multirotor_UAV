from nmpc.controllers.slung_load_nmpc import SlungLoadNMPC
from nmpc.models.slung_load_model import SlungLoadModel 

__author__ = "Kristoffer Jensen"
__contact__ = "krisj21@student.sdu.dk"

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock

from mavros.base import SENSOR_QOS
from mavros_msgs.srv import CommandBool
from mavros_msgs.msg import AttitudeTarget, Thrust
from geometry_msgs.msg import PoseStamped, Pose, Vector3, Quaternion, TwistStamped, Twist, Vector3Stamped
from nav_msgs.msg import Path
from nmpc_interfaces.msg import State
from std_msgs.msg import Header

from tf_transformations import euler_from_quaternion, quaternion_from_euler

class e_stop_compensated_node(Node):
    def __init__(self):
        super().__init__('e_stop_compensated_node')

        # Subscribe to estimated vehicle state from state estimator
        self.state_sub = self.create_subscription(
            State,
            'nmpc/estimated_state_ukf',
            self.state_callback,
            SENSOR_QOS
        )

        # Subscribe to the drones position from the motion capture system
        self.position_sub = self.create_subscription(
            PoseStamped,
            '/vrpn_mocap/protect_drone/pose',
            self.drone_position_callback,
            SENSOR_QOS
        )

        # Publisher to send bodyrate commands to the vehicle
        self.body_rate_publisher = self.create_publisher(
            AttitudeTarget,
            '/mavros/setpoint_raw/attitude',
            SENSOR_QOS
        )

        # Publisher to send setpoint positions
        self.setpoint_pub = self.create_publisher(
            PoseStamped, 
            '/mavros/setpoint_position/local', 
            10
        )

        self.log_sp_pub = self.create_publisher(
            PoseStamped,
            'nmpc/setpoint',
            10
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

        self.current_setpoint = PoseStamped()
        self.current_setpoint.header.frame_id = 'map'
        self.current_setpoint.pose.position.x = 0.0
        self.current_setpoint.pose.position.y = 3.0
        self.current_setpoint.pose.position.z = 3.75
        self.current_setpoint.pose.orientation.w = 1.0
        self.current_setpoint.pose.orientation.x = 0.0
        self.current_setpoint.pose.orientation.y = 0.0
        self.current_setpoint.pose.orientation.z = 0.0

        self.drone_pose = Pose()

        # Get starttime for reference trajectory generation in seconds
        self.start_time = self.get_clock().now()

        self.SM_state = "initialize"
        self.publish_mode = "position_setpoint" # Options: "position_setpoint", "nmpc_control"

    def drone_position_callback(self, msg):
        self.drone_pose = msg.pose

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

        # Create the initial state vector
        x0 = self.state.reshape(16,1)

        # Get current time in seconds for reference trajectory generation
        t_now = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        if self.SM_state == "initialize":
            self.publish_mode = "position_setpoint"
            self.current_setpoint.pose.position.x = 0.0
            self.current_setpoint.pose.position.y = 3.0
            self.current_setpoint.pose.position.z = 3.75
            # Check if the drone has reached the first setpoint
            if np.linalg.norm(np.array([self.drone_pose.position.x, self.drone_pose.position.y, self.drone_pose.position.z]) - np.array([self.current_setpoint.pose.position.x, self.current_setpoint.pose.position.y, self.current_setpoint.pose.position.z])) < 0.1:
                self.SM_state = "preparing_maneuver"
                self.start_time = self.get_clock().now()
        elif self.SM_state == "preparing_maneuver":
            if t_now >= 10.0:
                self.publish_mode = "position_setpoint"
                self.current_setpoint.pose.position.x = 0.0
                self.current_setpoint.pose.position.y = -2.0
                self.current_setpoint.pose.position.z = 3.75
                self.SM_state = "maneuvering"
        elif self.SM_state == "maneuvering":
            # Check if the drone has passed y = 0
            if self.drone_pose.position.y < 0.0:
                self.publish_mode = "nmpc_control"
                self.current_setpoint.pose.position.x = 0.0
                self.current_setpoint.pose.position.y = 0.0
                self.current_setpoint.pose.position.z = 0.75
                self.nmpc.set_setpoint(np.array([self.current_setpoint.pose.position.x, self.current_setpoint.pose.position.y, self.current_setpoint.pose.position.z]))
                self.start_time = self.get_clock().now()
                self.SM_state = "finished"
        elif self.SM_state == "finished":
            self.publish_mode = "nmpc_control"
            if t_now >= 10.0:
                self.SM_state = "initialize"

        # Try to solve the NMPC problem
        try:
            u_pred, x_pred = self.nmpc.solve(x0)
        except Exception as e:
            self.get_logger().error(f"NMPC solver failed: {e}")
            return

        # Get the first control input from the predicted control inputs
        u = u_pred[0, :]

        # Scale thrust to the range [0, 1]         
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

        self.current_setpoint.header.stamp = self.get_clock().now().to_msg()

        self.log_sp_pub.publish(self.current_setpoint)

        # Publish the setpoints
        if self.publish_mode == "position_setpoint":
            self.setpoint_pub.publish(self.current_setpoint)
        elif self.publish_mode == "nmpc_control":
            self.body_rate_publisher.publish(bodyrate_command)
        
        self.get_logger().info(f"Current state: {self.SM_state}, Setpoint: [{self.current_setpoint.pose.position.x}, {self.current_setpoint.pose.position.y}, {self.current_setpoint.pose.position.z}]")

def main(args=None):    
    rclpy.init(args=args)

    node = e_stop_compensated_node()

    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
