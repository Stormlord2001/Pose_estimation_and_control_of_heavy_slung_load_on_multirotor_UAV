from nmpc.controllers.bodyrate_nmpc import BodyrateMPC
from nmpc.models.bodyrate_model import BodyrateModel

__author__ = "Kristoffer Jensen"
__contact__ = "krisj21@student.sdu.dk"

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock

from mavros.base import SENSOR_QOS
from mavros_msgs.srv import CommandBool
from mavros_msgs.msg import AttitudeTarget, Thrust
from geometry_msgs.msg import PoseStamped, Vector3, Quaternion, TwistStamped, Twist, Vector3Stamped
from nav_msgs.msg import Odometry

from tf_transformations import euler_from_quaternion, quaternion_from_euler


def vector2PoseMsg(frame_id, position, attitude):
    pose_msg = PoseStamped()
    # msg.header.stamp = Clock().now().nanoseconds / 1000
    pose_msg.header.frame_id=frame_id
    pose_msg.pose.orientation.w = attitude[0]
    pose_msg.pose.orientation.x = attitude[1]
    pose_msg.pose.orientation.y = attitude[2]
    pose_msg.pose.orientation.z = attitude[3]
    pose_msg.pose.position.x = float(position[0])
    pose_msg.pose.position.y = float(position[1])
    pose_msg.pose.position.z = float(position[2])
    return pose_msg

class NMPC(Node):
    def __init__(self):
        super().__init__('nmpc_node')

        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/drone/pose',
            self.pose_callback,
            SENSOR_QOS
        )

        # Subscriber to get velocity of the vehicle
        self.vel_sub = self.create_subscription(
            Vector3Stamped,
            '/drone/velocity_local',
            self.vel_callback,
            SENSOR_QOS
        )

        '''self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            SENSOR_QOS
        )'''

        # Publisher to send bodyrate commands to the vehicle
        self.body_rate_publisher = self.create_publisher(
            AttitudeTarget,
            '/mavros/setpoint_raw/attitude',
            SENSOR_QOS
        )

        # Timer for the command loop
        timer_period = 0.02  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)

        # Create quadrotor bodyrate model
        self.model = BodyrateModel()

        # Create NMPC controller
        self.nmpc = BodyrateMPC(self.model)

        self.vehicle_attitude = np.array([1, 0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])

        self.setpoint_position = np.array([-0.5, 3.0, 2.5])

    def pose_callback(self, msg):
        # Save quaternion attitude to vehicle_attitude
        q = msg.pose.orientation
        self.vehicle_attitude = np.array([q.w, q.x, q.y, q.z])
        # TODO: Check that the received quaternion is in ENU, otherwise convert it to ENU [q.w, q.x, -q.y, -q.z]
        # I think it should be ENU though

        # Save position to vehicle_local_position
        p = msg.pose.position
        self.vehicle_local_position = np.array([p.x, p.y, p.z])
        # TODO: Check that the received position is in ENU, otherwise convert it to ENU [p.x, -p.y, -p.z]
        # I think it should be ENU though

    def vel_callback(self, msg):
        # Save velocity to vehicle_local_velocity
        v = msg.vector
        self.vehicle_local_velocity = np.array([v.x, v.y, v.z])
        # TODO: Check that the received velocity is in ENU, otherwise convert it to ENU [v.x, -v.y, -v.z]
        # I think it should be ENU though

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        self.vehicle_attitude = np.array([q.w, q.x, q.y, q.z])

        p = msg.pose.pose.position
        self.vehicle_local_position = np.array([p.x, p.y, p.z])

        v = msg.twist.twist.linear
        self.vehicle_local_velocity = np.array([v.x, v.y, v.z])

    def cmdloop_callback(self):
        # Get the current position error to use it as the initial state
        error_position = self.vehicle_local_position - self.setpoint_position

        # Create the initial state vector
        x0 = np.array([error_position[0], error_position[1], error_position[2],self.vehicle_local_velocity[0], self.vehicle_local_velocity[1], self.vehicle_local_velocity[2], self.vehicle_attitude[0], self.vehicle_attitude[1], self.vehicle_attitude[2], self.vehicle_attitude[3]]).reshape(10,1)

        # Solve the NMPC problem for the current state
        u_pred, x_pred = self.nmpc.solve(x0)
        
        # Get the first control input from the predicted control inputs
        u = u_pred[0, :]

        # Scale thrust to the range [0, 1] 
        # This is done here by multiplying the thrust by 0.07 and adding 0.0 to it
        #thrust_command = (u[0] * 0.03054989816 + 0.0)
        thrust_command = (u[0] * 0.01719183914 * 1.6129 + 0.0)

        # Create the bodyrate command message
        # To disable attitude control, we set the type_mask to IGNORE_ATTITUDE
        # https://docs.ros.org/en/noetic/api/mavros_msgs/html/msg/AttitudeTarget.html
        # The bodyrate is giving in body frame with NED convention
        bodyrate_command = AttitudeTarget()
        bodyrate_command.header.stamp = Clock().now().to_msg()
        bodyrate_command.type_mask = AttitudeTarget.IGNORE_ATTITUDE
        bodyrate_command.body_rate.x = float(u[1])
        bodyrate_command.body_rate.y = float(u[2])
        bodyrate_command.body_rate.z = float(u[3])
        bodyrate_command.thrust = float(thrust_command)

        # Publish the bodyrate command
        self.body_rate_publisher.publish(bodyrate_command)
        self.get_logger().info(f"Bodyrate command: {bodyrate_command}")
        #print("It works")

def main(args=None):    
    rclpy.init(args=args)

    nmpc_node = NMPC()

    rclpy.spin(nmpc_node)
    
    nmpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
