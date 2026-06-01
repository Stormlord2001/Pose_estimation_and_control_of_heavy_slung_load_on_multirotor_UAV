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

# ROS2 node that sends either setpoint (0, 1, 1) or (0, -1, 1) to the drone. Setpoint changes every ten seconds.
class Uncompensated_control(Node):
    def __init__(self):
        super().__init__('slung_load_uncompensated_node')

        # Publisher for position setpoints
        self.setpoint_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.timer = self.create_timer(0.02, self.publish_setpoint)
        self.start_time = self.get_clock().now()

    def publish_setpoint(self):
        t_now = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        setpoint_msg = PoseStamped()
        setpoint_msg.header = Header()
        setpoint_msg.header.stamp = self.get_clock().now().to_msg()
        setpoint_msg.header.frame_id = 'map'

        # Change between two setpoints every 10 seconds to test the response of the controller
        if int(t_now) % 30 < 15:
            setpoint_msg.pose.position.x = -2.0
            setpoint_msg.pose.position.y = 2.0
            setpoint_msg.pose.position.z = 4.0
            setpoint_msg.pose.orientation.w = 1.0
            setpoint_msg.pose.orientation.x = 0.0
            setpoint_msg.pose.orientation.y = 0.0
            setpoint_msg.pose.orientation.z = 0.0
            self.get_logger().info("Setpoint: [-2.0, 2.0, 4.0]")
        else:
            setpoint_msg.pose.position.x = 2.0
            setpoint_msg.pose.position.y = -2.0
            setpoint_msg.pose.position.z = 4.0
            setpoint_msg.pose.orientation.w = 1.0
            setpoint_msg.pose.orientation.x = 0.0
            setpoint_msg.pose.orientation.y = 0.0
            setpoint_msg.pose.orientation.z = 0.0
            self.get_logger().info("Setpoint: [2.0, -2.0, 4.0]")

        self.setpoint_pub.publish(setpoint_msg)

        self.setpoint_pub.publish(setpoint_msg)


def main(args=None):
    rclpy.init(args=args)
    node = Uncompensated_control()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
