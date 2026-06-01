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

class e_stop_uncompensated_node(Node):
    def __init__(self):
        super().__init__('e_stop_uncompensated_node')

        # Subscriber for the drones position from the motion capture system
        self.position_sub = self.create_subscription(
            PoseStamped,
            '/vrpn_mocap/protect_drone/pose',
            self.drone_position_callback,
            SENSOR_QOS
        )

        # Subscriber for the payloads position from the motion capture system
        self.payload_position_sub = self.create_subscription(
            PoseStamped,
            '/vrpn_mocap/protect_load/pose',
            self.payload_position_callback,
            SENSOR_QOS
        )

        # Publisher for position setpoints
        self.setpoint_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.timer = self.create_timer(0.02, self.control_loop)

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
        self.payload_pose = Pose()

        self.start_time = self.get_clock().now()

        self.state = "initialize"

    def drone_position_callback(self, msg):
        self.drone_pose = msg.pose

    def payload_position_callback(self, msg):
        self.payload_pose = msg.pose

    def control_loop(self):
        # State machine that has the drone fly to (0, 3, 4) and then after 10 seconds fly to (0, -2, 4). When the drone pases y = 0, the drone should stop immediately and not fly to the second setpoint to simulate an emergency stop.
        t_now = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        if self.state == "initialize":
            self.current_setpoint.pose.position.x = 0.0
            self.current_setpoint.pose.position.y = 3.0
            self.current_setpoint.pose.position.z = 3.75
            # Check if the drone has reached the first setpoint
            print("Drone:\n", self.drone_pose.position)
            print("Setpoint:\n", self.current_setpoint.pose.position)
            if np.linalg.norm(np.array([self.drone_pose.position.x, self.drone_pose.position.y, self.drone_pose.position.z]) - np.array([self.current_setpoint.pose.position.x, self.current_setpoint.pose.position.y, self.current_setpoint.pose.position.z])) < 0.1:
                self.state = "preparing_maneuver"
                self.start_time = self.get_clock().now()
        elif self.state == "preparing_maneuver":
            if t_now >= 10.0:
                self.current_setpoint.pose.position.x = 0.0
                self.current_setpoint.pose.position.y = -2.0
                self.current_setpoint.pose.position.z = 3.75
                self.state = "maneuvering"
        elif self.state == "maneuvering":
            # Check if the drone has passed y = 0
            if self.drone_pose.position.y < 0.0:
                self.current_setpoint.pose.position.x = 0.0
                self.current_setpoint.pose.position.y = 0.0
                self.current_setpoint.pose.position.z = 3.75
                self.state = "finished"
                self.start_time = self.get_clock().now()
        elif self.state == "finished":
            if t_now >= 10.0:
                self.state = "initialize"

        self.current_setpoint.header.stamp = self.get_clock().now().to_msg()
        self.setpoint_pub.publish(self.current_setpoint)
        self.get_logger().info(f"Current state: {self.state}, Setpoint: [{self.current_setpoint.pose.position.x}, {self.current_setpoint.pose.position.y}, {self.current_setpoint.pose.position.z}]")

def main(args=None):
    rclpy.init(args=args)
    node = e_stop_uncompensated_node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

