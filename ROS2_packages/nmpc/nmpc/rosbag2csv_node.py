################################
# Generic imports
################################
import numpy as np
import scipy as sp
import scipy.io as sio
from timeit import default_timer as timer
import csv
from datetime import datetime

################################
# Ros imports
################################
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from geometry_msgs.msg import PoseStamped, TwistStamped, Pose, Twist
from sensor_msgs.msg import Range, Imu
from nav_msgs.msg import Odometry, Path

################################
# MavROS
################################
from mavros.base import SENSOR_QOS
from mavros_msgs.srv import CommandBool
from mavros_msgs.msg import State, AttitudeTarget

class rosbag2csvNode(Node):
    def __init__(self):
        super().__init__('rosbag2csv_node')

        # Subscribe to the drone pose from motion capture
        self.drone_pose_sub = self.create_subscription(
            PoseStamped,
            '/vrpn_mocap/protect_drone/pose',
            self.drone_pose_callback,
            SENSOR_QOS
        )


        # Subscribe to the load pose from the motion capture
        self.load_pose_sub = self.create_subscription(
            PoseStamped,
            '/vrpn_mocap/protect_load/pose',
            self.load_pose_callback,
            SENSOR_QOS
        )

        # Subscribe to mavros state
        self.mavros_state_sub = self.create_subscription(
            State,
            'mavros/state',
            self.mavros_state_callback,
            SENSOR_QOS
        )

        # Subscribe to the setpoint topic to get the desired position of the drone
        self.setpoint_sub = self.create_subscription(
            PoseStamped,
            '/mavros/setpoint_position/local', #'/nmpc/setpoint_pose',
            self.setpoint_callback,
            SENSOR_QOS
        )

        # Subscribe to the nmpc setpoint topic to get the desired position of the load
        self.nmpc_setpoint_sub = self.create_subscription(
            PoseStamped,
            '/nmpc/setpoint_pose',
            self.nmpc_setpoint_callback,
            SENSOR_QOS
        )



        ########################################
        # Create CSV files
        ########################################
        # CSV file for state logging
        # x = {xL, vL, q, w, qe}
        # Make file name based on current date and time
        now = datetime.now()
        self.file_name = f'{now.strftime("%Y-%m-%d_%H-%M-%S")}.csv'
        f_states = open(f'/home/kristoffer/ros2_ws/src/nmpc/test_data/{self.file_name}', mode='w')
        # Create header
        writer = csv.writer(f_states)
        writer.writerow(['time', 
                         'xD_gt', 'yD_gt', 'zD_gt', 
                         'xL_gt', 'yL_gt', 'zL_gt', 
                         'xD_S', 'yD_S', 'zD_S',
                         'xL_S', 'yL_S', 'zL_S',
                         'qD_x', 'qD_y', 'qD_z', 'qD_w',
                         'qL_x', 'qL_y', 'qL_z', 'qL_w',
                         'mode'
                         ])
        
        self.xD = 0.0
        self.yD = 0.0
        self.zD = 0.0
        self.qxD = 0.0
        self.qyD = 0.0
        self.qzD = 0.0
        self.qwD = 0.0
        self.xL = 0.0
        self.yL = 0.0
        self.zL = 0.0
        self.qxL = 0.0
        self.qyL = 0.0
        self.qzL = 0.0
        self.qwL = 0.0
        self.setpoint_x = 0.0
        self.setpoint_y = 0.0
        self.setpoint_z = 0.0
        self.load_setpoint_x = 0.0
        self.load_setpoint_y = 0.0
        self.load_setpoint_z = 0.0
        self.mode = None

        self.start_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec*1e-9

        # Flags to see if new data has arrived
        self.new_drone_pose_data = False
        self.new_load_pose_data = False
        self.new_setpoint_data = False
        self.new_load_setpoint_data = False

        self.lines_written = 0


    
    def drone_pose_callback(self, msg):
        self.xD = msg.pose.position.x
        self.yD = msg.pose.position.y
        self.zD = msg.pose.position.z

        self.qxD = msg.pose.orientation.x
        self.qyD = msg.pose.orientation.y
        self.qzD = msg.pose.orientation.z
        self.qwD = msg.pose.orientation.w

        self.new_drone_pose_data = True

    def setpoint_callback(self, msg):
        self.setpoint_x = msg.pose.position.x
        self.setpoint_y = msg.pose.position.y
        self.setpoint_z = msg.pose.position.z

        self.new_setpoint_data = True

    def nmpc_setpoint_callback(self, msg):
        self.load_setpoint_x = msg.pose.position.x
        self.load_setpoint_y = msg.pose.position.y
        self.load_setpoint_z = msg.pose.position.z

        self.new_load_setpoint_data = True
    
    def load_pose_callback(self, msg):
        self.xL = msg.pose.position.x
        self.yL = msg.pose.position.y
        self.zL = msg.pose.position.z

        self.qxL = msg.pose.orientation.x
        self.qyL = msg.pose.orientation.y
        self.qzL = msg.pose.orientation.z
        self.qwL = msg.pose.orientation.w

        self.new_load_pose_data = True

        if self.new_drone_pose_data:
            self.new_drone_pose_data = False
            self.new_load_pose_data = False
            self.new_setpoint_data = False
            self.new_load_setpoint_data = False

            # Log data to CSV
            with open(f'/home/kristoffer/ros2_ws/src/nmpc/test_data/{self.file_name}', mode='a') as f_states:
                writer = csv.writer(f_states)
                writer.writerow([self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec*1e-9 - self.start_time,
                                 self.xD, self.yD, self.zD, 
                                 self.xL, self.yL, self.zL,
                                 self.setpoint_x, self.setpoint_y, self.setpoint_z,
                                 self.load_setpoint_x, self.load_setpoint_y, self.load_setpoint_z,
                                 self.qxD, self.qyD, self.qzD, self.qwD,
                                 self.qxL, self.qyL, self.qzL, self.qwL,
                                 self.mode
                                 ])
                
            self.lines_written += 1
            if self.lines_written % 100 == 0:
                self.get_logger().info(f'Written {self.lines_written} lines to CSV.')

    def mavros_state_callback(self, msg):
        if msg.mode == 'OFFBOARD':
            self.mode = 1
        else:
            self.mode = 0


def main(args=None):
    rclpy.init(args=args)

    node = rosbag2csvNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()