################################
# Generic imports
################################
import numpy as np
import scipy as sp
import scipy.io as sio
from timeit import default_timer as timer
import quaternion
import csv

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
from aruco_interfaces.msg import SetPoint
from tf_transformations import euler_from_quaternion, quaternion_from_euler

################################
# MavROS
################################
from mavros.base import SENSOR_QOS
from mavros_msgs.srv import CommandBool
from mavros_msgs.msg import State, AttitudeTarget

################################
# Custom imports
################################
from nmpc_interfaces.msg import State as NMPCState



class LoggerNode(Node):
    def __init__(self):
        super().__init__('logger_node')

        ########################################
        # Setup of ROS2 subscribers
        ########################################
        # Subscribe to the state estimate topic
        self.state_sub = self.create_subscription(
            NMPCState,
            'nmpc/estimated_state',
            self.state_callback,
            SENSOR_QOS
        )

        # Subscribe to the drone pose from motion capture
        self.drone_pose_sub = self.create_subscription(
            PoseStamped,
            '/vrpn_mocap/protect_drone/pose',
            self.drone_pose_callback,
            SENSOR_QOS
        )


        # Subscribe to body rate and thrust commands
        self.setpoint_sub = self.create_subscription(
            AttitudeTarget,
            '/mavros/setpoint_raw/attitude',
            self.setpoint_callback,
            SENSOR_QOS
        )

        # Subscribe to predicted drone path
        self.predicted_drone_trajectory_sub = self.create_subscription(
            Path,
            'nmpc/predicted_drone_trajectory',
            self.predicted_drone_trajectory_callback,
            SENSOR_QOS
        )

        # Subscribe to predicted bodyrates
        self.predicted_bodyrate_sub = self.create_subscription(
            Path,
            'nmpc/predicted_bodyrates',
            self.predicted_bodyrate_callback,
            SENSOR_QOS
        )

        # Subscribe to predicted load path
        self.predicted_load_trajectory_sub = self.create_subscription(
            Path,
            'nmpc/predicted_trajectory',
            self.predicted_load_trajectory_callback,
            SENSOR_QOS
        )

        # Subscribe to mavros state
        self.mavros_state_sub = self.create_subscription(
            State,
            'mavros/state',
            self.mavros_state_callback,
            SENSOR_QOS
        )



        ########################################
        # Create CSV files
        ########################################
        # CSV file for state logging
        # x = {xL, vL, q, w, qe}
        f_states = open('/home/drone/ros2_ws/src/nmpc/data/state_log.csv', mode='w')
        # Create header
        writer = csv.writer(f_states)
        writer.writerow(['time', 
                         'xD_gt', 'yD_gt', 'zD_gt', 
                         'xL_gt', 'yL_gt', 'zL_gt', 
                         'vLx_est', 'vLy_est', 'vLz_est',
                         'qx_gt', 'qy_gt', 'qz_gt',
                         'wLx_est', 'wLy_est', 'wLz_est',
                         'qew_gt', 'qex_gt', 'qey_gt', 'qez_gt',
                         'u_F', 'u_Wx', 'u_Wy', 'u_Wz',
                         'mode'
                         ])
        
        self.xD = 0.0
        self.yD = 0.0
        self.zD = 0.0
        self.xL = 0.0
        self.yL = 0.0
        self.zL = 0.0
        self.vLx = 0.0
        self.vLy = 0.0
        self.vLz = 0.0
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0
        self.wLx = 0.0
        self.wLy = 0.0
        self.wLz = 0.0
        self.qex = 0.0
        self.qey = 0.0
        self.qez = 0.0
        self.qew = 0.0
        self.mode = None

        # CSV file for load predicted path
        f_predicted_load = open('/home/drone/ros2_ws/src/nmpc/data/predicted_load_trajectory_log.csv', mode='w')
        # Create header (One coloumn for time and then 90 coloumns for x,y,z for 30 time steps)
        header_load = ['time']
        for i in range(30):
            header_load.append(f'xL_pred_{i}')
        for i in range(30):
            header_load.append(f'yL_pred_{i}')
        for i in range(30):
            header_load.append(f'zL_pred_{i}')
        writer_load = csv.writer(f_predicted_load)
        writer_load.writerow(header_load)

        # CSV file for drone predicted path
        f_predicted_drone = open('/home/drone/ros2_ws/src/nmpc/data/predicted_drone_trajectory_log.csv', mode='w')
        # Create header (One coloumn for time and then 90 coloumns for
        header_drone = ['time']
        for i in range(30):
            header_drone.append(f'xD_pred_{i}')
        for i in range(30):
            header_drone.append(f'yD_pred_{i}')
        for i in range(30):
            header_drone.append(f'zD_pred_{i}')
        writer_drone = csv.writer(f_predicted_drone)
        writer_drone.writerow(header_drone)

        # CSV file for load bodyrate predicted path
        f_predicted_bodyrate = open('/home/drone/ros2_ws/src/nmpc/data/predicted_bodyrate_log.csv', mode='w')
        # Create header (One coloumn for time and then 90 coloumns for
        header_bodyrate = ['time']
        for i in range(30):
            header_bodyrate.append(f'wx_pred_{i}')
        for i in range(30):
            header_bodyrate.append(f'wy_pred_{i}')
        for i in range(30):
            header_bodyrate.append(f'wz_pred_{i}')
        writer_bodyrate = csv.writer(f_predicted_bodyrate)
        writer_bodyrate.writerow(header_bodyrate)

        self.load_predicted_trajectory = None
        self.drone_predicted_trajectory = None
        self.bodyrate_predicted_trajectory = None

        # Flags to see if new data has arrived
        self.new_state_data = False
        self.new_drone_pose_data = False
        self.new_load_predicted_data = False
        self.new_drone_predicted_data = False
        self.new_bodyrate_predicted_data = False

    def state_callback(self, msg):
        self.xL = msg.load_position[0]
        self.yL = msg.load_position[1]
        self.zL = msg.load_position[2]
        
        self.vLx = msg.load_velocity[0]
        self.vLy = msg.load_velocity[1]
        self.vLz = msg.load_velocity[2]
        
        self.qx = msg.cable_vector[0]
        self.qy = msg.cable_vector[1]
        self.qz = msg.cable_vector[2]
        q = np.array([self.qx, self.qy, self.qz])
        q = q / np.linalg.norm(q)
        self.qx = q[0]
        self.qy = q[1]
        self.qz = q[2]

        self.wLx = msg.load_angular_velocity[0]
        self.wLy = msg.load_angular_velocity[1]
        self.wLz = msg.load_angular_velocity[2]

        self.qew = msg.drone_attitude[0]
        self.qex = msg.drone_attitude[1]
        self.qey = msg.drone_attitude[2]
        self.qez = msg.drone_attitude[3]

        self.new_state_data = True
    
    def drone_pose_callback(self, msg):
        self.xD = msg.pose.position.x
        self.yD = msg.pose.position.y
        self.zD = msg.pose.position.z

        self.new_drone_pose_data = True

    def predicted_load_trajectory_callback(self, msg):
        self.load_predicted_trajectory = msg
        self.new_load_predicted_data = True
    
    def predicted_drone_trajectory_callback(self, msg):
        self.drone_predicted_trajectory = msg
        self.new_drone_predicted_data = True

    def predicted_bodyrate_callback(self, msg):
        self.bodyrate_predicted_trajectory = msg
        self.new_bodyrate_predicted_data = True

    def mavros_state_callback(self, msg):
        if msg.mode == 'OFFBOARD':
            self.mode = 1
        else:
            self.mode = 0

    def setpoint_callback(self, msg):
        self.u_F = msg.thrust
        self.u_Wx = msg.body_rate.x
        self.u_Wy = msg.body_rate.y
        self.u_Wz = msg.body_rate.z

        # Check if all new data has arrived
        if self.new_state_data and self.new_drone_pose_data and self.new_load_predicted_data and self.new_drone_predicted_data and self.new_bodyrate_predicted_data:
            # Log state data
            with open('/home/drone/ros2_ws/src/nmpc/data/state_log.csv', mode='a') as f_states:
                writer = csv.writer(f_states)
                current_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9
                writer.writerow([current_time,
                                 self.xD, self.yD, self.zD,
                                 self.xL, self.yL, self.zL,
                                 self.vLx, self.vLy, self.vLz,
                                 self.qx, self.qy, self.qz,
                                 self.wLx, self.wLy, self.wLz,
                                 self.qew, self.qex, self.qey, self.qez,
                                 self.u_F, self.u_Wx, self.u_Wy, self.u_Wz,
                                 self.mode
                                 ])
            
            # Log predicted load trajectory
            with open('/home/drone/ros2_ws/src/nmpc/data/predicted_load_trajectory_log.csv', mode='a') as f_predicted_load:
                writer_load = csv.writer(f_predicted_load)
                row_load = [current_time]
                for pose in self.load_predicted_trajectory.poses:
                    row_load.append(pose.pose.position.x)
                for pose in self.load_predicted_trajectory.poses:
                    row_load.append(pose.pose.position.y)
                for pose in self.load_predicted_trajectory.poses:
                    row_load.append(pose.pose.position.z)
                writer_load.writerow(row_load)
            
            # Log predicted drone trajectory
            with open('/home/drone/ros2_ws/src/nmpc/data/predicted_drone_trajectory_log.csv', mode='a') as f_predicted_drone:
                writer_drone = csv.writer(f_predicted_drone)
                row_drone = [current_time]
                for pose in self.drone_predicted_trajectory.poses:
                    row_drone.append(pose.pose.position.x)
                for pose in self.drone_predicted_trajectory.poses:
                    row_drone.append(pose.pose.position.y)
                for pose in self.drone_predicted_trajectory.poses:
                    row_drone.append(pose.pose.position.z)
                writer_drone.writerow(row_drone)
            
            # Log predicted bodyrate trajectory
            with open('/home/drone/ros2_ws/src/nmpc/data/predicted_bodyrate_log.csv', mode='a') as f_predicted_bodyrate:
                writer_bodyrate = csv.writer(f_predicted_bodyrate)
                row_bodyrate = [current_time]
                for pose in self.bodyrate_predicted_trajectory.poses:
                    row_bodyrate.append(pose.pose.position.x)
                for pose in self.bodyrate_predicted_trajectory.poses:
                    row_bodyrate.append(pose.pose.position.y)
                for pose in self.bodyrate_predicted_trajectory.poses:
                    row_bodyrate.append(pose.pose.position.z)
                writer_bodyrate.writerow(row_bodyrate)

            # Reset new data flags
            self.new_state_data = False
            self.new_drone_pose_data = False
            self.new_load_predicted_data = False
            self.new_drone_predicted_data = False
            self.new_bodyrate_predicted_data = False

def main(args=None):
    rclpy.init(args=args)

    logger_node = LoggerNode()

    rclpy.spin(logger_node)

    logger_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
