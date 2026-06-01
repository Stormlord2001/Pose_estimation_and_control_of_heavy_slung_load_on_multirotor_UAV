import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from mavros.base import SENSOR_QOS
import numpy as np

class OptitrackEstimator(Node):
    def __init__(self):
        super().__init__('optitrack_estimator')
        
        self.drone_pose_sub = self.create_subscription(
            PoseStamped,
            '/vrpn_mocap/protect_drone/pose',
            self.drone_pose_callback,
            SENSOR_QOS
        )

        self.load_pose_sub = self.create_subscription(
            PoseStamped,
            '/vrpn_mocap/protect_load/pose',
            self.load_pose_callback,
            SENSOR_QOS
        )

        self.mavros_pose_pub = self.create_publisher(
            PoseStamped,
            '/mavros/vision_pose/pose',
            10
        )
        
        self.drone_pose_pub = self.create_publisher(
            PoseStamped,
            '/drone/pose',
            10
        )

        self.load_pose_pub = self.create_publisher(
            PoseStamped,
            '/load/pose',
            10
        )

        self.inertial_vel_pub = self.create_publisher(
            Vector3Stamped,
            '/load/velocity_inertial',
            10
        )

        self.cable_vector_pub = self.create_publisher(
            Vector3Stamped,
            '/load/cable_vector',
            10
        )

        self.angular_velocity_pub = self.create_publisher(
            Vector3Stamped,
            '/load/angular_velocity',
            10
        )


        self.drone_position = np.zeros(3)
        self.drone_attitude = np.zeros(4)
        self.load_position = np.zeros(3)
        self.load_attitude = np.zeros(4)
        self.load_velocity = np.zeros(3)

        self.prev_time = None
        self.prev_pose = None
        self.prev_Rib = np.eye(3)

        # Create a buffer for angular velocity smoothing of size 10
        self.angular_velocity_buffer = np.zeros((3, 10))



    def drone_pose_callback(self, msg):
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # Publish to MAVROS vision pose
        self.mavros_pose_pub.publish(msg)

        # Extract drone position and attitude
        current_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        
        current_orientation = np.array([
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z
        ])

        if self.prev_time is not None:
            dt = current_time - self.prev_time

            if dt <= 0:
                self.get_logger().warn('Non-positive time difference, skipping velocity calculation.')
                return
            
            # Calculate the velocity in the inertial frame
            vel_inertial = (current_position - self.prev_pose) / dt

            # Calculate rotation matrix from body-fixed rigidbody to inertial frame
            Lq = np.array(
                 [[-current_orientation[1],  current_orientation[0],  current_orientation[3], -current_orientation[2]],
                  [-current_orientation[2], -current_orientation[3],  current_orientation[0],  current_orientation[1]],
                  [-current_orientation[3],  current_orientation[2], -current_orientation[1],  current_orientation[0]]])
            
            Rq = np.array(
                 [[-current_orientation[1],  current_orientation[0], -current_orientation[3],  current_orientation[2]],
                  [-current_orientation[2],  current_orientation[3],  current_orientation[0], -current_orientation[1]],
                  [-current_orientation[3], -current_orientation[2],  current_orientation[1],  current_orientation[0]]])
            
            Rib = Rq @ Lq.T
            Rbi = Rib.T

            # Velocity in the body frame
            vel_body = Rbi @ vel_inertial

            # Calculate angular velocity based on lie algebra
            rotation_difference = -Rbi@self.prev_Rib/dt
            angular_velocity_raw = self.veemap(rotation_difference)
            self.prev_Rib = Rib
            
            # Update the angular velocity buffer
            self.angular_velocity_buffer = np.roll(self.angular_velocity_buffer, -1, axis=1)
            self.angular_velocity_buffer[:, -1] = angular_velocity_raw
            angular_velocity_smoothed = np.mean(self.angular_velocity_buffer, axis=1)

            angular_velocity_msg = Vector3Stamped()
            angular_velocity_msg.header = msg.header
            angular_velocity_msg.vector.x = angular_velocity_smoothed[0]
            angular_velocity_msg.vector.y = angular_velocity_smoothed[1]
            angular_velocity_msg.vector.z = angular_velocity_smoothed[2]
            # Publish the angular velocity message
            self.angular_velocity_pub.publish(angular_velocity_msg)

            # Create the messages
            inertial_pose_msg = PoseStamped()
            inertial_pose_msg.header = msg.header
            inertial_pose_msg.pose.position.x = current_position[0]
            inertial_pose_msg.pose.position.y = current_position[1]
            inertial_pose_msg.pose.position.z = current_position[2]
            inertial_pose_msg.pose.orientation.w = current_orientation[0]
            inertial_pose_msg.pose.orientation.x = current_orientation[1]
            inertial_pose_msg.pose.orientation.y = current_orientation[2]
            inertial_pose_msg.pose.orientation.z = current_orientation[3]

            inertial_vel_msg = Vector3Stamped()
            inertial_vel_msg.header = msg.header
            inertial_vel_msg.vector.x = vel_inertial[0]
            inertial_vel_msg.vector.y = vel_inertial[1]
            inertial_vel_msg.vector.z = vel_inertial[2]

            body_vel_msg = Vector3Stamped()
            body_vel_msg.header = msg.header
            body_vel_msg.vector.x = vel_body[0]
            body_vel_msg.vector.y = vel_body[1]
            body_vel_msg.vector.z = vel_body[2]

            # Publish the messages
            self.pose_pub.publish(inertial_pose_msg)
            self.inertial_vel_pub.publish(inertial_vel_msg)
            self.body_vel_pub.publish(body_vel_msg)
            
            self.get_logger().info(f'Inertial Velocity: {vel_inertial}, Body Velocity: {vel_body}')

        # Update the previous time and pose
        self.prev_time = current_time
        self.prev_pose = current_position

    def veemap(self, cross_matrix):
        vector = np.array([
            -cross_matrix[1,2],
            cross_matrix[0,2],
            -cross_matrix[0,1]
        ])
        return vector

def main(args=None):
    rclpy.init(args=args)
    node = OptitrackEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()