import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Vector3Stamped, Vector3
from mavros.base import SENSOR_QOS
from nmpc_interfaces.msg import State
import numpy as np

class StateEstimator(Node):
    def __init__(self):
        super().__init__('state_estimator')
        
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

        self.estimated_state_pub = self.create_publisher(
            State,
            '/nmpc/estimated_state',
            10
        )

        self.drone_pose_pub = self.create_publisher(
            PoseStamped,
            '/nmpc/drone_pose',
            10
        )

        self.load_pose_pub = self.create_publisher(
            PoseStamped,
            '/nmpc/load_pose',
            10
        )

        self.load_angular_velocity_pub = self.create_publisher(
            Vector3,
            '/nmpc/load_angular_velocity',
            10
        )

        self.drone_current_time = 0.0
        self.load_current_time = 0.0

        self.drone_pose_time = 0.0
        self.load_pose_time = 0.0

        self.drone_position = np.zeros(3)
        self.drone_attitude = np.zeros(4)

        self.load_position = np.zeros(3)
        self.load_attitude = np.zeros(4)

        self.prev_time = None
        self.prev_load_pose = None
        self.prev_Rib = np.eye(3)
        self.qk = np.array([1.0, 0.0, 0.0, 0.0])

        # Create a buffer for angular velocity smoothing of size 4
        self.angular_velocity_buffer = np.zeros((3, 4))

        # Create timer with callback to sync state measurements
        self.timer = self.create_timer(1/50, self.estimate_state)



    def drone_pose_callback(self, msg):
        # Extract time from message header
        self.drone_pose_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # Publish to MAVROS vision pose
        self.mavros_pose_pub.publish(msg)

        # Copy pose and change frame
        drone_pose_msg = PoseStamped()
        drone_pose_msg.header = msg.header
        drone_pose_msg.header.frame_id = 'map'
        drone_pose_msg.pose = msg.pose
        self.drone_pose_pub.publish(drone_pose_msg)

        # Extract drone position and attitude
        self.drone_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        self.drone_attitude = np.array([
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z
        ])



    def load_pose_callback(self, msg):
        # Extract time from message header
        self.load_pose_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        load_pose_msg = PoseStamped()
        load_pose_msg.header = msg.header
        load_pose_msg.header.frame_id = 'map'
        load_pose_msg.pose = msg.pose
        self.load_pose_pub.publish(load_pose_msg)

        # Extract load position and attitude
        self.load_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        self.load_attitude = np.array([
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z
        ])

        #if np.dot(self.load_attitude, self.qk) < 0:
        #    self.load_attitude = -self.load_attitude
        #self.qk = self.load_attitude.copy()



    def estimate_state(self):
        # Ensure we have received both drone and load poses
        if self.drone_pose_time == 0.0 or self.load_pose_time == 0.0:
            return

        # Use the latest timestamp
        current_time = self.load_pose_time
        #current_time = max(self.drone_pose_time, self.load_pose_time)

        # Initialize previous load pose and time if not set
        if self.prev_load_pose is None or self.prev_time is None:
            self.prev_load_pose = self.load_position
            self.prev_time = current_time
            return
       
        # Compute time difference
        dt = current_time - self.prev_time

        # Check for valid time step
        if dt <= 0:
            return

        # Calculate rotation matrix from body-fixed rigidbody to inertial frame
        Lq = np.array(
                [[-self.load_attitude[1],  self.load_attitude[0],  self.load_attitude[3], -self.load_attitude[2]],
                [ -self.load_attitude[2], -self.load_attitude[3],  self.load_attitude[0],  self.load_attitude[1]],
                [ -self.load_attitude[3],  self.load_attitude[2], -self.load_attitude[1],  self.load_attitude[0]]])

        Rq = np.array(  
                [[-self.load_attitude[1],  self.load_attitude[0], -self.load_attitude[3],  self.load_attitude[2]],
                [ -self.load_attitude[2],  self.load_attitude[3],  self.load_attitude[0], -self.load_attitude[1]],
                [ -self.load_attitude[3], -self.load_attitude[2],  self.load_attitude[1],  self.load_attitude[0]]])

        # Create rotation matrices
        Rib = Rq @ Lq.T
        Rbi = Rib.T
    
        # Estimate load velocity in inertial frame
        load_velocity = (self.load_position - self.prev_load_pose) / dt

        # Calculate angular velocity in inertial frame based on lie algebra
        rotation_difference = -Rbi@self.prev_Rib/dt
        bodyrate_raw = self.veemap(rotation_difference)
        omega_raw = Rib @ bodyrate_raw

        '''R_delta = Rib @ self.prev_Rib.T

        # Project to SO(3) using SVD
        U, _, Vt = np.linalg.svd(R_delta)
        R_delta = U @ Vt

        # Ensure determinant = +1
        if np.linalg.det(R_delta) < 0:
            U[:, -1] *= -1
            R_delta = U @ Vt

        #theta = np.arccos((np.trace(R_delta) - 1) / 2.0)
        skew = R_delta - R_delta.T
        sin_theta = np.linalg.norm(self.veemap(skew)) / 2.0
        cos_theta = (np.trace(R_delta) - 1.0) / 2.0

        theta = np.arctan2(sin_theta, cos_theta)

        if theta < 1e-8:
            omega_raw = 1/(2*dt) * self.veemap(R_delta - R_delta.T)
        else:
            factor = theta / (2*np.sin(theta))
            omega_raw = factor/dt * self.veemap(R_delta - R_delta.T)'''

        self.prev_Rib = Rib
        self.prev_time = current_time
        self.prev_load_pose = self.load_position

        #print("Rotation angle per step: ", theta)
        #print("dt: ", dt)
        #print("trace(R_delta): ", np.trace(R_delta))

        # Update angular velocity buffer and compute rolling average
        self.angular_velocity_buffer = np.roll(self.angular_velocity_buffer, -1, axis=1)
        self.angular_velocity_buffer[:, -1] = omega_raw
        smoothed_omega = np.mean(self.angular_velocity_buffer, axis=1)

        self.load_angular_velocity_pub.publish(Vector3(x=smoothed_omega[0], y=smoothed_omega[1], z=smoothed_omega[2]))

        # Publish estimated states
        state_msg = State()
        state_msg.load_position = self.load_position.tolist()
        state_msg.load_velocity = load_velocity.tolist()
        state_msg.cable_vector = (self.load_position - self.drone_position).tolist()
        state_msg.drone_attitude = self.drone_attitude.tolist()
        state_msg.load_angular_velocity = smoothed_omega.tolist()
        state_msg.timestamp = current_time

        self.estimated_state_pub.publish(state_msg)



    def veemap(self, cross_matrix):
        vector = np.array([
            -cross_matrix[1,2],
            cross_matrix[0,2],
            -cross_matrix[0,1]
        ])
        return vector
    
    def quat_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])



def main(args=None):
    rclpy.init(args=args)
    node = StateEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
