import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Vector3Stamped, Vector3
from std_msgs.msg import Float32MultiArray
from mavros.base import SENSOR_QOS
from mavros_msgs.msg import AttitudeTarget


# Constant parameters
# Physical constants
g = 9.82                    # gravity
e3 = np.array([0, 0, 1])    # unit vector in z direction

# System parameters
mass = 2.2941               # mass of the full system
mQ = 1.8304                 # mass of the quadrotor
L = 2.95                    # length of the cable
hover_throttle = 0.745      # throttle required to hover (empirically determined)


# Nonlinear equations for the load
def slung_load_dynamics(x,u):
    ########################################################
    #   x = [xL(3), vL(3), q(3), w(3)]
    #   u = (f, R)  thrust magnitude and rotation matrix
    ########################################################
    f = u[0]  # total thrust of the quadcopter
    qe = u[1:5]  # drone attitude quaternion

    R = quaternion_to_rotation_matrix(qe)

    xL = x[0:3]      # load position  
    vL = x[3:6]      # load velocity
    q = x[6:9]       # cable direction (unit vector from drone to load)
    wL = x[9:12]     # angular velocity of the load in inertial frame

    thrust_world = f * R @ e3  # thrust in world frame

    # System dynamics
    xL_dot = vL

    q_dot = np.cross(wL, q)

    vL_dot = 1/mass * (np.dot(q, thrust_world) - mQ * L * np.linalg.norm(q_dot)**2) * q - g * e3

    w_dot = (-1.0 / (mQ * L)) * np.cross(q, thrust_world)

    return np.hstack((xL_dot, vL_dot, q_dot, w_dot))

# Runge-Kutta 4th order integration for the nonlinear dynamics
def rk4_step(x, u, dt):
    k1 = slung_load_dynamics(x, u)
    k2 = slung_load_dynamics(x + 0.5 * dt * k1, u)
    k3 = slung_load_dynamics(x + 0.5 * dt * k2, u)
    k4 = slung_load_dynamics(x + dt * k3, u)

    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Normalize the cable direction to prevent drift
    x_next[6:9] = normalize(x_next[6:9])

    return x_next

# State transition function for the UKF
current_u = np.array([0.0, 1.0, 0.0, 0.0, 0.0])  # [f, q]
def fx(x, dt):
    global current_u
    return rk4_step(x, current_u, dt)

# Measurement function for the UKF
def hx(x):
    xL = x[0:3]                 # Load position
    q = normalize(x[6:9])       # Cable direction
    return np.hstack((xL, q))

def hx_yaw(x):
    wL = x[9:12]
    return np.array([wL[2]])  # only wz

################################################
#   Helper functions
################################################
def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return v
    return v / norm

def big_skew(v):
    return np.array([[0, -v[0], -v[1], -v[2]],
                     [v[0], 0, v[2], -v[1]],
                     [v[1], -v[2], 0, v[0]],
                     [v[2], v[1], -v[0], 0]])

def quaternion_to_rotation_matrix(q):
    qs = q[0]
    qv = q[1:4]
    I3 = np.eye(3)
    hat_qv = skew(qv)
    R = (qs**2 - np.dot(qv, qv)) * I3 + 2 * np.outer(qv, qv) + 2 * qs * hat_qv
    return R

def quaternion_rotation(q, w, dt):
    # Convert angular velocity to quaternion derivative
    w_quat = np.hstack(([0], w))
    q_dot = 0.5 * big_skew(w_quat) @ q

    # Integrate to get the new quaternion
    q_new = q + q_dot * dt

    # Normalize the quaternion
    return q_new / np.linalg.norm(q_new)

def yaw_rate_from_quat(q1, q2, dt):
    # Ensure shortest path
    if np.dot(q1, q2) < 0:
        q2 = -q2

    # Quaternion inverse
    q1_inv = np.array([q1[0], -q1[1], -q1[2], -q1[3]])

    # Quaternion multiplication q_rel = q2 * q1^{-1}
    w1,x1,y1,z1 = q2
    w2,x2,y2,z2 = q1_inv

    q_rel = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

    qw, qx, qy, qz = q_rel

    # Yaw increment
    delta_yaw = np.arctan2(
        2 * (qw * qz + qx * qy),
        1 - 2 * (qy*qy + qz*qz)
    )

    return delta_yaw / dt

def ensure_positive_definite(P):
    # Symmetrize
    P = 0.5 * (P + P.T)

    # Add small diagonal jitter
    eps = 1e-6
    P += np.eye(P.shape[0]) * eps

    return P


class UKFNodeROSBAG(Node):
    def __init__(self):
        super().__init__("ukf_node_ros_bag")

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

        self.load_angular_velocity_sub = self.create_subscription(
            Vector3,
            '/nmpc/load_angular_velocity',
            self.load_angular_velocity_callback,
            SENSOR_QOS
        )   
        
        self.payload_vector_sub = self.create_subscription(
            Vector3,
            '/payload/vector',
            self.payload_vector_callback,
            SENSOR_QOS
        )

        self.control_input_sub = self.create_subscription(
            AttitudeTarget,
            '/mavros/setpoint_raw/attitude',
            self.control_input_callback,
            SENSOR_QOS
        )

        self.mavros_pose_pub = self.create_publisher(
            PoseStamped,
            '/mavros/vision_pose/pose',
            10
        )

        # Variables for storing current states
        self.drone_pos = [0,0,0]
        self.load_pos = [0,0,0]
        self.load_vel = [0,0,0]
        self.payload_vector = [0,0,-1]
        self.payload_vector_fresh = False
        self.load_angular_velocity = [0,0,0]
        self.drone_quat = [1,0,0,0]
        self.u_F = 0
        self.u_W = [0,0,0]

        # Variables for storing previous states
        self.prev_load_pos = [0,0,0]
        self.load_pos_time = 0.0
        self.prev_load_pos_time = 0.0
        self.load_quat = [1,0,0,0]
        self.prev_load_quat = [1,0,0,0]

        # Timer for UKF update
        self.ukf_timer = self.create_timer(0.02, self.ukf_update)

        # Initialize UKF
        dt = 1/50
        dim_x = 12
        dim_z = 6

        self.points = MerweScaledSigmaPoints(n=dim_x, alpha=0.1, beta=2.0, kappa=0)
        self.ukf = UKF(dim_x=dim_x, dim_z=dim_z, dt=dt, fx=fx, hx=hx, points=self.points)

        # Initial conditions
        self.ukf.x = np.zeros(12)
        self.ukf.x[6:9] = np.array([0., 0., -1.])  # cable initially down
        self.ukf.P *= 0.5

        # Process noise
        self.ukf.Q = np.diag([
            1e-6, 1e-6, 1e-6,      # xL
            2e-1, 2e-1, 1e3,       # vL
            1e-2, 1e-2, 1e-2,      # q
            0.5, 0.5, 0.5       # w
        ])

        # Measurement noise
        self.ukf.R = np.diag([
            5e-3, 5e-3, 5e-3,      # position sensor
            1e-3, 1e-3, 1e-3       # cable direction sensor
        ])

        # Full state estimate publisher using float[]
        self.state_estimate_pub = self.create_publisher(
            Float32MultiArray,
            '/state_estimate',
            SENSOR_QOS
        )

        self.load_vel_pub = self.create_publisher(
            Vector3,
            '/payload/velocity',
            SENSOR_QOS
        )

        self.load_vector_ground_truth_pub = self.create_publisher(
            Vector3,
            '/payload/vector_ground_truth',
            SENSOR_QOS
        )

        self.state_error_pub = self.create_publisher(
            Float32MultiArray,
            '/state_error',
            SENSOR_QOS
        )
        

    def drone_pose_callback(self, msg):
        #self.get_logger().info(f"Received drone pose: {msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")
        self.drone_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.drone_quat = np.array([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])

    def load_pose_callback(self, msg):
        #self.get_logger().info(f"Received load pose: {msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")
        self.load_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.load_pos_time =  msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.load_quat = np.array([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])


    def load_angular_velocity_callback(self, msg):
        #self.get_logger().info(f"Received load angular velocity: {msg.x}, {msg.y}, {msg.z}")
        self.load_angular_velocity = np.array([msg.x, msg.y, msg.z])

    def payload_vector_callback(self, msg):
        #self.get_logger().info(f"Received payload vector: {msg.x}, {msg.y}, {msg.z}")
        self.payload_vector = np.array([msg.x, msg.y, msg.z])
        self.payload_vector_fresh = True

    def control_input_callback(self, msg):  
        #self.get_logger().info(f"Received control input: {msg.thrust}, {msg.body_rate.x}, {msg.body_rate.y}, {msg.body_rate.z}")
        self.u_F = msg.thrust
        self.u_W = np.array([msg.body_rate.x, msg.body_rate.y, msg.body_rate.z])

    def ukf_step(self, z, u, z_yaw=None, do_update=True):
        global current_u
        current_u = u

        self.ukf.P = ensure_positive_definite(self.ukf.P)

        self.ukf.predict()

        # Normalize q in all sigma points to prevent drift
        for i in range(self.ukf.sigmas_f.shape[0]):
            self.ukf.sigmas_f[i, 6:9] = normalize(self.ukf.sigmas_f[i, 6:9])

        if do_update:
            # ---- MAIN MEASUREMENT UPDATE ----
            self.ukf.update(z)

            # ---- YAW PSEUDO-MEASUREMENT UPDATE ----
            if z_yaw is not None:
                R_yaw = np.array([[10]])  # tune this!
                self.ukf.update(z_yaw, hx=hx_yaw, R=R_yaw)
        
        # Enforce unit vector constraint on q after update
        self.ukf.x[6:9] = normalize(self.ukf.x[6:9])

        return self.ukf.x.copy()

    def ukf_update(self):
        current_time = self.load_pos_time
        dt = current_time - self.prev_load_pos_time

        if dt <= 0:
            self.prev_load_pos = self.load_pos
            self.prev_load_pos_time = self.load_pos_time
            self.prev_load_quat = self.load_quat
            return  # Skip update if time hasn't advanced
        self.load_vel = np.subtract(self.load_pos, self.prev_load_pos) / dt

        payload_vector_gt = normalize(np.subtract(self.load_pos, self.drone_pos))

        payload_vector_ground_truth_msg = Vector3()
        payload_vector_ground_truth_msg.x = payload_vector_gt[0]
        payload_vector_ground_truth_msg.y = payload_vector_gt[1]
        payload_vector_ground_truth_msg.z = payload_vector_gt[2]
        self.load_vector_ground_truth_pub.publish(payload_vector_ground_truth_msg)

        # Publish load velocity for debugging
        vel_msg = Vector3()
        vel_msg.x = self.load_vel[0]
        vel_msg.y = self.load_vel[1]
        vel_msg.z = self.load_vel[2]
        self.load_vel_pub.publish(vel_msg)

        # Scale u_F to actual thrust values
        hover_thrust = mass * g
        max_thrust = hover_thrust / hover_throttle
        F = self.u_F * max_thrust

        # Create control input vector
        u = np.hstack((F, self.drone_quat))

        if self.payload_vector_fresh:
            # Consume the latest payload vector once. Subsequent 50 Hz ticks
            # run prediction-only until a new vector message arrives.
            self.payload_vector_fresh = False

            # Get the estimated payload position from drone pose and payload vector
            estimated_payload_pos = np.add(self.drone_pos, self.payload_vector)

            # Calculate yaw rate from drone quaternion change
            yaw_rate = yaw_rate_from_quat(self.prev_load_quat, self.load_quat, dt)
            z_yaw = np.array([yaw_rate])

            # Create measurement vector
            z = np.hstack((estimated_payload_pos, normalize(self.payload_vector)))

            # Run predict + update when fresh measurement data is available
            state_estimate = self.ukf_step(z, u, z_yaw, do_update=True)
        else:
            # Run prediction-only when no new payload/vector sample was received.
            state_estimate = self.ukf_step(None, u, do_update=False)

        # Set the yaw angular velocity = 0
        # state_estimate[11] = yaw_rate

        # Publish the 12-state UKF estimate as float[]
        state_msg = Float32MultiArray()
        state_msg.data = state_estimate.astype(np.float32).tolist()
        self.state_estimate_pub.publish(state_msg)

        self.get_logger().info(f"UKF state estimate: {state_estimate}")

        # State error calculation
        gt_payload_vector = normalize(np.subtract(self.load_pos, self.drone_pos))
        true_state = np.hstack((self.load_pos, self.load_vel, gt_payload_vector, self.load_angular_velocity))
        state_error = np.subtract(state_estimate, true_state)

        # Publish state error
        state_error_msg = Float32MultiArray()
        state_error_msg.data = state_error.astype(np.float32).tolist()
        self.state_error_pub.publish(state_error_msg)

        self.prev_load_pos = self.load_pos
        self.prev_load_pos_time = self.load_pos_time
        self.prev_load_quat = self.load_quat


def main(args=None):
    rclpy.init(args=args)
    node = UKFNodeROSBAG()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
