import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, PoseStamped, Pose

import numpy as np
from scipy.spatial.transform import Rotation as R
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy

class TransformationNode(Node):
    def __init__(self):
        super().__init__('transformation_node')
        self.get_logger().info("Transformation Node has been started.")

        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ========== Publishers ==========
        self.publish_drone_payload_vector_ = self.create_publisher(Vector3, '/payload/vector', 2)
        self.publish_payload_world_pose_ = self.create_publisher(PoseStamped, '/payload/world_pose', 2) 

        # ========== Subscribers ==========
        self.subscribe_payload_pose_ = self.create_subscription(Pose, '/payload/pose', self.payload_pose_callback, 2)
        self.subscribe_gimbal_angles_ = self.create_subscription(Vector3, '/gimbal/angles', self.gimbal_angles_callback, 2)
        self.subscribe_drone_pose_ = self.create_subscription(PoseStamped, '/vrpn_mocap/protect_drone/pose', self.drone_pose_callback, qos_profile=self.qos_profile)


        self.gimbal_angles = Vector3()
        self.payload_pose = Pose()
        self.drone_pose = PoseStamped()

        self.d_drone_to_gimbal = 0.107 # 10.7 cm from drone to gimbal
        self.d_gimbal_to_camera = 0.025 # 2.5 cm from gimbal to camera


    def payload_pose_callback(self, msg):
        self.payload_pose = msg
        #self.get_logger().info(f"Received payload pose: {msg.position.x}, {msg.position.y}, {msg.position.z}")

    def drone_pose_callback(self, msg):
        self.drone_pose = msg
        #self.get_logger().info(f"Received drone pose: {msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")

    def gimbal_angles_callback(self, msg):
        start_time = self.get_clock().now()
        self.gimbal_angles = msg
        #self.get_logger().info(f"Received gimbal angles: {msg.x}, {msg.y}, {msg.z}")


        # 1. World -> Drone
        p_D_W = np.array([self.drone_pose.pose.position.x, self.drone_pose.pose.position.y, self.drone_pose.pose.position.z])
        q_D_W = np.array([self.drone_pose.pose.orientation.x, self.drone_pose.pose.orientation.y, self.drone_pose.pose.orientation.z, self.drone_pose.pose.orientation.w])
        r_D_W = R.from_quat(q_D_W)
        
        # 2. Drone -> Gimbal base
        p_G0_D = np.array([0,0,-self.d_drone_to_gimbal])
        q_G0_D = R.from_euler('y', 0, degrees=True).as_quat() 

        # 3. World -> Gimbal base
        p_G0_W = p_D_W + r_D_W.apply(p_G0_D)
        q_G0_W = r_D_W * R.from_quat(q_G0_D)
        #self.get_logger().info(f"Gimbal base position in world: {p_G0_W}, Gimbal base orientation in world (quat): {q_G0_W.as_quat()}, angles (deg): {q_G0_W.as_euler('xyz', degrees=True)}")

        # 4. Gimbal base -> Gimbal (pure rotation)
        q_G_G0 = R.from_euler('XY', [self.gimbal_angles.x, self.gimbal_angles.y], degrees=True).as_quat()  # [x,y,z,w]

        # 5. World -> Gimbal
        p_G_W = p_G0_W
        q_G_W = q_G0_W * R.from_quat(q_G_G0)

        # 6. Gimbal -> Camera
        p_C_G = np.array([self.d_gimbal_to_camera,0,0])
        q_C_G = R.from_euler('xz', [-90, -90], degrees=True).as_quat() # Alternative way to get the same rotation

        # 7. World -> Camera
        p_C_W = p_G_W + q_G_W.apply(p_C_G)
        q_C_W = q_G_W * R.from_quat(q_C_G)
        angles = q_C_W.as_euler('xyz', degrees=True)
        #self.get_logger().info(f"Camera position in world: {p_C_W[0]:.3f}, {p_C_W[1]:.3f}, {p_C_W[2]:.3f}, angles (deg): {angles[0]:.2f}, {angles[1]:.2f}, {angles[2]:.2f}")

        # 8. Camera -> Payload
        p_P_C = np.array([self.payload_pose.position.x, self.payload_pose.position.y, self.payload_pose.position.z])
        q_P_C = np.array([self.payload_pose.orientation.x, self.payload_pose.orientation.y, self.payload_pose.orientation.z, self.payload_pose.orientation.w])

        # 9. World -> Payload
        p_P_W = p_C_W + q_C_W.apply(p_P_C)
        q_P_W = q_C_W * R.from_quat(q_P_C)
        angles = q_P_W.as_euler('xyz', degrees=True)
        #self.get_logger().info(f"Payload position in world: {p_P_W[0]:.3f}, {p_P_W[1]:.3f}, {p_P_W[2]:.3f}, angles (deg): {angles[0]:.2f}, {angles[1]:.2f}, {angles[2]:.2f}")

        payload_world_pose_msg = PoseStamped()
        payload_world_pose_msg.header.stamp = self.get_clock().now().to_msg()
        payload_world_pose_msg.header.frame_id = "world"
        payload_world_pose_msg.pose.position.x = p_P_W[0]
        payload_world_pose_msg.pose.position.y = p_P_W[1]
        payload_world_pose_msg.pose.position.z = p_P_W[2]
        payload_world_pose_msg.pose.orientation.x = q_P_W.as_quat()[0]
        payload_world_pose_msg.pose.orientation.y = q_P_W.as_quat()[1]
        payload_world_pose_msg.pose.orientation.z = q_P_W.as_quat()[2]
        payload_world_pose_msg.pose.orientation.w = q_P_W.as_quat()[3]

        self.publish_payload_world_pose_.publish(payload_world_pose_msg)

        drone_payload_vector = Vector3(x=p_P_W[0] - p_D_W[0], y=p_P_W[1] - p_D_W[1], z=p_P_W[2] - p_D_W[2])
        self.publish_drone_payload_vector_.publish(drone_payload_vector)
        end_time = self.get_clock().now()
        elapsed_time = (end_time - start_time).nanoseconds / 1e6  # Convert to milliseconds
        self.get_logger().info(f"Transformation took {elapsed_time:.2f}ms")

def main(args=None):
    rclpy.init(args=args)
    transformation_node = TransformationNode()
    rclpy.spin(transformation_node)
    transformation_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    
