import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, Pose
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from load_estimation.marker_detection.PoseEstimator import PoseEstimator
from load_estimation.marker_detection.MarkerTracker import MarkerTracker

import cv2
import time
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
#import cProfile # For profiling



class LoadEstimationNode(Node):
    def __init__(self):
        super().__init__('load_estimation_node')
        self.get_logger().info("Load Estimation Node has been started.")

        # ========== Publishers ==========
        self.publish_camera_load_angle_ = self.create_publisher(Vector3, '/gimbal/camera_load_angle', 2)
        self.publish_load_pose_ = self.create_publisher(Pose, '/payload/pose', 2)

        #self.publish_image_gray_ = self.create_publisher(Image, '/debug/image_gray', 2)
        self.publish_image_thres_ = self.create_publisher(Image, '/debug/image_thres', 2)
        self.publish_image_payload_ = self.create_publisher(Image, '/debug/image_payload', 2)
        

        # ========== Subscribers ==========
        self.subscribe_image_ = self.create_subscription(Image, '/image_raw', self.image_callback, 2)

        # ========== timers ==========
        #timer_period = 1/30  # 30 Hz

        # Camera intrinsics and distortion
        #intrinsics = np.array([[835.4362078622368, 0, 323.0605420101571],
        #                        [0, 835.9483791851382, 232.14120929722597],
        #                        [0, 0, 1]], dtype=float)
        #dist_coeffs = np.array([-0.0999921394506428, 2.185188066835036, -0.005726667745540125, 0.00027787706601120816, -7.636164458366145], dtype=float)

        intrinsics = np.array([[410.5372217, 0, 165.56905082],
                               [0, 410.26619602, 119.30780434],
                               [0, 0, 1]], dtype=float)
        dist_coeffs = np.array([-2.92583245e-02, 1.00683957e+00, -2.29972697e-03, 9.16865223e-04,-3.11679202e+00], dtype=float)
        

        #marker_ids = [17, 27, 39, 119]
        #marker_placements = {marker_ids[0]: (-0.16, -0.16, 0.0),
        #                    marker_ids[1]: (0.16, -0.16, 0.0),
        #                    marker_ids[2]: (0.16, 0.16, 0.0),
        #                    marker_ids[3]: (-0.16, 0.16, 0.0)}
        
        marker_ids = [17, 27, 39, 51, 95, 119]
        self.marker_ids = marker_ids
        #marker_placements = {marker_ids[0]: (0.2545, -0.2614, -0.0049),
        #                     marker_ids[1]: (0.3354, 0.0048, 0.0469),
        #                     marker_ids[2]: (0.2648, 0.2665, -0.0079),
        #                     marker_ids[3]: (-0.2653, 0.279, -0.0075),
        #                     marker_ids[4]: (-0.3502, 0.0156, 0.0472),
        #                     marker_ids[5]: (-0.2772, -0.2497, -0.0037)}
        

        # ========== These for the current payload design with 6 markers ==========
        #marker_placements = {marker_ids[0]: (-0.264, -0.264, 0.0),
        #                     marker_ids[1]: (0.0, -0.342, -0.05),
        #                     marker_ids[2]: (0.264, -0.264, 0.0),
        #                     marker_ids[3]: (0.264, 0.264, 0.0),
        #                     marker_ids[4]: (0.0, 0.342, -0.05),
        #                     marker_ids[5]: (-0.264, 0.264, 0.0)}
        
        marker_placements = {marker_ids[0]: (0.264, -0.264, 0.0),
                             marker_ids[1]: (0.0, -0.342, 0.05),
                             marker_ids[2]: (-0.264, -0.264, 0.0),
                             marker_ids[3]: (-0.264, 0.264, 0.0),
                             marker_ids[4]: (0.0, 0.342, 0.05),
                             marker_ids[5]: (0.264, 0.264, 0.0)}

        # ========== Tuning parameters ==========
        self.downscale_factor = 1

        # Marker parameters
        self.n_folds = 5            # Marker symmetry order
        self.marker_pixels = 28     # Diameter of the marker in pixels at the expected distance (3.1m)
        self.nfold_percent = 0.50    # How much of the marker radius is used for n-fold symmetry detection (0-1)
        self.scale_factor = 1
        self.threshold_marker = 0.25 # Relative threshold for marker detection (0-1), relative to the max response in the image

        # Pose estimation parameters
        self.alpha = 0.25
        self.max_reproj_error = 10.0


        self.bridge = CvBridge()
        self.MT = MarkerTracker(self.n_folds, self.marker_pixels, self.nfold_percent, self.scale_factor, self.threshold_marker, marker_ids, self.downscale_factor)
        self.LP = PoseEstimator(intrinsics, dist_coeffs, marker_ids, marker_placements, alpha=self.alpha, max_reproj_error=self.max_reproj_error, downscale_factor=self.downscale_factor)


        # Testing detection rate
        self.prev_positions = {marker_id: None for marker_id in marker_ids}
        self.first_frame = True

        

    def image_callback(self, msg: Image):
        start_time = time.time()
        # Convert ROS Image message to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Publish the grayscale image for debugging
        #self.publish_image_gray_.publish(self.bridge.cv2_to_imgmsg(gray, encoding='mono8'))

        marker_positions, debug_image = self.MT.locate_marker(gray)
        self.publish_image_thres_.publish(self.bridge.cv2_to_imgmsg(debug_image.astype(np.uint8), encoding='mono8'))
        
        count_correct = {id: 0 for id in self.marker_ids}
        count_marker = {id: 0 for id in self.marker_ids}
        count_mislabeled = {id: 0 for id in self.marker_ids}
        
        

        if marker_positions is not None:
            #self.get_logger().info(f"Found {len(marker_positions)} markers.")
            # count markers based on id
            for pose in marker_positions:
                count_marker[pose.id] += 1
            if self.first_frame:
                for pose in marker_positions:
                    self.prev_positions[pose.id] = (pose.x, pose.y)
                self.first_frame = False
            for pose in marker_positions:

                if pose.x <self.prev_positions[pose.id][0] + 80 and pose.x >self.prev_positions[pose.id][0] - 80 and pose.y <self.prev_positions[pose.id][1] + 20 and pose.y >self.prev_positions[pose.id][1] - 20:
                    count_correct[pose.id] += 1
                else:
                    count_mislabeled[pose.id] += 1
                    self.get_logger().warning(f"position jump detected for marker ID {pose.id}. Previous position: {self.prev_positions[pose.id]}, current position: ({pose.x}, {pose.y})")

                self.prev_positions[pose.id] = (pose.x, pose.y)


            if len(marker_positions) >= 3:
                pose = self.LP.estimate_load_pose(marker_positions)

                if pose is not None:
                    rvec = pose[0]
                    tvec = pose[1]
                    #print(f"Estimated pose: rvec: {rvec.flatten()}, tvec: {tvec.flatten()}")

                    # calc roll and pitch from tvec
                    x, y, z = tvec[0], tvec[1], tvec[2] # in camera frame to payload frame
                    pitch = math.atan2(y, z) * (180 / math.pi) # pitch in gimbal, roll in camera frame
                    yaw = -math.atan2(x, z) * (180 / math.pi) # yaw in gimbal, -pitch in camera frame

                    # Publish the setpoint
                    error = Vector3(x=yaw, y=pitch, z=0.0)
                    self.publish_camera_load_angle_.publish(error)
                    self.get_logger().info(f"Published load camera angle: yaw: {yaw:.2f}, pitch: {pitch:.2f}")

                    pose_msg = Pose()
                    pose_msg.position.x = tvec[0][0]
                    pose_msg.position.y = tvec[1][0]
                    pose_msg.position.z = tvec[2][0]
                    R_mat, _ = cv2.Rodrigues(rvec)
                    R_quat = R.from_matrix(R_mat).as_quat()  # [x, y, z, w]
                    pose_msg.orientation.x = R_quat[0]
                    pose_msg.orientation.y = R_quat[1]
                    pose_msg.orientation.z = R_quat[2]
                    pose_msg.orientation.w = R_quat[3]
                    self.publish_load_pose_.publish(pose_msg)

                    #self.get_logger().info(f"Published load pose: x: {pose_msg.position.x:.2f}, y: {pose_msg.position.y:.2f}, z: {pose_msg.position.z:.2f}")

                    display_frame = cv_image.copy()
                    for pose in marker_positions:
                        x = int(pose.x * self.downscale_factor)
                        y = int(pose.y * self.downscale_factor)
                        self.get_logger().info(f"Marker ID {pose.id} at pixel coordinates: ({x}, {y})")
                        cv2.circle(display_frame, (x, y), 14, (0, 255, 0), 1)
                        cv2.putText(display_frame, f"{pose.id}", (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    self.LP.display_pose(display_frame)
                    self.publish_image_payload_.publish(self.bridge.cv2_to_imgmsg(display_frame, encoding='bgr8'))
                    if count_mislabeled[self.marker_ids[0]] > 0:
                        self.get_logger().warning(f"Mislabeled marker detected: ID {self.marker_ids[0]}. Check the detection and labeling logic.")
                        self.get_logger().warning(f"mislabeled locations: {[(pose.x, pose.y) for pose in marker_positions if pose.id == self.marker_ids[0]]}")
                        self.get_logger().warning(f"previous position: {self.prev_positions[self.marker_ids[0]]}")
                        #assert False, "Mislabeled marker detected. Check the detection and labeling logic."
                    with open('detection_rate.csv', 'a') as f:
                        f.write(f"{1},{count_correct[self.marker_ids[0]]},{count_correct[self.marker_ids[1]]},{count_correct[self.marker_ids[2]]},{count_correct[self.marker_ids[3]]},{count_correct[self.marker_ids[4]]},{count_correct[self.marker_ids[5]]},{count_marker[self.marker_ids[0]]},{count_marker[self.marker_ids[1]]},{count_marker[self.marker_ids[2]]},{count_marker[self.marker_ids[3]]},{count_marker[self.marker_ids[4]]},{count_marker[self.marker_ids[5]]},{count_mislabeled[self.marker_ids[0]]},{count_mislabeled[self.marker_ids[1]]},{count_mislabeled[self.marker_ids[2]]},{count_mislabeled[self.marker_ids[3]]},{count_mislabeled[self.marker_ids[4]]},{count_mislabeled[self.marker_ids[5]]}\n")
                else:
                    self.get_logger().info("No position estimate.")
                    error = Vector3(x=0.0, y=0.0, z=0.0)
                    #self.publish_camera_load_angle_.publish(error)
                    with open('detection_rate.csv', 'a') as f:
                        f.write(f"{0},{count_correct[self.marker_ids[0]]},{count_correct[self.marker_ids[1]]},{count_correct[self.marker_ids[2]]},{count_correct[self.marker_ids[3]]},{count_correct[self.marker_ids[4]]},{count_correct[self.marker_ids[5]]},{count_marker[self.marker_ids[0]]},{count_marker[self.marker_ids[1]]},{count_marker[self.marker_ids[2]]},{count_marker[self.marker_ids[3]]},{count_marker[self.marker_ids[4]]},{count_marker[self.marker_ids[5]]},{count_mislabeled[self.marker_ids[0]]},{count_mislabeled[self.marker_ids[1]]},{count_mislabeled[self.marker_ids[2]]},{count_mislabeled[self.marker_ids[3]]},{count_mislabeled[self.marker_ids[4]]},{count_mislabeled[self.marker_ids[5]]}\n")
            else:
                self.get_logger().info("Not enough markers detected.")
                error = Vector3(x=0.0, y=0.0, z=0.0)
                #self.publish_camera_load_angle_.publish(error)
                with open('detection_rate.csv', 'a') as f:
                    f.write(f"{0},{count_correct[self.marker_ids[0]]},{count_correct[self.marker_ids[1]]},{count_correct[self.marker_ids[2]]},{count_correct[self.marker_ids[3]]},{count_correct[self.marker_ids[4]]},{count_correct[self.marker_ids[5]]},{count_marker[self.marker_ids[0]]},{count_marker[self.marker_ids[1]]},{count_marker[self.marker_ids[2]]},{count_marker[self.marker_ids[3]]},{count_marker[self.marker_ids[4]]},{count_marker[self.marker_ids[5]]},{count_mislabeled[self.marker_ids[0]]},{count_mislabeled[self.marker_ids[1]]},{count_mislabeled[self.marker_ids[2]]},{count_mislabeled[self.marker_ids[3]]},{count_mislabeled[self.marker_ids[4]]},{count_mislabeled[self.marker_ids[5]]}\n")
        else:
            self.get_logger().info("No markers detected.")
            error = Vector3(x=0.0, y=0.0, z=0.0)
            #self.publish_camera_load_angle_.publish(error)
            with open('detection_rate.csv', 'a') as f:
                f.write(f"{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0}\n")
        end_time = time.time()
        processing_time = end_time - start_time
        self.get_logger().info(f"Processing time: {processing_time:.3f} seconds")

                
        #start_time = time.time()
        #end_time = time.time()
        #processing_time = end_time - start_time
        #self.get_logger().info(f"Processing time: {processing_time:.3f} seconds")

        #self.frames += 1
        #self.processing_time += processing_time
        #self.get_logger().info(f"average time: {self.processing_time/self.frames:.3f} seconds")





def main(args=None):
    rclpy.init(args=args)
    load_estimation = LoadEstimationNode()
    rclpy.spin(load_estimation)
    load_estimation.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
