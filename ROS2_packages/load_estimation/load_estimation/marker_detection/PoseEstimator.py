import cv2
import numpy as np

class PoseEstimator:
    def __init__(self, camera_matrix, dist_coeffs, marker_ids, marker_placements, alpha=0.5, max_reproj_error=5.0, downscale_factor=1.0):
        """
        alpha: smoothing factor [0-1] (higher = smoother, slower to respond)
        max_reproj_error: max acceptable mean reprojection error in pixels
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_ids = marker_ids
        self.marker_placements = marker_placements
        
        self.alpha = alpha
        self.max_reproj_error = max_reproj_error
        self.downscale_factor = downscale_factor

        self.prev_rvec = None
        self.prev_tvec = None
        self.rvec_visual = None
        self.tvec_visual = None

    def estimate_load_pose(self, locations):
        marker_positions = {}
        marker_seen = {i: False for i in self.marker_ids}

        for pose in locations:
            # update marker seen status
            if pose.id in self.marker_ids:
                marker_seen[pose.id] = True
                marker_positions.update({pose.id: pose})
     
        marker_detections = {id: (marker_positions[id].x*self.downscale_factor, marker_positions[id].y*self.downscale_factor) for id in self.marker_ids if marker_seen[id] is True}

        if len(marker_detections) >= 3:
            rvec, tvec, R, cam_pos, inliers = self.estimate_pose(
                self.marker_placements,
                marker_detections
            )

            return rvec, tvec, R, cam_pos
        else:
            return None

    def estimate_pose(self, object_points_by_id, detections):
        # --- 1. build correspondence lists ---
        obj_pts = []
        img_pts = []
        for id_, obj_pt in object_points_by_id.items():
            if id_ in detections:
                obj_pts.append(obj_pt)
                img_pts.append(detections[id_])

        obj_pts = np.asarray(obj_pts, dtype=float)
        img_pts = np.asarray(img_pts, dtype=float)
        n = len(obj_pts)

        if n < 3:
            raise ValueError("Need at least 3 detected markers for pose")

        # --- 2. initial pose estimate ---
        if n == 3:
            retval, rvecs, tvecs, reprojErr = cv2.solvePnPGeneric(
                obj_pts, img_pts,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_SQPNP
            )
            rvec, tvec = rvecs[0], tvecs[0]
            inliers = np.array([[0],[1],[2]])
        else:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts, img_pts,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_SQPNP,
                reprojectionError=8.0,
                iterationsCount=100,
                confidence=0.99
            )
            if not success:
                raise RuntimeError("PnPRansac failed")

        # --- 3. refine using LM ---
        refined_rvec, refined_tvec = cv2.solvePnPRefineLM(
            obj_pts[inliers[:,0]],
            img_pts[inliers[:,0]],
            self.camera_matrix,
            self.dist_coeffs,
            rvec,
            tvec
        )

        # --- 4. check reprojection error ---
        proj_pts, _ = cv2.projectPoints(
            obj_pts[inliers[:,0]],
            refined_rvec,
            refined_tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        reproj_err = np.mean(np.linalg.norm(proj_pts.reshape(-1,2) - img_pts[inliers[:,0]], axis=1))
        if reproj_err > self.max_reproj_error:
            print(f"Warning: high reprojection error ({reproj_err:.2f}px), using previous pose")
            if self.prev_rvec is not None:
                return self.prev_rvec, self.prev_tvec, cv2.Rodrigues(self.prev_rvec)[0], \
                       -cv2.Rodrigues(self.prev_rvec)[0].T @ self.prev_tvec, inliers
            #else:
            #    raise RuntimeError("No previous pose available to fallback")

        
        # --- 5. smooth pose with previous frame --- (not used as it causes lag in the position estimate)
        if self.prev_rvec is not None and self.alpha > 0:
            # Rotation smoothing via Rodrigues + SLERP approximation
            R_prev, _ = cv2.Rodrigues(self.prev_rvec)
            R_curr, _ = cv2.Rodrigues(refined_rvec)
            R_smooth = self.alpha * R_prev + (1 - self.alpha) * R_curr
            # Re-orthonormalize
            U, _, Vt = np.linalg.svd(R_smooth)
            R_smooth = U @ Vt
            refined_rvec, _ = cv2.Rodrigues(R_smooth)

            # Translation smoothing
            refined_tvec = self.alpha * self.prev_tvec + (1 - self.alpha) * refined_tvec

        # --- 6. save for next frame ---
        self.prev_rvec = refined_rvec
        self.prev_tvec = refined_tvec

        # Rotation matrix and camera position
        R_final, _ = cv2.Rodrigues(refined_rvec)
        cam_pos = -R_final.T @ refined_tvec

        self.rvec_visual = R_final
        self.tvec_visual = refined_tvec

        return refined_rvec, refined_tvec, R_final, cam_pos, inliers
    
    def display_pose(self, image, axis_length=0.1):
        """
        Draws the coordinate axes on the image for visualization.
        """
        if self.rvec_visual is not None and self.tvec_visual is not None:
            cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, self.rvec_visual, self.tvec_visual, axis_length, 5)

