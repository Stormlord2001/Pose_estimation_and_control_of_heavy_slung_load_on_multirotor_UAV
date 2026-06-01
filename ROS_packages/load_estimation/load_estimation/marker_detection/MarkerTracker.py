import cv2
import numpy as np
import math

from load_estimation.marker_detection.MarkerPose import MarkerPose
from load_estimation.marker_detection.decode import decode_marker

class MarkerTracker:
    def __init__(self, order, marker_size, nfold_percent, scale_factor, threshold_marker, marker_ids, downscale_factor=1):
        # Calculate kernel size based on marker size and downscale factor
        kernel_size = int(marker_size * nfold_percent / downscale_factor)
        (kernel_real, kernel_imag) = self.generate_symmetry_detector_kernel(order, kernel_size)

        self.order = order
        self.mat_real = kernel_real / scale_factor
        self.mat_imag = kernel_imag / scale_factor

        self.threshold_marker = threshold_marker

        # showing the kernels
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(self.mat_real)
        #cv2.imshow("mat_real_norm", 255*self.mat_real/max_val)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(self.mat_imag)
        #cv2.imshow("mat_imag_norm", 255*self.mat_imag/max_val)

        self.y1 = int(math.floor(float(kernel_size)/2))
        self.y2 = int(math.ceil(float(kernel_size)/2))
        self.x1 = int(math.floor(float(kernel_size)/2))
        self.x2 = int(math.ceil(float(kernel_size)/2))

        # Using codering to id markers
        r_code_inner = int(math.floor((marker_size * nfold_percent / downscale_factor)/2)) #int(21/downscale_factor) 14
        self.r_code_outer = int(math.ceil((marker_size / downscale_factor)/2)) #int(32/downscale_factor) 20
        bits = 8
        transitions = 2
        self.decoder = decode_marker(r_code_inner, self.r_code_outer, bits, transitions, marker_ids)

    @staticmethod
    def generate_symmetry_detector_kernel(order, kernel_size):
        # type: (int, int) -> np.ndarray
        value_range = np.linspace(-1, 1, kernel_size)
        temp1 = np.meshgrid(value_range, value_range)
        kernel = temp1[0] + 1j * temp1[1]

        magnitude = abs(kernel)
        kernel = np.power(kernel, order)
        kernel = kernel * np.exp(-8 * magnitude ** 2)

        return np.real(kernel), np.imag(kernel)
    
    def locate_marker(self, frame):
        assert len(frame.shape) == 2, "Input image is not a single channel image."
        frame_real = frame.copy()
        frame_imag = frame.copy()

        # Convolve image with kernels.
        frame_real = cv2.filter2D(frame_real, cv2.CV_32F, self.mat_real)
        frame_imag = cv2.filter2D(frame_imag, cv2.CV_32F, self.mat_imag)
        frame_real_squared = cv2.multiply(frame_real, frame_real, dtype=cv2.CV_32F)
        frame_imag_squared = cv2.multiply(frame_imag, frame_imag, dtype=cv2.CV_32F)
        frame_sum_squared = cv2.add(frame_real_squared, frame_imag_squared, dtype=cv2.CV_32F)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(frame_sum_squared)

        threshold_value = max_val * self.threshold_marker 

        thres_img = np.where(frame_sum_squared > threshold_value, frame_sum_squared, 0)
        min_val, max_val_thresh, min_loc, max_loc_thresh = cv2.minMaxLoc(thres_img)
       
        # This is the golden plots
        ###cv2.imshow("frame_sum_squared_norm", 100*255*frame_sum_squared)
        ###cv2.imshow("thres_img_norm", 255*thres_img/max_val_thresh)

        contours, hierarchy = cv2.findContours(np.uint8(thres_img/max_val_thresh*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        poses = []

        # If too many contours are detected, skip the frame. as this takes too long to process, and is always due to noise.
        #print(f"Detected {len(contours)} contours.")
        if len(contours) > 10:
            #print(f"Too many contours detected ({len(contours)}), skipping frame.")
            return None, 255*thres_img/max_val_thresh

        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)

            # If contours are to close to the edge, skip them as markers can not be decoded there.
            if x - self.r_code_outer < 0 or x + self.r_code_outer + 1 >= frame.shape[1]:
                continue
            if y - self.r_code_outer < 0 or y + self.r_code_outer + 1 >= frame.shape[0]:
                continue
            
            frame_sum_cutout = self.extract_window_around_marker_location(frame_sum_squared, (int(x), int(y)))
            min_val_c, max_val_c, min_loc_c, max_loc_c = cv2.minMaxLoc(frame_sum_cutout)

            (dx, dy) = self.refine_marker_location(frame_sum_cutout, max_loc_c[1], max_loc_c[0])
            if abs(dx) > 1.5 or abs(dy) > 1.5:
                print(f"Refinement too large: dx: {dx}, dy: {dy}, skipping marker.")
                continue
            refined_location = (x-self.x1+max_loc_c[0] + dx, y-self.y1+max_loc_c[1] + dy)

            # If refined location is out of bounds, skip the marker.
            if refined_location[0] - self.r_code_outer < 0 or refined_location[0] + self.r_code_outer + 1 >= frame.shape[1]:
                print("Refined location out of bounds in x-direction, skipping marker.")
                continue
            if refined_location[1] - self.r_code_outer < 0 or refined_location[1] + self.r_code_outer + 1 >= frame.shape[0]:
                print("Refined location out of bounds in y-direction, skipping marker.")
                continue

            marker_id = self.decoder.extract_and_decode(frame, (int(refined_location[0]), int(refined_location[1])))
            #print(f"Marker ID: {marker_id}")
            if marker_id is None:
                continue

            pose = MarkerPose(refined_location[0], refined_location[1], marker_id)
            #print(f"Detected pos: x: {x}, y: {y}, refined x: {refined_location[0]:.2f}, refined y: {refined_location[1]:.2f}, dx: {dx:.2f}, dy: {dy:.2f} fully refined x: {pose.x:.2f}, fully refined y: {pose.y:.2f}")
            poses.append(pose)

        return poses, 255*frame_sum_squared/max_val
        
    def extract_window_around_marker_location(self, frame, marker_loc):
        (xm, ym) = marker_loc
        frame_tmp = np.array(frame[ym - self.y1:ym + self.y2, xm - self.x1:xm + self.x2])
        return frame_tmp

    def refine_marker_location(self, frame_sum_squared, x, y):
        try: 
            delta = 1
            # Fit a parabola to the frame_sum_squared marker response
            # and then locate the top of the parabola.
            frame_sum_squared_cutout = frame_sum_squared[x-delta:x+delta+1, y-delta:y+delta+1]

            # Taking the square root of the frame_sum_squared improves the accuracy of the 
            # refied marker position.
            frame_sum_squared_cutout = np.sqrt(frame_sum_squared_cutout)

            nx, ny = (1 + 2*delta, 1 + 2*delta)
            x = np.linspace(-delta, delta, nx)
            y = np.linspace(-delta, delta, ny)
            xv, yv = np.meshgrid(x, y)

            xv = xv.ravel()
            yv = yv.ravel()

            coefficients = np.concatenate([[xv**2], [xv], [yv**2], [yv], [yv**0]], axis = 0).transpose()
            values = frame_sum_squared_cutout.ravel().reshape(-1, 1)
            solution, residuals, rank, s = np.linalg.lstsq(coefficients, values, rcond=None)
            dx = -solution[1] / (2*solution[0])
            dy = -solution[3] / (2*solution[2])

            return dx[0], dy[0]
        except np.linalg.LinAlgError as e:
            # This error is triggered when the marker is detected close to an edge.
            # In that case the refine method bails out and returns two zeros.
            print("error in refine_marker_location")
            print(e)
            return 0, 0
        
