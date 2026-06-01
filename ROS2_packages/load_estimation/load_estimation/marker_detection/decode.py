import cv2
import math
import numpy as np

from load_estimation.marker_detection.generate_marker_codes import getRingCodes, findSmallestRotation


class decode_marker():
    def __init__(self, r_code_inner, r_code_outer, bits, transitions, marker_ids):
        self.r_code_inner = r_code_inner
        self.r_code_outer = r_code_outer
        self.r_code = (r_code_inner + r_code_outer)/2

        self.bits = bits
        self.transitions = transitions
        #print("Generating ring codes for bits:", bits, "transitions:", transitions)
        self.ring_codes = getRingCodes(bits, transitions)
        #print("Generated ring codes:", self.ring_codes)
        self.ring_codes = marker_ids


        self.r_voting = 0 # 0 will mean a single pixel, and 1 is a circle with diameter 3
        # precompute mask for a given radius
        y, x = np.ogrid[-self.r_voting:self.r_voting+1, -self.r_voting:self.r_voting+1]
        self.mask = x*x + y*y <= self.r_voting*self.r_voting

        #print(self.mask)


    def extract_and_decode(self, image, center):
        #print("center: ", center)
        #print(f"image shape: {image.shape}, center: {center}, r_code_outer: {self.r_code_outer}")

        marker = image[int(center[1] - self.r_code_outer):int(center[1] + self.r_code_outer + 1), 
                           int(center[0] - self.r_code_outer):int(center[0] + self.r_code_outer + 1)]
        

        #print("extracted marker shape:", marker.shape)
        if len(marker.shape) > 2:
            marker_gray = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)
            _, marker_bin = cv2.threshold(marker_gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, marker_bin = cv2.threshold(marker, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if marker_bin is None:
            # Sometimes thresholding fails
            return None


        samples = []
        for i in range(6):
            sample = []
            for j in range(self.bits):
                theta = 2 * math.pi * j / (self.bits) + (math.pi/36)*i
                sample_x = self.r_code_outer + int(self.r_code * math.cos(theta))
                sample_y = self.r_code_outer + int(self.r_code * math.sin(theta))
                result = self.circle_vote(marker_bin, sample_x, sample_y)
                if result is None:
                    #print("circle_vote returned None")
                    return None
                sample.append(result)
                cv2.circle(image, (sample_x+center[0]-self.r_code_outer, sample_y+center[1]-self.r_code_outer), self.r_voting, (0,0,255), -1)

            sample = self.list_to_binary(sample)
            min_sample = findSmallestRotation(sample, self.bits)
            #print(min_sample)
            samples.append(min_sample)

        #cv2.imshow("marker", marker / marker.max())
        #cv2.waitKey(0)
        voted_marker_id = max(set(samples), key=samples.count)

        ring_code_count = {}
        for code in self.ring_codes:
            ring_code_count.update({code:samples.count(code)})
        voted_marker_id = max(ring_code_count, key=ring_code_count.get)

        print_marker = False
        if ring_code_count[voted_marker_id] >= 2:
            if print_marker:
                cv2.imshow("marker", cv2.resize(marker, (0,0), fx=10.0, fy=10.0))
                cv2.imshow("marker_bin", cv2.resize(marker_bin * 255, (0,0), fx=10.0, fy=10.0))
            #cv2.circle(image, (center[0], center[1]), self.r_code_outer, (0,255,0), 2)
            return voted_marker_id
        else:
            #print("marker id not found in ring codes: ", voted_marker_id, samples, ring_code_count)
            if print_marker:
                cv2.imshow("marker", cv2.resize(marker, (0,0), fx=10.0, fy=10.0))
                cv2.imshow("marker_bin", cv2.resize(marker_bin * 255, (0,0), fx=10.0, fy=10.0))
            #print("samples:", samples)
            return None


    def list_to_binary(self, list):
        bits = [int(b) for b in list]
        if any(b not in (0, 1) for b in bits):
            raise ValueError("Input must be a list of 0s and 1s")
        value = 0
        for b in bits:
            value = (value << 1) | b
        return value
    
    def circle_vote(self, img, cx, cy):
        # extract region of interest
        x1, x2 = cx - self.r_voting, cx + self.r_voting + 1
        y1, y2 = cy - self.r_voting, cy + self.r_voting + 1
        roi = img[x1:x2, y1:y2]
        #print(f"ROI shape: {roi.shape}, expected shape: {self.mask.shape}, img shape: {img.shape}, cx: {cx}, cy: {cy}, x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}")

        # binary decision for black/white region
        # print(roi.shape)
        if roi.shape != self.mask.shape:
            #print("ROI shape does not match mask shape")
            #print(f"ROI shape: {roi.shape}, mask shape: {self.mask.shape}, img shape: {img.shape}, cx: {cx}, cy: {cy}, x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}")
            cv2.imshow("img", cv2.resize(img * 255, (0,0), fx=10.0, fy=10.0))
            cv2.waitKey(0)
            return None  # default to black if out of bounds
        black_votes = (roi[self.mask] == 0).sum()
        white_votes = (roi[self.mask] != 0).sum()

        #print(f"black: {black_votes}, white: {white_votes}")

        if black_votes < white_votes:
            return 1
        else:
            return 0


if __name__ == "__main__":
    # ========== Variables ==========
    r_code_inner = 30
    r_code_outer = 42
    bits = 8
    transitions = 2

    # ========== decoder init ==========
    decoder = decode_marker(r_code_inner, r_code_outer, bits, transitions)

    # ========== read image ==========
    bgr_image = cv2.imread('Coded_markers.jpg', cv2.IMREAD_COLOR)
    print("Image shape:", bgr_image.shape)
    scale_factor = 0.25
    #bgr_image = cv2.resize(bgr_image, (0, 0), fx=scale_factor, fy=scale_factor)
    cx, cy = 580, 255  # Example center coordinates

    # ========== decode at known center ==========
    result = decoder.extract_and_decode(bgr_image, (cx,cy))
    print("Decoded marker ID:", result)

    cv2.imshow("marker", bgr_image)
    cv2.waitKey(0)


