import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge

import time
import gpiod
from random import randrange


class LedImageTimerNode(Node):

    def __init__(self):
        super().__init__('led_image_timer_node')
        self.get_logger().info("Initializing LED Image Timer Node")

        # ========== Publishers ==========
        self.publish_timer_ = self.create_publisher(Float64, '/timer', 2)

        # ========== Subscribers ==========
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, 10)

        # ========== Timers ==========
        timer_period = 1/1000  # Check every 1/1000 sec for brightness (adjust as needed)
        self.timer_LED = self.create_timer(timer_period, self.led_timer_callback)

        # ========== GPIO Setup ==========
        self.chip = gpiod.Chip('/dev/gpiochip4')
        self.led = self.chip.get_line(17)
        self.led.request(consumer="led", type=gpiod.LINE_REQ_DIR_OUT)

        # ========== CONFIG ==========
        self.brightness_threshold = 10  # 0–255 scale

        self.bridge = CvBridge()

        # ========== STATE ==========
        self.LED_counter = 0
        self.LED_turned_on = False

        self.timer_running = False
        self.start_time = None

        self.random_LED_delay = randrange(40, 80, 1)

        self.get_logger().info("Brightness-based LED Timer Node started")

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # ---------------- MONITOR CENTER PIXEL ----------------
        if self.timer_running:
            b, g, r = frame[0, 160]  # Center pixel (320x240 resolution)
            if self.is_center_pixel_bright(b, g, r):
                self.stop_timer()


    def led_timer_callback(self):
        if self.LED_counter == self.random_LED_delay:
            self.led.set_value(1)
            self.LED_turned_on = True
            self.start_timer()  # Start timer when LED is turned on
        if self.LED_counter == self.random_LED_delay + 6:
            self.led.set_value(0)
            self.LED_turned_on = False

            self.random_LED_delay = randrange(40, 80, 1)  # Randomize next LED delay
            self.LED_counter = 0

        self.LED_counter += 1


         

    # =========================================================
    # TIMER
    # =========================================================
    def start_timer(self):
        self.start_time = time.time()
        self.timer_running = True
        #self.get_logger().info("Timer started")

    def stop_timer(self):
        elapsed = time.time() - self.start_time
        msg = Float64()
        msg.data = elapsed
        self.publish_timer_.publish(msg)

        self.timer_running = False
        self.start_time = None

        self.get_logger().info(f"BRIGHT DETECTED → elapsed time: {elapsed:.4f} sec")
        with open('camera_latency_25_top.csv', 'a') as f:
            f.write(f"{elapsed}\n")

    # =========================================================
    # BRIGHTNESS CHECK
    # =========================================================
    def is_center_pixel_bright(self, b, g, r):

        # luminance (standard perceived brightness)
        brightness = 0.114 * b + 0.587 * g + 0.299 * r

        # luminance (standard perceived brightness)
        #brightness = 0.114 * b + 0.587 * g + 0.299 * r
        #self.get_logger().info(f"brightness: {brightness:.4f}")

        return brightness > self.brightness_threshold


# =========================================================
# MAIN
# =========================================================
def main(args=None):
    rclpy.init(args=args)

    node = LedImageTimerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.led.set_value(0)
        node.led.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
