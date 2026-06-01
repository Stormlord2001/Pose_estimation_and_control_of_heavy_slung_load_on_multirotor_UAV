import rclpy
import math
import numpy as np
import time
from rclpy.node import Node
from geometry_msgs.msg import Vector3


# Dynamixel stuff
from dynamixel_sdk import *  # Uses Dynamixel SDK library

# -------------------- Dynamixel Setup --------------------
# Control table address (XL430)
ADDR_TORQUE_ENABLE       = 64
ADDR_GOAL_POSITION       = 116
ADDR_PRESENT_POSITION    = 132
LEN_GOAL_POSITION        = 4

# Protocol version
PROTOCOL_VERSION         = 2.0

# Default setting
DXL_IDs                  = [0,1]   # Motor IDs
BAUDRATE                 = 57600
DEVICENAME               = '/dev/ttyUSB0' 

TORQUE_ENABLE            = 1
TORQUE_DISABLE           = 0
DXL_MIN_POS              = [525, 0]
DXL_MAX_POS              = [1539, 1350]   # XL430 has 0–4095 for 0–360°

# Initialize PortHandler and PacketHandler
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Create GroupSyncWrite instance
groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)


class DynGimbalControl(Node):
    def __init__(self):
        super().__init__('dyn_gimbal_control')
        self.get_logger().info("Dynamixel Gimbal Control Node Started")

        # ==================== Publishers ====================
        self.publisher_angles_ = self.create_publisher(Vector3, '/gimbal/angles', 10)

        # ==================== Subscribers ====================
        self.subscription_angle_error_ = self.create_subscription(
            Vector3,
            '/gimbal/camera_load_angle',
            self.angle_error_callback,
            10
        )

        # ==================== Timers ====================
        timer_period = 1/30  # seconds
        self.timer = self.create_timer(timer_period, self.read_motor_angles)


        # ==================== Dynamixel Initialization ====================
        # Open port
        if not portHandler.openPort():
            raise IOError("Failed to open port")

        # Set baudrate
        if not portHandler.setBaudRate(BAUDRATE):
            raise IOError("Failed to set baudrate")

        # Enable torque for all motors
        for dxl_id in DXL_IDs:
            dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
            self.get_logger().info(f"Enable Torque Motor {dxl_id}: CommResult={dxl_comm_result}, Error={dxl_error}")
    
        # Store motor positions in bits for offset handling
        self.motor_bits = [0, 0]
        
        # Move to initial position 
        initial_position = [1024, 1024] # 90 degrees for both motors
        self.set_motor_goal_angle(initial_position)
        self.get_logger().info(f"Moving to initial position: {initial_position}")

        

    def angle_error_callback(self, msg):
        motor_1_error_bits = self.angle_to_bits(msg.x)
        motor_2_error_bits = self.angle_to_bits(msg.y)
        position = [motor_1_error_bits + self.motor_bits[0], motor_2_error_bits + self.motor_bits[1]]
        self.set_motor_goal_angle(position)

    def set_motor_goal_angle(self, position):
        
        for dxl_id, pos in zip(DXL_IDs, position):
            # Ensure position is within limits
            pos = max(DXL_MIN_POS[dxl_id], min(DXL_MAX_POS[dxl_id], pos))
            # Convert position to byte array
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(pos)),
                DXL_HIBYTE(DXL_LOWORD(pos)),
                DXL_LOBYTE(DXL_HIWORD(pos)),
                DXL_HIBYTE(DXL_HIWORD(pos))
            ]
        
            # Add parameter storage for Dynamixel SyncWrite
            groupSyncWrite.addParam(dxl_id, param_goal_position)
        
        # SyncWrite goal position
        dxl_comm_result = groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            self.get_logger().error(f"Failed to send SyncWrite: {packetHandler.getTxRxResult(dxl_comm_result)}")
        groupSyncWrite.clearParam()

    def read_motor_angles(self):
        msg = Vector3()

        positions = []
        current_pos_bits = []
        positions_msg = []
        for dxl_id in DXL_IDs:
            dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
                portHandler, dxl_id, ADDR_PRESENT_POSITION
            )
            if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                positions.append(dxl_present_position)
                positions_msg.append(f"ID {dxl_id}: {self.bits_to_angle(dxl_present_position):.2f}°")
                current_pos_bits.append(dxl_present_position)
            else:
                positions.append(-5000.0)
                positions_msg.append(f"ID {dxl_id}: ReadFail")

        self.motor_bits = [positions[0], positions[1]]
        msg.x = self.bits_to_angle(positions[0]) - 90
        msg.y = self.bits_to_angle(positions[1])
        msg.z = 0.0
        self.publisher_angles_.publish(msg)

        self.get_logger().info(" | ".join(positions_msg))

    def angle_to_bits(self, angle_deg):
        # Convert angle in degrees to bits (0-4095 for 0-360°)
        return int((angle_deg / 360.0) * 4095)
    
    def bits_to_angle(self, bits):
        # Convert bits to angle in degrees
        return ((bits % 4096) / 4096.0) * 360.0



def main(args=None):
    rclpy.init(args=args)
    node = DynGimbalControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
