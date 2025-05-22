#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
import tf_transformations as tf_trans
import math

class ArucoRepublisher(Node):
    def __init__(self):
        super().__init__('aruco_odom_republisher')
        
        # Kameraposition als Parameter (X, Y, Z Offset in Metern)
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_offset_x', 0.0),  # seitlich
                ('camera_offset_y', 0.17), # vor/zurück
                ('camera_offset_z', 0.7)   # Höhenoffset
            ])
        
        self.camera_offset = [
            self.get_parameter('camera_offset_x').value,
            self.get_parameter('camera_offset_y').value,
            self.get_parameter('camera_offset_z').value
        ]
        
        self.get_logger().info(f"Kamera-Offset eingestellt: {self.camera_offset}")
        
        
        self.subscription = self.create_subscription(
            PoseStamped,
            'aruco_odom',
            self.callback,
            10)
        
        self.publisher = self.create_publisher(
            PoseStamped,
            'aruco_odom_transformed',
            10)
        
    def callback(self, msg):
        rotated_msg = PoseStamped()
        rotated_msg.header = msg.header
        
        # 1. Pose drehen (180° um Z-Achse)
        original_pose = msg.pose
        quat = [
            original_pose.orientation.x,
            original_pose.orientation.y,
            original_pose.orientation.z,
            original_pose.orientation.w
        ]
        
        euler = tf_trans.euler_from_quaternion(quat)
        rotated_euler = (euler[0], euler[1], euler[2] + math.pi)
        rotated_quat = tf_trans.quaternion_from_euler(*rotated_euler)
        
        # 2. Position korrigieren (Kamera-Offset anwenden)
        corrected_position = Point()
        corrected_position.x = original_pose.position.x - self.camera_offset[0]
        corrected_position.y = original_pose.position.y - self.camera_offset[1]
        corrected_position.z = original_pose.position.z - self.camera_offset[2]
        
        
        rotated_msg.pose.position = corrected_position
        rotated_msg.pose.orientation.x = rotated_quat[0]
        rotated_msg.pose.orientation.y = rotated_quat[1]
        rotated_msg.pose.orientation.z = rotated_quat[2]
        rotated_msg.pose.orientation.w = rotated_quat[3]
        
        self.publisher.publish(rotated_msg)
        self.get_logger().debug(f"Korrigierte Pose: {rotated_msg.pose}")

def main(args=None):
    rclpy.init(args=args)
    node = ArucoRepublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
