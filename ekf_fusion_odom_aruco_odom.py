#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance, TwistWithCovariance
import numpy as np
from numpy.linalg import inv

class EKFFusion(Node):
    def __init__(self):
        super().__init__('ekf_fusion')
        
        # EKF State: [x, y, theta, vx, vy, omega] (Pose + Twist)
        self.x = np.zeros(6)
        self.P = np.eye(6)  # Kovarianzmatrix
        
        # Rauschparameter (anpassen!)
        self.Q_odom = np.diag([0.1, 0.1, 0.05, 0.01, 0.01, 0.01])  # Odometrie-Rauschen
        self.R_aruco = np.diag([0.05, 0.05, 0.02])  # ArUco-Messrauschen (nur Pose)
        
        # Subscriber
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.sub_aruco = self.create_subscription(
            Odometry, '/aruco_odom', self.aruco_callback, 10)  #/aruco_odom_transformed
        
        # Publisher
        self.pub_fused_odom = self.create_publisher(
            Odometry, '/fused_odom', 10)
        
        self.last_odom_time = self.get_clock().now()
        self.last_twist = np.zeros(3)  # [vx, vy, omega] aus /odom

    def odom_callback(self, msg):
        # Twist-Daten aus /odom speichern
        self.last_twist = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.angular.z
        ])
        
        # Prädiktionsschritt (nur mit Twist, falls keine ArUco-Daten)
        current_time = self.get_clock().now()
        dt = (current_time - self.last_odom_time).nanoseconds * 1e-9
        self.last_odom_time = current_time
        
        self.predict(dt)
        self.publish_fused_odom()

    def aruco_callback(self, msg):
        # Pose-Daten aus /aruco_odom (kein Twist!)
        aruco_x = msg.pose.pose.position.x
        aruco_y = msg.pose.pose.position.y
        aruco_theta = self.quaternion_to_yaw(msg.pose.pose.orientation)
        z = np.array([aruco_x, aruco_y, aruco_theta])
        
        # Update-Schritt mit ArUco-Pose
        self.update(z)

    def predict(self, dt):
        # Bewegungsmodell: Integration der Twist-Daten
        theta = self.x[2]
        self.x[0] += self.last_twist[0] * dt * np.cos(theta) - self.last_twist[1] * dt * np.sin(theta)
        self.x[1] += self.last_twist[0] * dt * np.sin(theta) + self.last_twist[1] * dt * np.cos(theta)
        self.x[2] += self.last_twist[2] * dt
        self.x[3:] = self.last_twist  # Geschwindigkeiten direkt übernehmen
        
        # Jacobi-Matrix des Bewegungsmodells
        F = np.eye(6)
        F[0, 2] = -self.last_twist[0] * dt * np.sin(theta) - self.last_twist[1] * dt * np.cos(theta)
        F[1, 2] = self.last_twist[0] * dt * np.cos(theta) - self.last_twist[1] * dt * np.sin(theta)
        
        # Kovarianz aktualisieren
        self.P = F @ self.P @ F.T + self.Q_odom

    def update(self, z):
        # Messmatrix H (misst nur Pose, nicht Twist)
        H = np.zeros((3, 6))
        H[0, 0] = 1  # x
        H[1, 1] = 1  # y
        H[2, 2] = 1  # theta
        
        # Kalman-Gain
        S = H @ self.P @ H.T + self.R_aruco
        K = self.P @ H.T @ inv(S)
        
        # Zustand und Kovarianz korrigieren
        y = z - H @ self.x
        self.x += K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    def publish_fused_odom(self):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"  # Frame von ArUco
        msg.child_frame_id = "base_link"
        
        # Pose
        msg.pose.pose.position.x = self.x[0]
        msg.pose.pose.position.y = self.x[1]
        msg.pose.pose.position.z = 0.0  # ArUco liefert z, aber wir ignorieren es hier
        msg.pose.pose.orientation = self.yaw_to_quaternion(self.x[2])
        
        # Twist (aus /odom übernommen)
        msg.twist.twist.linear.x = self.x[3]
        msg.twist.twist.linear.y = self.x[4]
        msg.twist.twist.angular.z = self.x[5]
        
        self.pub_fused_odom.publish(msg)

    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def yaw_to_quaternion(self, yaw):
        from geometry_msgs.msg import Quaternion
        q = Quaternion()
        q.w = np.cos(yaw / 2)
        q.x = 0.0
        q.y = 0.0
        q.z = np.sin(yaw / 2)
        return q

def main(args=None):
    rclpy.init(args=args)
    ekf_fusion = EKFFusion()
    rclpy.spin(ekf_fusion)
    ekf_fusion.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()