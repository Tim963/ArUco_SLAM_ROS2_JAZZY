import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from collections import deque

class ArucoMapper(Node):
    def __init__(self):
        super().__init__('aruco_mapper')
        
        # Initialisierung
        self.subscription = self.create_subscription(
            Image,
            'image_gray',
            self.image_callback,
            10)
        
        self.pose_publisher = self.create_publisher(PoseStamped, 'aruco_pose', 10)
        self.map_publisher = self.create_publisher(PoseArray, 'aruco_map', 10)
        self.bridge = CvBridge()
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        self.parameters = aruco.DetectorParameters_create()
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Kartendaten
        self.reference_marker_id = 0  # Fester Referenzmarker (Ursprung)
        self.marker_map = {}  # {marker_id: Transformationsmatrix zum Referenzmarker}
        self.marker_history = {}  # Für Driftkorrektur
        self.camera_poses = deque(maxlen=30)  # Letzte Kameraposen
        
        # Kalibrierung
        self.focal_length = 1000.0
        self.marker_size = 0.1  # in Metern
        self.drift_threshold = 0.05  # 5cm Drift erlaubt
        
        self.get_logger().info("Aruco Mapper mit Driftkorrektur gestartet")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'mono8')
            corners, ids, _ = aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.parameters)
            
            if ids is None or len(ids) == 0:
                return

            ids = ids.flatten()
            h, w = cv_image.shape[:2]
            camera_matrix = np.array([
                [self.focal_length, 0, w/2],
                [0, self.focal_length, h/2],
                [0, 0, 1]], dtype=np.float32)
            
            current_transforms = {}
            for i, marker_id in enumerate(ids):
                try:
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                        [corners[i][0]], self.marker_size, camera_matrix, np.zeros((5,1)))
                    
                    rot_matrix, _ = cv2.Rodrigues(rvec)
                    T_marker_to_cam = np.eye(4)
                    T_marker_to_cam[:3,:3] = rot_matrix
                    T_marker_to_cam[:3,3] = tvec[0][0]
                    current_transforms[int(marker_id)] = T_marker_to_cam
                    
                    cv2.drawFrameAxes(cv_image, camera_matrix, np.zeros((5,1)), rvec, tvec, self.marker_size/2)
                except Exception as e:
                    self.get_logger().warn(f"Marker {marker_id} Fehler: {str(e)}")

            cv_image = aruco.drawDetectedMarkers(cv_image, corners, ids)
            cv2.imshow("Aruco Detection", cv_image)
            cv2.waitKey(1)

            # Kartenaktualisierung und Driftkorrektur
            self.update_marker_map(current_transforms)
            self.estimate_camera_pose(current_transforms)
            self.check_and_correct_drift(current_transforms)
            self.publish_map()

        except Exception as e:
            self.get_logger().error(f"Callback Fehler: {str(e)}")

    def update_marker_map(self, current_transforms):
        """Aktualisiert die Karte mit neuen Markern"""
        if self.reference_marker_id in current_transforms:
            T_ref_to_cam = current_transforms[self.reference_marker_id]
            T_cam_to_ref = np.linalg.inv(T_ref_to_cam)
            
            for marker_id in current_transforms:
                if marker_id not in self.marker_map:
                    T_marker_to_ref = T_cam_to_ref @ current_transforms[marker_id]
                    self.marker_map[marker_id] = T_marker_to_ref
                    self.marker_history[marker_id] = deque(maxlen=10)
                    self.get_logger().info(f"Neuer Marker {marker_id} kartiert")

    def estimate_camera_pose(self, current_transforms):
        """Schätzt die Kameraposition unter Verwendung aller bekannter Marker"""
        known_markers = [m for m in current_transforms if m in self.marker_map]
        if not known_markers:
            return

        positions = []
        rotations = []
        
        for marker_id in known_markers:
            T_marker_to_ref = self.marker_map[marker_id]
            T_marker_to_cam = current_transforms[marker_id]
            T_cam_to_ref = T_marker_to_ref @ np.linalg.inv(T_marker_to_cam)
            
            positions.append(T_cam_to_ref[:3,3])
            rotations.append(T_cam_to_ref[:3,:3])
            
            # Für Driftkorrektur speichern
            self.marker_history[marker_id].append(T_marker_to_cam)

        avg_position = np.mean(positions, axis=0)
        avg_rotation = rotations[0]  # Vereinfachte Mittelung
        
        # Pose publizieren
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = float(avg_position[0])
        pose_msg.pose.position.y = float(avg_position[1])
        pose_msg.pose.position.z = float(avg_position[2])
        
        quaternion = self.rotation_matrix_to_quaternion(avg_rotation)
        pose_msg.pose.orientation.x = float(quaternion[0])
        pose_msg.pose.orientation.y = float(quaternion[1])
        pose_msg.pose.orientation.z = float(quaternion[2])
        pose_msg.pose.orientation.w = float(quaternion[3])
        
        self.pose_publisher.publish(pose_msg)
        self.camera_poses.append(avg_position)

    def check_and_correct_drift(self, current_transforms):
        """Überprüft und korrigiert Drift relativ zum Referenzmarker"""
        if self.reference_marker_id not in current_transforms:
            return

        T_ref_to_cam = current_transforms[self.reference_marker_id]
        T_cam_to_ref = np.linalg.inv(T_ref_to_cam)
        
        # Drift für alle Marker berechnen
        for marker_id in list(self.marker_map.keys()):
            if marker_id == self.reference_marker_id:
                continue
                
            if marker_id in current_transforms:
                # Aktuelle Transformation
                T_marker_to_cam = current_transforms[marker_id]
                T_marker_to_ref_current = T_cam_to_ref @ T_marker_to_cam
                
                # Original in der Karte gespeicherte Transformation
                T_marker_to_ref_original = self.marker_map[marker_id]
                
                # Drift berechnen
                drift_distance = np.linalg.norm(T_marker_to_ref_current[:3,3] - T_marker_to_ref_original[:3,3])
                
                if drift_distance > self.drift_threshold:
                    self.get_logger().warn(f"Drift bei Marker {marker_id}: {drift_distance:.3f}m")
                    # Korrektur anwenden (gewichtete Mittelung)
                    correction_factor = 0.3  # Wie stark korrigiert wird
                    corrected_transform = (1-correction_factor)*T_marker_to_ref_original + correction_factor*T_marker_to_ref_current
                    self.marker_map[marker_id] = corrected_transform

    def publish_map(self):
        map_msg = PoseArray()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = "map"
        
        for marker_id, transform in self.marker_map.items():
            pose = Pose()
            pose.position.x = float(transform[0,3])
            pose.position.y = float(transform[1,3])
            pose.position.z = float(transform[2,3])
            
            quaternion = self.rotation_matrix_to_quaternion(transform[:3,:3])
            pose.orientation.x = float(quaternion[0])
            pose.orientation.y = float(quaternion[1])
            pose.orientation.z = float(quaternion[2])
            pose.orientation.w = float(quaternion[3])
            
            map_msg.poses.append(pose)
        
        self.map_publisher.publish(map_msg)

    def rotation_matrix_to_quaternion(self, R):
        trace = R[0,0] + R[1,1] + R[2,2]
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2,1] - R[1,2]) / S
            qy = (R[0,2] - R[2,0]) / S
            qz = (R[1,0] - R[0,1]) / S
        elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            qw = (R[2,1] - R[1,2]) / S
            qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S
            qz = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S
        return np.array([qx, qy, qz, qw])

def main(args=None):
    rclpy.init(args=args)
    node = ArucoMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
