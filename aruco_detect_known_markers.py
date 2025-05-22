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

class ArucoMapper(Node):
    def __init__(self):
        super().__init__('aruco_mapper')
        
        # Initialisierung
        self.subscription = self.create_subscription(
            Image,
            'image_gray',
            self.image_callback,
            10)
        
        self.pose_publisher = self.create_publisher(PoseStamped, 'aruco_odom', 10)
        self.map_publisher = self.create_publisher(PoseArray, 'aruco_map', 10)
        self.bridge = CvBridge()
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50) #aruco.DICT_4X4_250
        self.parameters = aruco.DetectorParameters_create()
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Kartendaten
        self.reference_marker_id = 0  # Fester Referenzmarker (Ursprung)
        self.marker_map = {}  # {marker_id: Transformationsmatrix zum Referenzmarker}
        
        # Virtuelle Kalibrierung
        self.focal_length = 1000.0
        self.marker_size = 0.1  # in Metern
        self.initialize_room_markers()
        
        self.get_logger().info("Aruco Mapper gestartet - Referenzmarker ID 0")
        
    def add_known_marker(self, marker_id, x, y, z, roll, pitch, yaw):
        """Fügt einen bekannten Marker mit globaler Pose zur Karte hinzu"""
        quat = self.euler_to_quaternion(roll, pitch, yaw)
        T = np.eye(4)
        T[:3, 3] = [x, y, z]
        T[:3, :3] = self.quaternion_to_matrix(quat)
        self.marker_map[marker_id] = T
    
    def quaternion_to_matrix(self, q):
        """Wandelt Quaternion in 3x3-Rotationsmatrix um"""
        x, y, z, w = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        """Wandelt Eulerwinkel (roll, pitch, yaw) in Quaternion [x, y, z, w] um"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy
        return [qx, qy, qz, qw]

    def initialize_room_markers(self):
        """Definiert bekannte Marker korrekt mit globaler Ausrichtung"""
        self.add_known_marker(0, 0.0, 0.0, 0.0, 0.0, 0.0, np.pi/2)              # Boden RICHTIG
        self.add_known_marker(1, 0.0, -0.3, 0.2, np.pi/2, 0.0, np.pi)         # Linke Wand RICHTIG
        self.add_known_marker(2, -0.3, 0.0, 0.2, np.pi/2, 0.0, np.pi/2)     # Vordere Wand RICHTIG
        self.add_known_marker(3, 0.0, 0.3, 0.2, np.pi/2, 0.0, 0.0)        # Rechte Wand RICHTIG
        self.add_known_marker(4, 0.3, 0.0, 0.2, np.pi/2, 0.0, -np.pi/2)     # Hintere Wand RICHTIG





    def image_callback(self, msg):
        try:
            # Bildkonvertierung
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'mono8')
            
            # Markererkennung
            corners, ids, _ = aruco.detectMarkers(
                cv_image, 
                self.aruco_dict, 
                parameters=self.parameters)
            
            if ids is None or len(ids) == 0:
                return  # Keine Marker erkannt

            # Sicherstellen, dass corners und ids verarbeitet werden können
            ids = ids.reshape(-1)  # IDs in 1D-Array umwandeln
            num_markers = len(ids)
            
            # Kameramatrix erstellen
            h, w = cv_image.shape[:2]
            camera_matrix = np.array([
                [self.focal_length, 0, w/2],
                [0, self.focal_length, h/2],
                [0, 0, 1]], dtype=np.float32)
            
            # Transformationsmatrizen für alle sichtbaren Marker berechnen
            current_transforms = {}
            
            for i in range(num_markers):
                try:
                    marker_id = int(ids[i])
                    corners_i = np.array([corners[i][0]])  # Korrekte Corner-Formatierung
                    
                    # Pose-Schätzung für jeden Marker
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                        corners_i,
                        self.marker_size,
                        camera_matrix,
                        np.zeros((5,1)))
                    
                    # Transformationsmatrix erstellen
                    rot_matrix, _ = cv2.Rodrigues(rvec)
                    T_marker_to_cam = np.eye(4)
                    T_marker_to_cam[:3,:3] = rot_matrix
                    T_marker_to_cam[:3,3] = tvec[0][0]
                    current_transforms[marker_id] = T_marker_to_cam
                    
                    # Visualisierung
                    cv2.drawFrameAxes(cv_image, camera_matrix, np.zeros((5,1)), 
                                    rvec, tvec, self.marker_size/2)
                except Exception as e:
                    self.get_logger().warn(f"Fehler bei Marker {marker_id}: {str(e)}")
                    continue
            
            # Marker zeichnen
            cv_image = aruco.drawDetectedMarkers(cv_image, corners, ids)
            
            # Neue Marker zur Karte hinzufügen (wenn Referenzmarker oder andere bekannte Marker sichtbar sind)
            self.update_marker_map(current_transforms)
            
            # Kameraposition berechnen (relativ zu bekannten Markern)
            self.estimate_camera_pose(current_transforms)
            
            # Karte publizieren
            self.publish_map()
            
            # Bild anzeigen
            cv2.imshow("Aruco Detection", cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Allgemeiner Fehler: {str(e)}", throttle_duration_sec=1)

    def update_marker_map(self, current_transforms):
        """Fügt neue Marker zur Karte hinzu, wenn sie zusammen mit bekannten Markern gesehen werden"""
        # Wenn Referenzmarker sichtbar ist und noch nicht in der Karte
        if self.reference_marker_id in current_transforms and self.reference_marker_id not in self.marker_map:
            T_ref_to_cam = current_transforms[self.reference_marker_id]
            T_cam_to_ref = np.linalg.inv(T_ref_to_cam)
            self.marker_map[self.reference_marker_id] = np.eye(4)  # Identitätsmatrix
            
            # Alle anderen sichtbaren Marker hinzufügen
            for marker_id in current_transforms:
                if marker_id != self.reference_marker_id:
                    T_marker_to_cam = current_transforms[marker_id]
                    T_marker_to_ref = T_cam_to_ref @ T_marker_to_cam
                    self.marker_map[marker_id] = T_marker_to_ref
                    self.get_logger().info(f"Initialer Marker {marker_id} hinzugefügt")
        
        # Wenn andere bekannte Marker sichtbar sind
        known_markers = [m for m in current_transforms if m in self.marker_map]
        if known_markers:
            reference_id = known_markers[0]  # Verwende den ersten bekannten Marker
            T_ref_to_cam = current_transforms[reference_id]
            T_cam_to_ref = np.linalg.inv(T_ref_to_cam)
            
            # Neue Marker hinzufügen
            for marker_id in current_transforms:
                if marker_id not in self.marker_map:
                    T_marker_to_cam = current_transforms[marker_id]
                    T_marker_to_ref = self.marker_map[reference_id] @ T_cam_to_ref @ T_marker_to_cam
                    self.marker_map[marker_id] = T_marker_to_ref
                    self.get_logger().info(f"Neuer Marker {marker_id} zur Karte hinzugefügt (relativ zu Marker {reference_id})")

    def estimate_camera_pose(self, current_transforms):
        """Schätzt die Kameraposition relativ zum Referenzkoordinatensystem"""
        # Finde alle bekannten Marker, die aktuell sichtbar sind
        known_markers = [m for m in current_transforms if m in self.marker_map]
        
        if not known_markers:
            return  # Keine bekannten Marker sichtbar
        
        # Durchschnittliche Kameraposition berechnen (über alle bekannten Marker)
        positions = []
        rotations = []
        
        for marker_id in known_markers:
            T_marker_to_cam = current_transforms[marker_id]
            T_cam_to_marker = np.linalg.inv(T_marker_to_cam)
            T_cam_to_ref = self.marker_map[marker_id] @ T_cam_to_marker
            
            positions.append(T_cam_to_ref[:3,3])
            rotations.append(T_cam_to_ref[:3,:3])
        
        # Durchschnittsposition und -rotation berechnen
        avg_position = np.mean(positions, axis=0)
        avg_rotation = self.average_rotations(rotations)
        quaternion = self.rotation_matrix_to_quaternion(avg_rotation)
        
        # Pose publizieren
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = float(avg_position[0])
        pose_msg.pose.position.y = float(avg_position[1])
        pose_msg.pose.position.z = float(avg_position[2])
        pose_msg.pose.orientation.x = float(quaternion[0])
        pose_msg.pose.orientation.y = float(quaternion[1])
        pose_msg.pose.orientation.z = float(quaternion[2])
        pose_msg.pose.orientation.w = float(quaternion[3])
        self.pose_publisher.publish(pose_msg)
        
        # TF-Transformation senden
        transform = TransformStamped()
        transform.header.stamp = pose_msg.header.stamp
        transform.header.frame_id = "map"
        transform.child_frame_id = "camera"
        transform.transform.translation.x = pose_msg.pose.position.x
        transform.transform.translation.y = pose_msg.pose.position.y
        transform.transform.translation.z = pose_msg.pose.position.z
        transform.transform.rotation = pose_msg.pose.orientation
        self.tf_broadcaster.sendTransform(transform)

    def average_rotations(self, rotations):
        """Berechnet die durchschnittliche Rotation aus mehreren Rotationsmatrizen"""
        # Einfache Implementierung: Nehme die erste Matrix
        # (Für bessere Ergebnisse könnte man Quaternion-Mittelung verwenden)
        return rotations[0]

    def publish_map(self):
        try:
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
        except Exception as e:
            self.get_logger().error(f"Fehler in publish_map: {str(e)}", throttle_duration_sec=1)

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