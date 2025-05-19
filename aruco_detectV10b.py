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
from collections import defaultdict, deque

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
        self.reference_marker_id = 0
        self.marker_map = {}  # {marker_id: T_marker_to_ref}
        self.marker_connections = defaultdict(set)  # Speichert Marker-Verbindungen
        self.observed_pairs = defaultdict(list)  # Relative Positionen zwischen Markern
        
        # Kalibrierung
        self.focal_length = 1000.0
        self.marker_size = 0.1
        self.drift_threshold = 0.05  # 5cm
        self.global_correction_factor = 0.2  # Konservative Korrektur
        
        self.get_logger().info("Aruco Mapper mit vernetzter Driftkorrektur (ohne networkx) gestartet")

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
            visible_markers = set()
            
            # 1. Marker detektieren und Transformationen berechnen
            for i, marker_id in enumerate(ids):
                try:
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                        [corners[i][0]], self.marker_size, camera_matrix, np.zeros((5,1)))
                    
                    rot_matrix, _ = cv2.Rodrigues(rvec)
                    T_marker_to_cam = np.eye(4)
                    T_marker_to_cam[:3,:3] = rot_matrix
                    T_marker_to_cam[:3,3] = tvec[0][0]
                    current_transforms[int(marker_id)] = T_marker_to_cam
                    visible_markers.add(int(marker_id))
                    
                    cv2.drawFrameAxes(cv_image, camera_matrix, np.zeros((5,1)), rvec, tvec, self.marker_size/2)
                except Exception as e:
                    self.get_logger().warn(f"Marker {marker_id} Fehler: {str(e)}")

            cv_image = aruco.drawDetectedMarkers(cv_image, corners, ids)
            cv2.imshow("Aruco Detection", cv_image)
            cv2.waitKey(1)

            # 2. Relative Positionen zwischen sichtbaren Markern speichern
            self.record_relative_positions(current_transforms)
            
            # 3. Kameraposition schätzen (funktioniert mit jedem bekannten Marker)
            self.estimate_camera_pose(current_transforms)
            
            # 4. Karte aktualisieren und Drift korrigieren
            self.update_marker_map(current_transforms)
            self.global_drift_correction(visible_markers)
            
            # 5. Aktualisierte Karte publizieren
            self.publish_map()

        except Exception as e:
            self.get_logger().error(f"Callback Fehler: {str(e)}")

    def record_relative_positions(self, current_transforms):
        """Speichert relative Positionen zwischen allen sichtbaren Markern"""
        marker_ids = list(current_transforms.keys())
        for i in range(len(marker_ids)):
            for j in range(i+1, len(marker_ids)):
                id1, id2 = marker_ids[i], marker_ids[j]
                T1 = current_transforms[id1]
                T2 = current_transforms[id2]
                T_rel = np.linalg.inv(T1) @ T2  # Transformation von id1 zu id2
                self.observed_pairs[(id1, id2)].append(T_rel)
                self.marker_connections[id1].add(id2)
                self.marker_connections[id2].add(id1)

    def find_shortest_path(self, start, target):
        """BFS-Algorithmus für Pfadsuche zwischen Markern"""
        visited = set()
        queue = [[start]]
        
        if start == target:
            return [start]
            
        while queue:
            path = queue.pop(0)
            node = path[-1]
            
            if node not in visited:
                neighbors = self.marker_connections.get(node, set())
                for neighbor in neighbors:
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
                    
                    if neighbor == target:
                        return new_path
                visited.add(node)
        return None

    def update_marker_map(self, current_transforms):
        """Fügt neue Marker zur Karte hinzu"""
        known_markers = [m for m in current_transforms if m in self.marker_map]
        
        if not known_markers and self.reference_marker_id in current_transforms:
            # Initialer Marker
            T_ref_to_cam = current_transforms[self.reference_marker_id]
            T_cam_to_ref = np.linalg.inv(T_ref_to_cam)
            self.marker_map[self.reference_marker_id] = np.eye(4)
            known_markers = [self.reference_marker_id]
        
        for marker_id in current_transforms:
            if marker_id not in self.marker_map and known_markers:
                # Nehme den nächstgelegenen bekannten Marker als Referenz
                ref_id = min(known_markers, 
                           key=lambda x: np.linalg.norm(current_transforms[x][:3,3] - current_transforms[marker_id][:3,3]))
                
                T_ref_to_cam = current_transforms[ref_id]
                T_cam_to_ref = np.linalg.inv(T_ref_to_cam)
                T_marker_to_ref = T_cam_to_ref @ current_transforms[marker_id]
                self.marker_map[marker_id] = T_marker_to_ref
                self.get_logger().info(f"Neuer Marker {marker_id} hinzugefügt (relativ zu {ref_id})")

    def estimate_camera_pose(self, current_transforms):
        """Schätzt die Kameraposition basierend auf allen bekannten Markern"""
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

        avg_position = np.mean(positions, axis=0)
        avg_rotation = self.average_rotations(rotations)
        
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

    def global_drift_correction(self, visible_markers):
        """Führt eine globale Driftkorrektur durch"""
        if len(visible_markers) < 2:
            return

        try:
            # 1. Berechne durchschnittliche Abweichungen für Markerpaare
            pair_errors = {}
            for (id1, id2), observations in self.observed_pairs.items():
                if id1 in visible_markers and id2 in visible_markers:
                    avg_observed = np.mean(observations[-10:], axis=0)  # Letzte 10 Beobachtungen
                    T1 = self.marker_map[id1]
                    T2 = self.marker_map[id2]
                    T_expected = np.linalg.inv(T1) @ T2
                    error = np.linalg.norm(avg_observed[:3,3] - T_expected[:3,3])
                    pair_errors[(id1, id2)] = error

            if not pair_errors:
                return

            # 2. Finde den Marker mit der größten inkonsistenz
            worst_pair = max(pair_errors.items(), key=lambda x: x[1])[0]
            if pair_errors[worst_pair] > self.drift_threshold:
                self.get_logger().info(f"Driftkorrektur aktiv (max Error: {pair_errors[worst_pair]:.3f}m)")
                
                # 3. Berechne Korrektur für direkt verbundene Marker
                ref_id = worst_pair[0]
                for marker_id in self.marker_connections.get(ref_id, set()):
                    if marker_id in self.marker_map:
                        try:
                            # Finde Beobachtungen zwischen den Markern
                            if (ref_id, marker_id) in self.observed_pairs:
                                T_rel = np.mean(self.observed_pairs[(ref_id, marker_id)][-5:], axis=0)
                            else:
                                T_rel = np.mean(self.observed_pairs[(marker_id, ref_id)][-5:], axis=0)
                                T_rel = np.linalg.inv(T_rel)
                            
                            # Berechne erwartete Position
                            T_expected = self.marker_map[ref_id] @ T_rel
                            
                            # Berechne Korrektur
                            correction = T_expected @ np.linalg.inv(self.marker_map[marker_id])
                            
                            # Wende Korrektur an (nur Position)
                            new_pos = (1-self.global_correction_factor)*self.marker_map[marker_id][:3,3] + \
                                      self.global_correction_factor*(self.marker_map[marker_id] @ correction)[:3,3]
                            self.marker_map[marker_id][:3,3] = new_pos
                        except Exception as e:
                            self.get_logger().warn(f"Korrektur für {marker_id} fehlgeschlagen: {str(e)}")

        except Exception as e:
            self.get_logger().warn(f"Driftkorrektur Fehler: {str(e)}")

    def average_rotations(self, rotations):
        """Einfache Rotationmittelung (könnte mit Quaternionen verbessert werden)"""
        return rotations[0] if rotations else np.eye(3)

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
