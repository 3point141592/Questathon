import sys
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Optional


class PoseEnergyAnalyzer:
    """Tracks pose landmarks and computes per-frame joint-angle velocity (“angular energy”)."""

    _JOINTS = {
        "shoulder": [(11, 13, 23), (12, 14, 24)],  # hip-shoulder-elbow
        "elbow":    [(11, 13, 15), (12, 14, 16)],  # shoulder-elbow-wrist
        "hip":      [(25, 23, 11), (26, 24, 12)],  # knee-hip-shoulder
        "knee":     [(23, 25, 27), (24, 26, 28)],  # hip-knee-ankle
    }

    def __init__(self, smooth_landmarks: bool = True) -> None:
        mp_pose = mp.solutions.pose
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._prev_landmarks: Optional[List[mp.framework.formats.landmark_pb2.NormalizedLandmark]] = None

    def process(self, frame_bgr: np.ndarray):
        """Returns (pose_landmarks, angular_energy). `angular_energy` is None on the very first valid frame."""
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(image_rgb)

        if not results.pose_landmarks:
            return None, None

        curr_landmarks = results.pose_landmarks.landmark
        energy = None
        if self._prev_landmarks is not None:
            energy = self._compute_energy(curr_landmarks, self._prev_landmarks)
        self._prev_landmarks = curr_landmarks
        return results.pose_landmarks, energy

    @staticmethod
    def _angle(p1, p2, p3) -> float:
        a = np.array([p1.x, p1.y]) - np.array([p2.x, p2.y])
        b = np.array([p3.x, p3.y]) - np.array([p2.x, p2.y])
        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    def _joint_angles(self, lm) -> List[float]:
        angles = []
        for triplets in self._JOINTS.values():
            for a, b, c in triplets:
                if (lm[a].visibility < 0.6 or lm[b].visibility < 0.6 or lm[c].visibility < 0.6):
                    angles.append(0.0)
                else:
                    angles.append(self._angle(lm[a], lm[b], lm[c]))
        return angles

    def _compute_energy(self, curr, prev) -> float:
        curr_angles = self._joint_angles(curr)
        prev_angles = self._joint_angles(prev)
        return float(sum(abs(c - p) for c, p in zip(curr_angles, prev_angles))) // 50 # 50 was found through brute forcing


def main(q = None):
    '''Run a test energy calculator'''
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    analyzer = PoseEnergyAnalyzer()
    mp_drawing = mp.solutions.drawing_utils
    mp_pose_spec = mp.solutions.pose.POSE_CONNECTIONS

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks, energy = analyzer.process(frame)

        if landmarks:
            # draw skeleton
            mp_drawing.draw_landmarks(
                frame,
                landmarks,
                mp_pose_spec,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=2),
            )
            # overlay energy
            if energy is not None:
                cv2.putText(
                    frame,
                    f"Energy: {energy:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        cv2.imshow("Pose Angular-Energy (OpenCV)", frame)
        if q:
            self.q.put(energy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
