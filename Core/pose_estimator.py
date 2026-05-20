import cv2
import mediapipe as mp
import numpy as np
 
from Config import (
    MEDIAPIPE_MODEL_COMPLEXITY,
    MEDIAPIPE_MIN_DETECTION_CONF,
    MEDIAPIPE_MIN_TRACKING_CONF,
)

# Tên 33 landmark theo thứ tự MediaPipe
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Index các landmark hay dùng — tránh magic number trong code khác
class LM:
    LEFT_SHOULDER   = 11
    RIGHT_SHOULDER  = 12
    LEFT_HIP        = 23
    RIGHT_HIP       = 24
    LEFT_KNEE       = 25
    RIGHT_KNEE      = 26
    LEFT_ANKLE      = 27
    RIGHT_ANKLE     = 28
    LEFT_HEEL       = 29
    RIGHT_HEEL      = 30

class PoseEstimator:
    def __init__(self, static_image_mode: bool = True):
        """
        static_image_mode=True  → dùng khi xử lý ảnh tĩnh / batch frames
        static_image_mode=False → dùng khi inference realtime từ camera
        """
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONF,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONF
        )
        
    def extract(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """
        Trả về array shape (33, 4): [x, y, z, visibility] mỗi landmark.
        Tọa độ x, y đã chuẩn hóa [0, 1] theo kích thước frame.
        Trả về None nếu không detect được người.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._pose.process(frame_rgb)
        frame_rgb.flags.writeable = True
        
        if not results.pose_landmarks:
            return None
        
        return np.array(
            [[lm.x, lm.y, lm.z, lm.visibility]
             for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        )
        
    def extract_with_drawing(
        self, frame_bgr: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """
        Giống extract nhưng trả thêm skeleton lên hình
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._pose.process(frame_rgb)
        frame_rgb.flags.writeable = True
        
        if not results.pose_landmarks:
            return None
        
        annotated = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        mp.solutions.drawing_utils.draw_landmarks(
            annotated,
            results.pose_landmarks,
            self._mp_pose.POSE_CONNECTIONS,
        )
        
        keypoints = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility]
             for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        )
        return keypoints, annotated
        
        
        
        
        
