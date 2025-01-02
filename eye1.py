# import cv2
# import mediapipe as mp
# import numpy as np
# import pyautogui
# import time
# import keyboard
# import logging
# from datetime import datetime
# import os
#
#
# class EyeTracker:
#     def __init__(self):
#         # Setup logging first
#         log_dir = 'logs'
#         if not os.path.exists(log_dir):
#             os.makedirs(log_dir)
#
#         logging.basicConfig(
#             level=logging.INFO,
#             format='%(asctime)s - %(levelname)s - %(message)s',
#             handlers=[
#                 logging.StreamHandler(),
#                 logging.FileHandler(f'logs/eye_tracker_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
#             ]
#         )
#         self.logger = logging.getLogger(__name__)
#
#         # Initialize MediaPipe
#         self.mp_face_mesh = mp.solutions.face_mesh
#         self.face_mesh = self.mp_face_mesh.FaceMesh(
#             max_num_faces=1,
#             refine_landmarks=True,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
#
#         # Initialize camera
#         self.cap = cv2.VideoCapture(0)
#         if not self.cap.isOpened():
#             self.logger.error("Failed to open camera")
#             raise RuntimeError("Failed to open camera")
#
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#
#         # Control variables
#         self.dragging = False
#         self.smoothing_factor = 0.7
#         self.last_x, self.last_y = pyautogui.size()[0] // 2, pyautogui.size()[1] // 2
#         self.blink_start_time = None
#         self.running = True
#         self.face_detected = False
#
#         # Safety settings
#         pyautogui.FAILSAFE = True
#         pyautogui.PAUSE = 0.1
#
#     def move_mouse(self, eye_x, eye_y):
#         screen_width, screen_height = pyautogui.size()
#         target_x = int((eye_x / 640) * screen_width)
#         target_y = int((eye_y / 480) * screen_height)
#
#         new_x = int(self.last_x + (target_x - self.last_x) * self.smoothing_factor)
#         new_y = int(self.last_y + (target_y - self.last_y) * self.smoothing_factor)
#
#         new_x = max(0, min(new_x, screen_width))
#         new_y = max(0, min(new_y, screen_height))
#
#         self.last_x, self.last_y = new_x, new_y
#         pyautogui.moveTo(new_x, new_y)
#
#     def handle_actions(self, face_landmarks, frame_shape):
#         try:
#             # Eye tracking
#             left_eye = np.mean([[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y]
#                                 for i in [362, 385, 387, 263, 373, 380]], axis=0)
#             right_eye = np.mean([[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y]
#                                  for i in [33, 160, 158, 133, 153, 144]], axis=0)
#
#             frame_width, frame_height = frame_shape[:2]
#             left_eye = np.array([left_eye[0] * frame_width, left_eye[1] * frame_height])
#             right_eye = np.array([right_eye[0] * frame_width, right_eye[1] * frame_height])
#
#             # Mouse movement
#             avg_eye_x = (left_eye[0] + right_eye[0]) / 2
#             avg_eye_y = (left_eye[1] + right_eye[1]) / 2
#             self.move_mouse(avg_eye_x, avg_eye_y)
#
#             # Blink detection
#             left_points = np.array([[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y]
#                                     for i in [362, 385, 387, 263, 373, 380]])
#             right_points = np.array([[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y]
#                                      for i in [33, 160, 158, 133, 153, 144]])
#
#             left_blink = np.linalg.norm(left_points[1] - left_points[5]) / np.linalg.norm(
#                 left_points[0] - left_points[3]) < 0.2
#             right_blink = np.linalg.norm(right_points[1] - right_points[5]) / np.linalg.norm(
#                 right_points[0] - right_points[3]) < 0.2
#
#             # Handle clicks
#             if left_blink:
#                 pyautogui.click(button='left')
#             elif right_blink:
#                 pyautogui.click(button='right')
#
#             # Handle scrolling
#             nose_tip = face_landmarks.landmark[1]
#             if nose_tip.y < 0.05:
#                 pyautogui.scroll(10)
#             elif nose_tip.y > 0.95:
#                 pyautogui.scroll(-10)
#
#             # Handle drag and drop
#             if left_blink:
#                 current_time = time.time()
#                 if self.blink_start_time is None:
#                     self.blink_start_time = current_time
#                 elif current_time - self.blink_start_time > 1:
#                     if not self.dragging:
#                         pyautogui.mouseDown()
#                         self.dragging = True
#                     else:
#                         pyautogui.mouseUp()
#                         self.dragging = False
#                     self.blink_start_time = None
#
#         except Exception as e:
#             self.logger.error(f"Error in handle_actions: {str(e)}")
#
#     def run(self):
#         print("Eye tracking started. Press 'q' to quit.")
#         try:
#             while self.running:
#                 if keyboard.is_pressed('q'):
#                     break
#
#                 ret, frame = self.cap.read()
#                 if not ret:
#                     self.logger.error("Failed to capture frame")
#                     break
#
#                 frame = cv2.resize(frame, (640, 480))
#                 rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results = self.face_mesh.process(rgb_frame)
#
#                 if results.multi_face_landmarks:
#                     if not self.face_detected:
#                         self.logger.info("Face detected")
#                         self.face_detected = True
#                     self.handle_actions(results.multi_face_landmarks[0], frame.shape)
#                 elif self.face_detected:
#                     self.logger.info("Face lost")
#                     self.face_detected = False
#
#         except Exception as e:
#             self.logger.error(f"Runtime error: {str(e)}")
#         finally:
#             self.cleanup()
#
#     def cleanup(self):
#         self.logger.info("Shutting down eye tracker")
#         if self.cap is not None:
#             self.cap.release()
#         if self.face_mesh is not None:
#             self.face_mesh.close()
#         print("Eye tracking stopped.")
#
#
# if __name__ == "__main__":
#     tracker = EyeTracker()
#     tracker.run()

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import keyboard
import logging
from datetime import datetime
import os
from collections import deque


class EmotionalEyeTracker:
    def __init__(self):
        self._setup_logging()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.cap = self._setup_camera()

        # Basic controls
        self.dragging = False
        self.smoothing_factor = 0.7
        self.last_x, self.last_y = pyautogui.size()[0] // 2, pyautogui.size()[1] // 2
        self.blink_start_time = None
        self.running = True

        # Emotion tracking
        self.emotion_points = {
            'left_eyebrow': [276, 283, 282, 295, 285],
            'right_eyebrow': [46, 53, 52, 65, 55],
            'mouth_outer': [61, 291, 39, 181, 0, 17],
            'mouth_inner': [78, 308, 324, 402, 317, 14, 87, 178]
        }
        self.emotion_history = deque(maxlen=30)  # Track last 30 frames
        self.current_emotion = 'neutral'
        self.last_expression_time = time.time()

        # Mouse sensitivity adjustments based on emotion
        self.emotion_factors = {
            'happy': 1.2,  # Increased sensitivity
            'sad': 0.8,  # Decreased sensitivity
            'angry': 0.7,  # More controlled
            'surprised': 1.3,  # Quick response
            'neutral': 1.0  # Default
        }

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1

    def _setup_logging(self):
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'logs/emotional_tracker_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.logger.error("Failed to open camera")
            raise RuntimeError("Failed to open camera")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap

    def detect_emotion(self, landmarks):
        try:
            # Calculate emotional features
            left_eyebrow_height = self._get_eyebrow_height(landmarks, self.emotion_points['left_eyebrow'])
            right_eyebrow_height = self._get_eyebrow_height(landmarks, self.emotion_points['right_eyebrow'])
            mouth_ratio = self._get_mouth_ratio(landmarks)

            # Emotion classification
            if left_eyebrow_height > 0.15 and right_eyebrow_height > 0.15:
                if mouth_ratio > 0.6:
                    emotion = 'surprised'
                else:
                    emotion = 'angry'
            elif mouth_ratio > 0.5:
                emotion = 'happy'
            elif mouth_ratio < 0.2:
                emotion = 'sad'
            else:
                emotion = 'neutral'

            self.emotion_history.append(emotion)

            # Smooth emotion detection using majority voting
            if len(self.emotion_history) >= 10:
                emotion_counts = {}
                for e in self.emotion_history:
                    emotion_counts[e] = emotion_counts.get(e, 0) + 1
                self.current_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]

                # Log emotion changes
                if time.time() - self.last_expression_time > 2:  # Prevent too frequent logging
                    self.logger.info(f"Detected emotion: {self.current_emotion}")
                    self.last_expression_time = time.time()

            return self.current_emotion

        except Exception as e:
            self.logger.error(f"Error in emotion detection: {str(e)}")
            return 'neutral'

    def _get_eyebrow_height(self, landmarks, points):
        eyebrow_points = np.array([[landmarks.landmark[i].x, landmarks.landmark[i].y] for i in points])
        return np.max(eyebrow_points[:, 1]) - np.min(eyebrow_points[:, 1])

    def _get_mouth_ratio(self, landmarks):
        outer_points = np.array([[landmarks.landmark[i].x, landmarks.landmark[i].y]
                                 for i in self.emotion_points['mouth_outer']])
        inner_points = np.array([[landmarks.landmark[i].x, landmarks.landmark[i].y]
                                 for i in self.emotion_points['mouth_inner']])

        outer_height = np.max(outer_points[:, 1]) - np.min(outer_points[:, 1])
        outer_width = np.max(outer_points[:, 0]) - np.min(outer_points[:, 0])
        return outer_height / outer_width if outer_width != 0 else 0

    def move_mouse(self, eye_x, eye_y):
        screen_width, screen_height = pyautogui.size()
        emotion_factor = self.emotion_factors.get(self.current_emotion, 1.0)

        target_x = int((eye_x / 640) * screen_width)
        target_y = int((eye_y / 480) * screen_height)

        new_x = int(self.last_x + (target_x - self.last_x) * self.smoothing_factor * emotion_factor)
        new_y = int(self.last_y + (target_y - self.last_y) * self.smoothing_factor * emotion_factor)

        new_x = max(0, min(new_x, screen_width))
        new_y = max(0, min(new_y, screen_height))

        self.last_x, self.last_y = new_x, new_y
        pyautogui.moveTo(new_x, new_y)

    def handle_actions(self, face_landmarks, frame_shape):
        try:
            # Emotion detection
            emotion = self.detect_emotion(face_landmarks)

            # Eye tracking with emotion-adjusted sensitivity
            left_eye = np.mean([[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y]
                                for i in [362, 385, 387, 263, 373, 380]], axis=0)
            right_eye = np.mean([[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y]
                                 for i in [33, 160, 158, 133, 153, 144]], axis=0)

            frame_width, frame_height = frame_shape[:2]
            left_eye = np.array([left_eye[0] * frame_width, left_eye[1] * frame_height])
            right_eye = np.array([right_eye[0] * frame_width, right_eye[1] * frame_height])

            avg_eye_x = (left_eye[0] + right_eye[0]) / 2
            avg_eye_y = (left_eye[1] + right_eye[1]) / 2

            # Adjust mouse movement based on emotion
            self.move_mouse(avg_eye_x, avg_eye_y)

            # Blink detection with emotion consideration
            left_points = np.array([[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y]
                                    for i in [362, 385, 387, 263, 373, 380]])
            right_points = np.array([[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y]
                                     for i in [33, 160, 158, 133, 153, 144]])

            blink_threshold = 0.2 * self.emotion_factors.get(emotion, 1.0)
            left_blink = np.linalg.norm(left_points[1] - left_points[5]) / np.linalg.norm(
                left_points[0] - left_points[3]) < blink_threshold
            right_blink = np.linalg.norm(right_points[1] - right_points[5]) / np.linalg.norm(
                right_points[0] - right_points[3]) < blink_threshold

            # Adjust click behavior based on emotion
            if emotion == 'angry':
                # More controlled clicking for angry state
                if left_blink and time.time() - self.last_expression_time > 1:
                    pyautogui.click(button='left')
                    self.last_expression_time = time.time()
            else:
                if left_blink:
                    pyautogui.click(button='left')
                elif right_blink:
                    pyautogui.click(button='right')

            # Scrolling with emotion-based speed
            nose_tip = face_landmarks.landmark[1]
            scroll_speed = int(10 * self.emotion_factors.get(emotion, 1.0))
            if nose_tip.y < 0.05:
                pyautogui.scroll(scroll_speed)
            elif nose_tip.y > 0.95:
                pyautogui.scroll(-scroll_speed)

            # Drag and drop with emotion consideration
            if left_blink:
                current_time = time.time()
                if self.blink_start_time is None:
                    self.blink_start_time = current_time
                elif current_time - self.blink_start_time > (0.5 if emotion == 'surprised' else 1):
                    if not self.dragging:
                        pyautogui.mouseDown()
                        self.dragging = True
                    else:
                        pyautogui.mouseUp()
                        self.dragging = False
                    self.blink_start_time = None

        except Exception as e:
            self.logger.error(f"Error in handle_actions: {str(e)}")

    def run(self):
        print("Emotional eye tracking started. Press 'q' to quit.")
        try:
            while self.running:
                if keyboard.is_pressed('q'):
                    break

                ret, frame = self.cap.read()
                if not ret:
                    self.logger.error("Failed to capture frame")
                    break

                frame = cv2.resize(frame, (640, 480))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    self.handle_actions(results.multi_face_landmarks[0], frame.shape)

        except Exception as e:
            self.logger.error(f"Runtime error: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        self.logger.info("Shutting down emotional eye tracker")
        if self.cap is not None:
            self.cap.release()
        if self.face_mesh is not None:
            self.face_mesh.close()
        print("Eye tracking stopped.")


if __name__ == "__main__":
    tracker = EmotionalEyeTracker()
    tracker.run()