# import cv2
# import mediapipe as mp
# import numpy as np
# import pyautogui
# import tkinter as tk
# from tkinter import ttk
# from PIL import Image, ImageTk
# import threading
# import time

# class EyeTrackingApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Eye Tracking Mouse Control")
        
#         # Initialize MediaPipe Face Mesh
#         self.mp_face_mesh = mp.solutions.face_mesh
#         self.face_mesh = self.mp_face_mesh.FaceMesh(
#             max_num_faces=1,
#             refine_landmarks=True,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
        
#         # Initialize video capture
#         self.cap = cv2.VideoCapture(0)
        
#         # Control variables
#         self.tracking_active = False
#         self.smoothing_factor = 0.5
#         self.last_x, self.last_y = pyautogui.size()[0] // 2, pyautogui.size()[1] // 2
        
#         # Eye landmarks indices
#         self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
#         self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
#         self.setup_gui()
        
#     def setup_gui(self):
#         # Create main frame
#         main_frame = ttk.Frame(self.root, padding="10")
#         main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
#         # Video feed frame
#         self.video_label = ttk.Label(main_frame)
#         self.video_label.grid(row=0, column=0, columnspan=2, pady=5)
        
#         # Control buttons
#         ttk.Button(main_frame, text="Start Tracking", 
#                   command=self.start_tracking).grid(row=1, column=0, pady=5)
#         ttk.Button(main_frame, text="Stop Tracking", 
#                   command=self.stop_tracking).grid(row=1, column=1, pady=5)
        
#         # Smoothing control
#         ttk.Label(main_frame, text="Smoothing:").grid(row=2, column=0, pady=5)
#         self.smoothing_scale = ttk.Scale(main_frame, from_=0.1, to=0.9, 
#                                        orient=tk.HORIZONTAL, value=0.5,
#                                        command=self.update_smoothing)
#         self.smoothing_scale.grid(row=2, column=1, pady=5)
        
#         # Status label
#         self.status_label = ttk.Label(main_frame, text="Status: Not tracking")
#         self.status_label.grid(row=3, column=0, columnspan=2, pady=5)
        
#         # Calibration button
#         ttk.Button(main_frame, text="Calibrate", 
#                   command=self.calibrate).grid(row=4, column=0, columnspan=2, pady=5)
        
#         # Start video feed
#         self.update_video()
        
#     def update_smoothing(self, value):
#         self.smoothing_factor = float(value)
        
#     def get_eye_center(self, landmarks, eye_indices):
#         eye_points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
#         return np.mean(eye_points, axis=0)
        
#     def process_eye_tracking(self, frame, face_landmarks):
#         frame_height, frame_width = frame.shape[:2]
        
#         # Get eye centers
#         left_center = self.get_eye_center(face_landmarks.landmark, self.LEFT_EYE)
#         right_center = self.get_eye_center(face_landmarks.landmark, self.RIGHT_EYE)
        
#         # Convert relative coordinates to pixel coordinates
#         left_center = np.array([left_center[0] * frame_width, left_center[1] * frame_height])
#         right_center = np.array([right_center[0] * frame_width, right_center[1] * frame_height])
        
#         # Draw circles at eye centers for visualization
#         cv2.circle(frame, tuple(left_center.astype(int)), 3, (0, 255, 0), -1)
#         cv2.circle(frame, tuple(right_center.astype(int)), 3, (0, 255, 0), -1)
        
#         # Calculate average eye position
#         avg_x = (left_center[0] + right_center[0]) / 2
#         avg_y = (left_center[1] + right_center[1]) / 2
        
#         return avg_x, avg_y
        
#     def move_mouse(self, eye_x, eye_y):
#         # Get screen dimensions
#         screen_width, screen_height = pyautogui.size()
        
#         # Convert eye coordinates to screen coordinates with smoothing
#         target_x = int((eye_x / 640) * screen_width)
#         target_y = int((eye_y / 480) * screen_height)
        
#         # Apply smoothing
#         new_x = int(self.last_x + (target_x - self.last_x) * self.smoothing_factor)
#         new_y = int(self.last_y + (target_y - self.last_y) * self.smoothing_factor)
        
#         # Ensure coordinates are within screen bounds
#         new_x = max(0, min(new_x, screen_width))
#         new_y = max(0, min(new_y, screen_height))
        
#         # Update last position
#         self.last_x, self.last_y = new_x, new_y
        
#         # Move mouse
#         pyautogui.moveTo(new_x, new_y)
        
#     def calibrate(self):
#         self.status_label.config(text="Status: Calibrating...")
#         # Reset mouse to center of screen
#         screen_width, screen_height = pyautogui.size()
#         self.last_x, self.last_y = screen_width // 2, screen_height // 2
#         pyautogui.moveTo(self.last_x, self.last_y)
#         time.sleep(1)
#         self.status_label.config(text="Status: Calibration complete")
        
#     def update_video(self):
#         ret, frame = self.cap.read()
#         if ret:
#             # Convert frame to RGB for MediaPipe
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
#             if self.tracking_active:
#                 # Process frame with MediaPipe
#                 results = self.face_mesh.process(rgb_frame)
                
#                 if results.multi_face_landmarks:
#                     face_landmarks = results.multi_face_landmarks[0]
                    
#                     # Process eye tracking
#                     eye_x, eye_y = self.process_eye_tracking(rgb_frame, face_landmarks)
                    
#                     # Move mouse in separate thread to prevent GUI freezing
#                     threading.Thread(target=self.move_mouse, 
#                                   args=(eye_x, eye_y)).start()
            
#             # Convert to PhotoImage for tkinter
#             img = Image.fromarray(rgb_frame)
#             img = ImageTk.PhotoImage(image=img)
            
#             # Update video feed
#             self.video_label.imgtk = img
#             self.video_label.configure(image=img)
        
#         # Schedule next update
#         self.root.after(10, self.update_video)
        
#     def start_tracking(self):
#         self.tracking_active = True
#         self.status_label.config(text="Status: Tracking active")
        
#     def stop_tracking(self):
#         self.tracking_active = False
#         self.status_label.config(text="Status: Not tracking")
        
#     def cleanup(self):
#         self.cap.release()
#         self.face_mesh.close()
#         self.root.destroy()

# if __name__ == "__main__":
#     # Create main window
#     root = tk.Tk()
#     app = EyeTrackingApp(root)
    
#     # Set cleanup on window close
#     root.protocol("WM_DELETE_WINDOW", app.cleanup)
    
#     # Start application
#     root.mainloop()

# import cv2
# import mediapipe as mp
# import numpy as np
# import pyautogui
# import tkinter as tk
# from tkinter import ttk
# from PIL import Image, ImageTk
# import threading
# import time

# class EyeTrackingApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Enhanced Eye Tracking Mouse Control")
        
#         # Initialize MediaPipe Face Mesh
#         self.mp_face_mesh = mp.solutions.face_mesh
#         self.face_mesh = self.mp_face_mesh.FaceMesh(
#             max_num_faces=1,
#             refine_landmarks=True,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
        
#         # Initialize video capture
#         self.cap = cv2.VideoCapture(0)
        
#         # Control variables
#         self.tracking_active = False
#         self.smoothing_factor = 0.5
#         self.last_x, self.last_y = pyautogui.size()[0] // 2, pyautogui.size()[1] // 2
        
#         # Eye landmarks indices
#         self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
#         self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
#         # Blink thresholds
#         self.blink_threshold = 0.2
#         self.left_blink = False
#         self.right_blink = False
        
#         self.setup_gui()
        
#     def setup_gui(self):
#         # Create main frame
#         main_frame = ttk.Frame(self.root, padding="10")
#         main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
#         # Video feed frame
#         self.video_label = ttk.Label(main_frame)
#         self.video_label.grid(row=0, column=0, columnspan=2, pady=5)
        
#         # Control buttons
#         ttk.Button(main_frame, text="Start Tracking", 
#                   command=self.start_tracking).grid(row=1, column=0, pady=5)
#         ttk.Button(main_frame, text="Stop Tracking", 
#                   command=self.stop_tracking).grid(row=1, column=1, pady=5)
        
#         # Smoothing control
#         ttk.Label(main_frame, text="Smoothing:").grid(row=2, column=0, pady=5)
#         self.smoothing_scale = ttk.Scale(main_frame, from_=0.1, to=0.9, 
#                                        orient=tk.HORIZONTAL, value=0.5,
#                                        command=self.update_smoothing)
#         self.smoothing_scale.grid(row=2, column=1, pady=5)
        
#         # Status label
#         self.status_label = ttk.Label(main_frame, text="Status: Not tracking")
#         self.status_label.grid(row=3, column=0, columnspan=2, pady=5)
        
#         # Calibration button
#         ttk.Button(main_frame, text="Calibrate", 
#                   command=self.calibrate).grid(row=4, column=0, columnspan=2, pady=5)
        
#         # Start video feed
#         self.update_video()
        
#     def update_smoothing(self, value):
#         self.smoothing_factor = float(value)
        
#     def get_eye_center(self, landmarks, eye_indices):
#         eye_points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
#         return np.mean(eye_points, axis=0)
        
#     def detect_blink(self, landmarks, eye_indices):
#         eye_points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
#         vertical_dist = np.linalg.norm(eye_points[1] - eye_points[5])
#         horizontal_dist = np.linalg.norm(eye_points[0] - eye_points[3])
#         blink_ratio = vertical_dist / horizontal_dist
#         return blink_ratio < self.blink_threshold
        
#     def process_eye_tracking(self, frame, face_landmarks):
#         frame_height, frame_width = frame.shape[:2]
        
#         # Get eye centers
#         left_center = self.get_eye_center(face_landmarks.landmark, self.LEFT_EYE)
#         right_center = self.get_eye_center(face_landmarks.landmark, self.RIGHT_EYE)
        
#         # Convert relative coordinates to pixel coordinates
#         left_center = np.array([left_center[0] * frame_width, left_center[1] * frame_height])
#         right_center = np.array([right_center[0] * frame_width, right_center[1] * frame_height])
        
#         # Detect blinks
#         self.left_blink = self.detect_blink(face_landmarks.landmark, self.LEFT_EYE)
#         self.right_blink = self.detect_blink(face_landmarks.landmark, self.RIGHT_EYE)
        
#         # Draw circles at eye centers for visualization
#         cv2.circle(frame, tuple(left_center.astype(int)), 3, (0, 255, 0), -1)
#         cv2.circle(frame, tuple(right_center.astype(int)), 3, (0, 255, 0), -1)
        
#         # Calculate average eye position
#         avg_x = (left_center[0] + right_center[0]) / 2
#         avg_y = (left_center[1] + right_center[1]) / 2
        
#         return avg_x, avg_y
        
#     def perform_gesture_action(self):
#         if self.left_blink and self.right_blink:
#             pyautogui.rightClick()
#         elif self.left_blink:
#             pyautogui.click()
#         elif self.right_blink:
#             pyautogui.scroll(100)  # Scroll up
        
#     def move_mouse(self, eye_x, eye_y):
#         # Get screen dimensions
#         screen_width, screen_height = pyautogui.size()
        
#         # Convert eye coordinates to screen coordinates with smoothing
#         target_x = int((eye_x / 640) * screen_width)
#         target_y = int((eye_y / 480) * screen_height)
        
#         # Apply smoothing
#         new_x = int(self.last_x + (target_x - self.last_x) * self.smoothing_factor)
#         new_y = int(self.last_y + (target_y - self.last_y) * self.smoothing_factor)
        
#         # Ensure coordinates are within screen bounds
#         new_x = max(0, min(new_x, screen_width))
#         new_y = max(0, min(new_y, screen_height))
        
#         # Update last position
#         self.last_x, self.last_y = new_x, new_y
        
#         # Move mouse
#         pyautogui.moveTo(new_x, new_y)
        
#     def calibrate(self):
#         self.status_label.config(text="Status: Calibrating...")
#         screen_width, screen_height = pyautogui.size()
#         self.last_x, self.last_y = screen_width // 2, screen_height // 2
#         pyautogui.moveTo(self.last_x, self.last_y)
#         time.sleep(1)
#         self.status_label.config(text="Status: Calibration complete")
        
#     def update_video(self):
#         ret, frame = self.cap.read()
#         if ret:
#             # Resize frame for faster processing
#             frame = cv2.resize(frame, (640, 480))
            
#             # Convert frame to RGB for MediaPipe
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
#             if self.tracking_active:
#                 # Process frame with MediaPipe
#                 results = self.face_mesh.process(rgb_frame)
                
#                 if results.multi_face_landmarks:
#                     face_landmarks = results.multi_face_landmarks[0]
                    
#                     # Process eye tracking
#                     eye_x, eye_y = self.process_eye_tracking(rgb_frame, face_landmarks)
                    
#                     # Move mouse in separate thread to prevent GUI freezing
#                     threading.Thread(target=self.move_mouse, 
#                                   args=(eye_x, eye_y)).start()
                    
#                     # Perform gesture-based actions
#                     self.perform_gesture_action()
            
#             # Convert to PhotoImage for tkinter
#             img = Image.fromarray(rgb_frame)
#             img = ImageTk.PhotoImage(image=img)
            
#             # Update video feed
#             self.video_label.imgtk = img
#             self.video_label.configure(image=img)
        
#         # Schedule next update
#         self.root.after(10, self.update_video)
        
#     def start_tracking(self):
#         self.tracking_active = True
#         self.status_label.config(text="Status: Tracking active")
        
#     def stop_tracking(self):
#         self.tracking_active = False
#         self.status_label.config(text="Status: Not tracking")
        
#     def cleanup(self):
#         self.cap.release()
#         self.face_mesh.close()
#         self.root.destroy()

# if __name__ == "__main__":
#     # Create main window
#     root = tk.Tk()
#     app = EyeTrackingApp(root)
    
#     # Set cleanup on window close
#     root.protocol("WM_DELETE_WINDOW", app.cleanup)
    
#     # Start application
#     root.mainloop()


import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time

class AdvancedEyeTrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Eye Tracking Mouse Control")

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Video capture
        self.cap = cv2.VideoCapture(0)

        # Control variables
        self.tracking_active = False
        self.dragging = False  # Dragging state
        self.smoothing_factor = 0.7
        self.last_x, self.last_y = pyautogui.size()[0] // 2, pyautogui.size()[1] // 2
        self.blink_start_time = None

        # Eye and head pose parameters
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.NOSE_TIP = 1
        self.blink_threshold = 0.2
        self.scroll_threshold = 0.05  # Threshold for scroll actions

        self.setup_gui()

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Video feed frame
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=0, columnspan=2, pady=5)

        # Control buttons
        ttk.Button(main_frame, text="Start Tracking", command=self.start_tracking).grid(row=1, column=0, pady=5)
        ttk.Button(main_frame, text="Stop Tracking", command=self.stop_tracking).grid(row=1, column=1, pady=5)

        # Status label
        self.status_label = ttk.Label(main_frame, text="Status: Not tracking")
        self.status_label.grid(row=2, column=0, columnspan=2, pady=5)

        # Start video feed
        self.update_video()

    def process_eye_tracking(self, frame, face_landmarks):
        frame_height, frame_width = frame.shape[:2]

        # Eye coordinates
        left_eye = self.get_eye_center(face_landmarks.landmark, self.LEFT_EYE)
        right_eye = self.get_eye_center(face_landmarks.landmark, self.RIGHT_EYE)

        # Convert to pixel coordinates
        left_eye = np.array([left_eye[0] * frame_width, left_eye[1] * frame_height])
        right_eye = np.array([right_eye[0] * frame_width, right_eye[1] * frame_height])

        # Detect blink
        left_blink = self.detect_blink(face_landmarks.landmark, self.LEFT_EYE)
        right_blink = self.detect_blink(face_landmarks.landmark, self.RIGHT_EYE)

        # Detect head movement for scrolling
        nose_tip = face_landmarks.landmark[self.NOSE_TIP]
        scroll_up = nose_tip.y < self.scroll_threshold
        scroll_down = nose_tip.y > (1 - self.scroll_threshold)

        return (left_eye, right_eye), (left_blink, right_blink), (scroll_up, scroll_down)

    def detect_blink(self, landmarks, eye_indices):
        eye_points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
        vertical_dist = np.linalg.norm(eye_points[1] - eye_points[5])
        horizontal_dist = np.linalg.norm(eye_points[0] - eye_points[3])
        blink_ratio = vertical_dist / horizontal_dist
        return blink_ratio < self.blink_threshold

    def get_eye_center(self, landmarks, eye_indices):
        eye_points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
        return np.mean(eye_points, axis=0)

    def move_mouse(self, eye_x, eye_y):
        screen_width, screen_height = pyautogui.size()

        # Smooth mouse movement
        target_x = int((eye_x / 640) * screen_width)
        target_y = int((eye_y / 480) * screen_height)

        # Exponential moving average for smoothing
        new_x = int(self.last_x + (target_x - self.last_x) * self.smoothing_factor)
        new_y = int(self.last_y + (target_y - self.last_y) * self.smoothing_factor)

        self.last_x, self.last_y = new_x, new_y
        pyautogui.moveTo(new_x, new_y)

    def handle_clicks_and_scroll(self, left_blink, right_blink, scroll_up, scroll_down):
        # Handle blinks for clicking
        if left_blink:
            pyautogui.click(button='left')
        elif right_blink:
            pyautogui.click(button='right')

        # Handle scrolling
        if scroll_up:
            pyautogui.scroll(10)
        elif scroll_down:
            pyautogui.scroll(-10)

        # Handle drag and drop
        if left_blink and self.blink_start_time is None:
            self.blink_start_time = time.time()
        elif left_blink and self.blink_start_time and (time.time() - self.blink_start_time > 1):
            if not self.dragging:
                pyautogui.mouseDown()
                self.dragging = True
            else:
                pyautogui.mouseUp()
                self.dragging = False
            self.blink_start_time = None

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.tracking_active:
                results = self.face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    (left_eye, right_eye), (left_blink, right_blink), (scroll_up, scroll_down) = self.process_eye_tracking(
                        frame, face_landmarks)

                    avg_eye_x = (left_eye[0] + right_eye[0]) / 2
                    avg_eye_y = (left_eye[1] + right_eye[1]) / 2

                    threading.Thread(target=self.move_mouse, args=(avg_eye_x, avg_eye_y)).start()

                    # Handle clicks and scrolling
                    self.handle_clicks_and_scroll(left_blink, right_blink, scroll_up, scroll_down)

            img = Image.fromarray(rgb_frame)
            img = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = img
            self.video_label.configure(image=img)

        self.root.after(10, self.update_video)

    def start_tracking(self):
        self.tracking_active = True
        self.status_label.config(text="Status: Tracking active")

    def stop_tracking(self):
        self.tracking_active = False
        self.status_label.config(text="Status: Not tracking")

    def cleanup(self):
        self.cap.release()
        self.face_mesh.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedEyeTrackingApp(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()
