import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time

class EyeTrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye Tracking Mouse Control")
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        # Control variables
        self.tracking_active = False
        self.smoothing_factor = 0.5
        self.last_x, self.last_y = pyautogui.size()[0] // 2, pyautogui.size()[1] // 2
        
        # Eye landmarks indices
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        self.setup_gui()
        
    def setup_gui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video feed frame
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=0, columnspan=2, pady=5)
        
        # Control buttons
        ttk.Button(main_frame, text="Start Tracking", 
                  command=self.start_tracking).grid(row=1, column=0, pady=5)
        ttk.Button(main_frame, text="Stop Tracking", 
                  command=self.stop_tracking).grid(row=1, column=1, pady=5)
        
        # Smoothing control
        ttk.Label(main_frame, text="Smoothing:").grid(row=2, column=0, pady=5)
        self.smoothing_scale = ttk.Scale(main_frame, from_=0.1, to=0.9, 
                                       orient=tk.HORIZONTAL, value=0.5,
                                       command=self.update_smoothing)
        self.smoothing_scale.grid(row=2, column=1, pady=5)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Status: Not tracking")
        self.status_label.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Calibration button
        ttk.Button(main_frame, text="Calibrate", 
                  command=self.calibrate).grid(row=4, column=0, columnspan=2, pady=5)
        
        # Start video feed
        self.update_video()
        
    def update_smoothing(self, value):
        self.smoothing_factor = float(value)
        
    def get_eye_center(self, landmarks, eye_indices):
        eye_points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
        return np.mean(eye_points, axis=0)
        
    def process_eye_tracking(self, frame, face_landmarks):
        frame_height, frame_width = frame.shape[:2]
        
        # Get eye centers
        left_center = self.get_eye_center(face_landmarks.landmark, self.LEFT_EYE)
        right_center = self.get_eye_center(face_landmarks.landmark, self.RIGHT_EYE)
        
        # Convert relative coordinates to pixel coordinates
        left_center = np.array([left_center[0] * frame_width, left_center[1] * frame_height])
        right_center = np.array([right_center[0] * frame_width, right_center[1] * frame_height])
        
        # Draw circles at eye centers for visualization
        cv2.circle(frame, tuple(left_center.astype(int)), 3, (0, 255, 0), -1)
        cv2.circle(frame, tuple(right_center.astype(int)), 3, (0, 255, 0), -1)
        
        # Calculate average eye position
        avg_x = (left_center[0] + right_center[0]) / 2
        avg_y = (left_center[1] + right_center[1]) / 2
        
        return avg_x, avg_y
        
    def move_mouse(self, eye_x, eye_y):
        # Get screen dimensions
        screen_width, screen_height = pyautogui.size()
        
        # Convert eye coordinates to screen coordinates with smoothing
        target_x = int((eye_x / 640) * screen_width)
        target_y = int((eye_y / 480) * screen_height)
        
        # Apply smoothing
        new_x = int(self.last_x + (target_x - self.last_x) * self.smoothing_factor)
        new_y = int(self.last_y + (target_y - self.last_y) * self.smoothing_factor)
        
        # Ensure coordinates are within screen bounds
        new_x = max(0, min(new_x, screen_width))
        new_y = max(0, min(new_y, screen_height))
        
        # Update last position
        self.last_x, self.last_y = new_x, new_y
        
        # Move mouse
        pyautogui.moveTo(new_x, new_y)
        
    def calibrate(self):
        self.status_label.config(text="Status: Calibrating...")
        # Reset mouse to center of screen
        screen_width, screen_height = pyautogui.size()
        self.last_x, self.last_y = screen_width // 2, screen_height // 2
        pyautogui.moveTo(self.last_x, self.last_y)
        time.sleep(1)
        self.status_label.config(text="Status: Calibration complete")
        
    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.tracking_active:
                # Process frame with MediaPipe
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    # Process eye tracking
                    eye_x, eye_y = self.process_eye_tracking(rgb_frame, face_landmarks)
                    
                    # Move mouse in separate thread to prevent GUI freezing
                    threading.Thread(target=self.move_mouse, 
                                  args=(eye_x, eye_y)).start()
            
            # Convert to PhotoImage for tkinter
            img = Image.fromarray(rgb_frame)
            img = ImageTk.PhotoImage(image=img)
            
            # Update video feed
            self.video_label.imgtk = img
            self.video_label.configure(image=img)
        
        # Schedule next update
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
    # Create main window
    root = tk.Tk()
    app = EyeTrackingApp(root)
    
    # Set cleanup on window close
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    
    # Start application
    root.mainloop()

# import cv2
# import mediapipe as mp
# import numpy as np
# import pyautogui
# import tkinter as tk
# from tkinter import ttk
# from PIL import Image, ImageTk
# import threading
# import time
# from datetime import datetime
# import json
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import seaborn as sns

# class EyeTrackingApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Advanced Eye Tracking Mouse Control")
        
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
        
#         # Blink detection variables
#         self.blink_threshold = 0.2
#         self.last_blink_time = time.time()
#         self.blink_cooldown = 0.5  # Seconds between blinks
#         self.click_on_blink = False
        
#         # Calibration variables
#         self.calibration_points = []
#         self.calibrating = False
#         self.calibration_index = 0
        
#         # Statistics variables
#         self.gaze_points = []
#         self.blink_count = 0
#         self.start_time = None
#         self.heatmap_data = []
        
#         # Eye landmarks indices
#         self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
#         self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
#         # Settings
#         self.settings = {
#             'smoothing_factor': 0.5,
#             'blink_threshold': 0.2,
#             'blink_cooldown': 0.5,
#             'click_on_blink': False
#         }
        
#         self.setup_gui()
#         self.load_settings()
        
#     def setup_gui(self):
#         # Create notebook for tabs
#         self.notebook = ttk.Notebook(self.root)
#         self.notebook.pack(expand=True, fill='both', padx=5, pady=5)
        
#         # Main tab
#         self.main_tab = ttk.Frame(self.notebook)
#         self.notebook.add(self.main_tab, text='Main')
        
#         # Statistics tab
#         self.stats_tab = ttk.Frame(self.notebook)
#         self.notebook.add(self.stats_tab, text='Statistics')
        
#         # Settings tab
#         self.settings_tab = ttk.Frame(self.notebook)
#         self.notebook.add(self.settings_tab, text='Settings')
        
#         self.setup_main_tab()
#         self.setup_stats_tab()
#         self.setup_settings_tab()
        
#     def setup_main_tab(self):
#         # Video feed frame
#         self.video_label = ttk.Label(self.main_tab)
#         self.video_label.grid(row=0, column=0, columnspan=2, pady=5)
        
#         # Control buttons frame
#         control_frame = ttk.LabelFrame(self.main_tab, text="Controls")
#         control_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=5)
        
#         ttk.Button(control_frame, text="Start Tracking", 
#                   command=self.start_tracking).grid(row=0, column=0, padx=5, pady=5)
#         ttk.Button(control_frame, text="Stop Tracking", 
#                   command=self.stop_tracking).grid(row=0, column=1, padx=5, pady=5)
#         ttk.Button(control_frame, text="Start Calibration", 
#                   command=self.start_calibration).grid(row=0, column=2, padx=5, pady=5)
        
#         # Status frame
#         status_frame = ttk.LabelFrame(self.main_tab, text="Status")
#         status_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=5)
        
#         self.status_label = ttk.Label(status_frame, text="Status: Not tracking")
#         self.status_label.grid(row=0, column=0, pady=5)
        
#         self.blink_label = ttk.Label(status_frame, text="Blinks: 0")
#         self.blink_label.grid(row=0, column=1, pady=5)
        
#     def setup_stats_tab(self):
#         # Create matplotlib figure for heatmap
#         self.fig, self.ax = plt.subplots(figsize=(6, 4))
#         self.canvas = FigureCanvasTkAgg(self.fig, master=self.stats_tab)
#         self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
#         # Statistics frame
#         stats_frame = ttk.LabelFrame(self.stats_tab, text="Session Statistics")
#         stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
#         self.stats_text = tk.Text(stats_frame, height=5, width=40)
#         self.stats_text.pack(padx=5, pady=5)
        
#         # Export button
#         ttk.Button(self.stats_tab, text="Export Statistics", 
#                   command=self.export_statistics).pack(pady=5)
        
#     def setup_settings_tab(self):
#         # Smoothing control
#         smooth_frame = ttk.LabelFrame(self.settings_tab, text="Mouse Smoothing")
#         smooth_frame.pack(fill=tk.X, padx=5, pady=5)
        
#         self.smoothing_scale = ttk.Scale(smooth_frame, from_=0.1, to=0.9, 
#                                        orient=tk.HORIZONTAL, value=self.smoothing_factor,
#                                        command=self.update_smoothing)
#         self.smoothing_scale.pack(padx=5, pady=5)
        
#         # Blink controls
#         blink_frame = ttk.LabelFrame(self.settings_tab, text="Blink Controls")
#         blink_frame.pack(fill=tk.X, padx=5, pady=5)
        
#         self.click_blink_var = tk.BooleanVar(value=self.click_on_blink)
#         ttk.Checkbutton(blink_frame, text="Click on Blink", 
#                        variable=self.click_blink_var,
#                        command=self.toggle_blink_click).pack(padx=5, pady=5)
        
#         ttk.Label(blink_frame, text="Blink Threshold:").pack()
#         self.blink_threshold_scale = ttk.Scale(blink_frame, from_=0.1, to=0.5, 
#                                              orient=tk.HORIZONTAL, value=self.blink_threshold,
#                                              command=self.update_blink_threshold)
#         self.blink_threshold_scale.pack(padx=5, pady=5)
        
#         # Save settings button
#         ttk.Button(self.settings_tab, text="Save Settings", 
#                   command=self.save_settings).pack(pady=5)
        
#     def update_smoothing(self, value):
#         self.smoothing_factor = float(value)
#         self.settings['smoothing_factor'] = self.smoothing_factor
        
#     def update_blink_threshold(self, value):
#         self.blink_threshold = float(value)
#         self.settings['blink_threshold'] = self.blink_threshold
        
#     def toggle_blink_click(self):
#         self.click_on_blink = self.click_blink_var.get()
#         self.settings['click_on_blink'] = self.click_on_blink
        
#     def save_settings(self):
#         with open('eye_tracker_settings.json', 'w') as f:
#             json.dump(self.settings, f)
            
#     def load_settings(self):
#         try:
#             with open('eye_tracker_settings.json', 'r') as f:
#                 loaded_settings = json.load(f)
#                 self.settings.update(loaded_settings)
                
#                 # Apply loaded settings
#                 self.smoothing_factor = self.settings['smoothing_factor']
#                 self.blink_threshold = self.settings['blink_threshold']
#                 self.click_on_blink = self.settings['click_on_blink']
                
#                 # Update GUI elements
#                 self.smoothing_scale.set(self.smoothing_factor)
#                 self.blink_threshold_scale.set(self.blink_threshold)
#                 self.click_blink_var.set(self.click_on_blink)
#         except FileNotFoundError:
#             pass
            
#     def calculate_eye_aspect_ratio(self, eye_points):
#         """Calculate the eye aspect ratio to detect blinks"""
#         vertical_dist1 = np.linalg.norm(eye_points[1] - eye_points[5])
#         vertical_dist2 = np.linalg.norm(eye_points[2] - eye_points[4])
#         horizontal_dist = np.linalg.norm(eye_points[0] - eye_points[3])
        
#         ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
#         return ear
        
#     def detect_blink(self, left_eye_points, right_eye_points):
#         left_ear = self.calculate_eye_aspect_ratio(left_eye_points)
#         right_ear = self.calculate_eye_aspect_ratio(right_eye_points)
        
#         avg_ear = (left_ear + right_ear) / 2.0
        
#         current_time = time.time()
#         if avg_ear < self.blink_threshold and (current_time - self.last_blink_time) > self.blink_cooldown:
#             self.last_blink_time = current_time
#             self.blink_count += 1
#             self.blink_label.config(text=f"Blinks: {self.blink_count}")
            
#             if self.click_on_blink:
#                 pyautogui.click()
            
#             return True
#         return False
        
#     def start_calibration(self):
#         self.calibrating = True
#         self.calibration_points = [
#             (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
#             (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
#             (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
#         ]
#         self.calibration_index = 0
#         self.status_label.config(text="Status: Calibrating - Look at the red dot")
        
#     def draw_calibration_point(self, frame):
#         if self.calibration_index < len(self.calibration_points):
#             h, w = frame.shape[:2]
#             x = int(w * self.calibration_points[self.calibration_index][0])
#             y = int(h * self.calibration_points[self.calibration_index][1])
#             cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
            
#             if time.time() - self.last_blink_time > 2.0:  # Move to next point after 2 seconds
#                 self.calibration_index += 1
#                 if self.calibration_index >= len(self.calibration_points):
#                     self.calibrating = False
#                     self.status_label.config(text="Status: Calibration complete")
#                 self.last_blink_time = time.time()
        
#     def update_heatmap(self, x, y):
#         screen_width, screen_height = pyautogui.size()
#         normalized_x = x / screen_width
#         normalized_y = y / screen_height
#         self.heatmap_data.append([normalized_x, normalized_y])
        
#         if len(self.heatmap_data) > 100:  # Update heatmap every 100 points
#             data = np.array(self.heatmap_data)
#             self.ax.clear()
#             sns.kdeplot(data=data, x=data[:, 0], y=data[:, 1], 
#                        cmap="YlOrRd", fill=True, ax=self.ax)
#             self.ax.set_title("Gaze Heatmap")
#             self.canvas.draw()
#             self.heatmap_data = []
        
#     def export_statistics(self):
#         if self.start_time:
#             duration = time.time() - self.start_time
#             stats = {
#                 'session_duration': duration,
#                 'total_blinks': self.blink_count,
#                 'blinks_per_minute': (self.blink_count / duration) * 60,
#                 'gaze_points': self.gaze_points,
#                 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#             }
            
#             with open(f'eye_tracking_stats_{int(time.time())}.json', 'w') as f:
#                 json.dump(stats, f)
            
#             self.status_label.config(text="Status: Statistics exported")
        
#     def process_eye_tracking(self, frame, face_landmarks):
#         frame_height, frame_width = frame.shape[:2]
        
#         # Get eye landmarks
#         left_eye_points = np.array([[face_landmarks.landmark[i].x * frame_width,
#                                    face_landmarks.landmark[i].y * frame_height] 
#                                   for i in self.LEFT_EYE])
#         right_eye_points = np.array([[face_landmarks.landmark[i].x * frame_width,
#                                     face_landmarks.landmark[i].y * frame_height] 
#                                    for i in self.RIGHT_EYE])
        
#         # Detect blinks
#         self.detect_blink(left_eye_points, right_eye_points)
        
#         # Calculate eye centers
#         left_center = np.mean(left_eye_points, axis=0)
#         right_center = np.mean(right_eye_points, axis=0)
        
#         # Draw eye centers
#         cv2.circle(frame, tuple(left_center.astype(int)), 3, (0, 255, 0), -1)
#         cv2.circle(frame, tuple(right_center.astype(int)), 3, (0, 255, 0), -1)
        
#         # Calculate average eye position
#         avg_x = (left_center[0] + right_center[0]) / 2
#         avg_y = (left_center[1] + right_center[1]) / 2
        
#         return avg_x, avg_y
        
#     def move_mouse(self, eye_x, eye_y):
#         screen_width, screen_height = pyautogui.size()
        
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
        
#         # Store gaze point for statistics
#         self.gaze_points.append((new_x, new_y))
        
#         # Update heatmap
#         self.update_heatmap(new_x, new_y)
        
#         # Move mouse
#         pyautogui.moveTo(new_x, new_y)
        
#     def update_statistics(self):
#         if self.start_time:
#             duration = time.time() - self.start_time
#             blinks_per_minute = (self.blink_count / duration) * 60
            
#             stats_text = (
#                 f"Session Duration: {int(duration)} seconds\n"
#                 f"Total Blinks: {self.blink_count}\n"
#                 f"Blinks per Minute: {blinks_per_minute:.1f}\n"
#                 f"Gaze Points Recorded: {len(self.gaze_points)}\n"
#             )
            
#             self.stats_text.delete(1.0, tk.END)
#             self.stats_text.insert(tk.END, stats_text)
        
#     def update_video(self):
#         ret, frame = self.cap.read()
#         if ret:
#             # Flip frame horizontally for mirror effect
#             frame = cv2.flip(frame, 1)
            
#             # Convert frame to RGB for MediaPipe
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
#             if self.calibrating:
#                 self.draw_calibration_point(rgb_frame)
            
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
                    
#                     # Update statistics
#                     self.update_statistics()
            
#             # Draw UI elements on frame
#             self.draw_ui_elements(rgb_frame)
            
#             # Convert to PhotoImage for tkinter
#             img = Image.fromarray(rgb_frame)
#             img = img.resize((640, 480), Image.Resampling.LANCZOS)
#             img = ImageTk.PhotoImage(image=img)
            
#             # Update video feed
#             self.video_label.imgtk = img
#             self.video_label.configure(image=img)
        
#         # Schedule next update
#         self.root.after(10, self.update_video)
        
#     def draw_ui_elements(self, frame):
#         # Draw tracking status
#         status_text = "Tracking Active" if self.tracking_active else "Tracking Inactive"
#         cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
#                    1, (0, 255, 0) if self.tracking_active else (0, 0, 255), 2)
        
#         # Draw blink indicator
#         if time.time() - self.last_blink_time < 0.2:  # Show blink indicator for 200ms
#             cv2.putText(frame, "BLINK!", (frame.shape[1]//2 - 40, 30), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
#         # Draw FPS
#         fps = self.cap.get(cv2.CAP_PROP_FPS)
#         cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 120, 30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
#     def start_tracking(self):
#         self.tracking_active = True
#         self.start_time = time.time() if not self.start_time else self.start_time
#         self.status_label.config(text="Status: Tracking active")
        
#     def stop_tracking(self):
#         self.tracking_active = False
#         self.status_label.config(text="Status: Not tracking")
        
#     def cleanup(self):
#         # Save settings before closing
#         self.save_settings()
        
#         # Export final statistics
#         self.export_statistics()
        
#         # Release resources
#         self.cap.release()
#         self.face_mesh.close()
        
#         # Close matplotlib figure
#         plt.close(self.fig)
        
#         # Destroy main window
#         self.root.destroy()

# def main():
#     # Enable DPI awareness for better display on Windows
#     try:
#         from ctypes import windll
#         windll.shcore.SetProcessDpiAwareness(1)
#     except:
#         pass
    
#     # Create main window
#     root = tk.Tk()
#     root.title("Advanced Eye Tracking Mouse Control")
    
#     # Set minimum window size
#     root.minsize(800, 600)
    
#     # Create application instance
#     app = EyeTrackingApp(root)
    
#     # Set cleanup on window close
#     root.protocol("WM_DELETE_WINDOW", app.cleanup)
    
#     # Start application
#     root.mainloop()

# if __name__ == "__main__":
#     main()