import cv2
import os
import csv
from datetime import datetime
import time
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading

# Configuration
DATASET_PATH = r"C:\spyder\dataset"
CSV_FILE = "Attedance.csv"
SIMILARITY_THRESHOLD = 0.8
CHECK_INTERVAL = 2

class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Attendance System")
        
        # GUI Setup
        self.setup_gui()
        
        # System variables
        self.cap = None
        self.running = False
        self.dataset = []
        self.last_check_time = 0
        
    def setup_gui(self):
        """Initialize all GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Camera display
        self.camera_label = ttk.Label(main_frame)
        self.camera_label.pack()
        
        # Status display
        self.status_var = tk.StringVar(value="System Ready")
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var, foreground="blue").pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Control buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = ttk.Button(btn_frame, text="Start", command=self.toggle_attendance)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="View Attendance", command=self.view_attendance).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Exit", command=self.cleanup).pack(side=tk.RIGHT, padx=5)
        
        # Log display
        log_frame = ttk.Frame(main_frame)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(log_frame, text="Activity Log:").pack(anchor=tk.W)
        self.log_text = tk.Text(log_frame, height=8, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
    def log_message(self, message):
        """Add message to log display"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
    def toggle_attendance(self):
        """Start/stop the attendance system"""
        if not self.running:
            self.start_system()
        else:
            self.stop_system()
            
    def start_system(self):
        """Initialize and start the attendance system"""
        self.log_message("Initializing system...")
        
        # Load dataset
        self.dataset = self.load_dataset()
        if not self.dataset:
            messagebox.showerror("Error", "No valid images found in dataset")
            return
            
        # Initialize CSV
        if not os.path.exists(CSV_FILE):
            with open(CSV_FILE, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['ID','Name','Attendance','Date','Time'])
                writer.writeheader()
            self.log_message("Created new attendance file")
        
        # Start camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return
            
        self.running = True
        self.start_btn.config(text="Stop")
        self.status_var.set("System Running")
        self.log_message("Attendance system started")
        
        # Start the camera update thread
        self.update_camera()
        
    def stop_system(self):
        """Stop the attendance system"""
        self.running = False
        self.start_btn.config(text="Start")
        self.status_var.set("System Ready")
        self.log_message("Attendance system stopped")
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
    def update_camera(self):
        """Update the camera feed and perform face checks"""
        if not self.running or not self.cap:
            return
            
        ret, frame = self.cap.read()
        if ret:
            # Convert to RGB for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.camera_label.imgtk = imgtk
            self.camera_label.config(image=imgtk)
            
            # Automatic check at intervals
            current_time = time.time()
            if current_time - self.last_check_time > CHECK_INTERVAL:
                self.check_attendance(frame)
                self.last_check_time = current_time
                
        # Schedule next update
        self.root.after(30, self.update_camera)
        
    def check_attendance(self, frame):
        """Check for attendance matches"""
        captured_img = cv2.resize(frame, (200, 200))
        
        best_match = None
        best_similarity = 0
        
        for person in self.dataset:
            similarity = self.compare_images(person['image'], captured_img)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person
        
        if best_match and best_similarity > SIMILARITY_THRESHOLD:
            if self.update_attendance(best_match['id'], best_match['name']):
                # Visual feedback
                cv2.putText(frame, "RECORDED", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.log_message(f"Attendance recorded for {best_match['name']} (ID: {best_match['id']})")
                
    def load_dataset(self):
        """Load dataset images with Name_ID.jpg format"""
        dataset = []
        if not os.path.exists(DATASET_PATH):
            messagebox.showerror("Error", f"Dataset folder not found at {DATASET_PATH}")
            return dataset
            
        for filename in os.listdir(DATASET_PATH):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    name_id = os.path.splitext(filename)[0]
                    parts = name_id.split('_')
                    if len(parts) >= 2:
                        person_name = parts[0]
                        person_id = '_'.join(parts[1:])
                        image_path = os.path.join(DATASET_PATH, filename)
                        image = cv2.imread(image_path)
                        if image is not None:
                            dataset.append({
                                'id': person_id,
                                'name': person_name,
                                'image': cv2.resize(image, (200, 200))
                            })
                            self.log_message(f"Loaded: {person_name} (ID: {person_id})")
                except Exception as e:
                    self.log_message(f"Skipping {filename}: {str(e)}")
        return dataset
        
    def compare_images(self, img1, img2):
        """Simple image comparison using histogram analysis"""
        if img1 is None or img2 is None:
            return 0
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
    def update_attendance(self, person_id, person_name):
        """Update CSV with attendance record"""
        now = datetime.now()
        new_entry = {
            'ID': person_id,
            'Name': person_name,
            'Attendance': 'Present',
            'Date': now.strftime("%Y-%m-%d"),
            'Time': now.strftime("%H:%M:%S")
        }
        
        # Check for existing entry today
        existing_entries = []
        if os.path.exists(CSV_FILE):
            with open(CSV_FILE, 'r') as file:
                reader = csv.DictReader(file)
                existing_entries = list(reader)
                
        today = now.strftime("%Y-%m-%d")
        duplicate = any(
            entry['ID'] == person_id and entry['Date'] == today
            for entry in existing_entries
        )
        
        if not duplicate:
            with open(CSV_FILE, 'a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['ID','Name','Attendance','Date','Time'])
                if file.tell() == 0:
                    writer.writeheader()
                writer.writerow(new_entry)
            return True
        else:
            self.log_message(f"Duplicate detected: {person_name} (ID: {person_id})")
            return False
            
    def view_attendance(self):
        """Display attendance records"""
        if os.path.exists(CSV_FILE):
            os.startfile(CSV_FILE)
        else:
            messagebox.showinfo("Info", "No attendance records found")
            
    def cleanup(self):
        """Clean up resources before closing"""
        self.stop_system()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()