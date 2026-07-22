import cv2
import os
import csv
from datetime import datetime
import time

# Configuration
DATASET_PATH = "C:\spyder\dataset"r
CSV_FILE = "Attedance.csv"  # Using your specified filename
SIMILARITY_THRESHOLD = 0.8  # Adjust matching sensitivity (0-1)
CHECK_INTERVAL = 2  # Seconds between automatic checks

def load_dataset():
    """Load dataset images with Name_ID.jpg format"""
    dataset = []
    for filename in os.listdir(DATASET_PATH):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Extract Name and ID from filename (format: "Name_ID.jpg")
                name_id = os.path.splitext(filename)[0]
                parts = name_id.split('_')
                if len(parts) >= 2:
                    person_name = parts[0]
                    person_id = '_'.join(parts[1:])  # In case ID has underscores
                    image_path = os.path.join(DATASET_PATH, filename)
                    image = cv2.imread(image_path)
                    if image is not None:
                        dataset.append({
                            'id': person_id,
                            'name': person_name,
                            'image': cv2.resize(image, (200, 200))  # Standard size
                        })
                        print(f"Loaded: {person_name} (ID: {person_id})")
            except Exception as e:
                print(f"Skipping {filename}: {str(e)}")
    return dataset

def compare_images(img1, img2):
    """Simple image comparison using histogram analysis"""
    if img1 is None or img2 is None:
        return 0
    
    # Convert to grayscale and compare histograms
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def update_attendance(person_id, person_name):
    """Update CSV with your specified format"""
    now = datetime.now()
    new_entry = {
        'ID': person_id,
        'Name': person_name,
        'Attendance': 'Present',
        'Date': now.strftime("%Y-%m-%d"),
        'Time': now.strftime("%H:%M:%S")
    }
    
    # Check if entry already exists today
    existing_entries = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'r') as file:
            reader = csv.DictReader(file)
            existing_entries = list(reader)
    
    # Prevent duplicate entries for same person on same day
    today = now.strftime("%Y-%m-%d")
    duplicate = any(
        entry['ID'] == person_id and entry['Date'] == today
        for entry in existing_entries
    )
    
    if not duplicate:
        with open(CSV_FILE, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['ID','Name','Attendance','Date','Time'])
            if file.tell() == 0:  # Write header if file is empty
                writer.writeheader()
            writer.writerow(new_entry)
        print(f"Attendance recorded: {person_name} (ID: {person_id})")
        return True
    else:
        print(f"Already recorded today: {person_name} (ID: {person_id})")
        return False

def main():
    dataset = load_dataset()
    if not dataset:
        print("Error: No valid images found in dataset")
        return
    
    # Initialize CSV if needed
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['ID','Name','Attendance','Date','Time'])
            writer.writeheader()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Automatic Attendance System Running...")
    print("Press Q to quit")
    
    last_check_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from webcam")
            break
        
        # Display live feed
        cv2.imshow('Attendance System - Press Q to quit', frame)
        
        # Automatic check at intervals
        current_time = time.time()
        if current_time - last_check_time > CHECK_INTERVAL:
            captured_img = cv2.resize(frame, (200, 200))
            
            best_match = None
            best_similarity = 0
            
            for person in dataset:
                similarity = compare_images(person['image'], captured_img)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person
            
            if best_match and best_similarity > SIMILARITY_THRESHOLD:
                if update_attendance(best_match['id'], best_match['name']):
                    # Visual feedback
                    cv2.putText(frame, "RECORDED", (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Attendance System - Press Q to quit', frame)
                    cv2.waitKey(1000)  # Show confirmation for 1 second
            
            last_check_time = current_time
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()