import cv2
import os

CASCADE_PATH = haar_cascade_filedownload
PREDEFINED_IMAGE_PATH = your_image
PREDEFINED_VIDEO_PATH = your_video

def load_cascade():
    if not os.path.isfile(CASCADE_PATH):
        print(f"Error: Could not find the cascade file at {CASCADE_PATH}")
        return None
    
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        print("Error: Failed to load the cascade file")
        return None
    
    return cascade

def detect_faces_in_image(image_path):
    face_cascade = load_cascade()
    if face_cascade is None:
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read the image file at: {image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    cv2.imshow('Face Detection - Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_faces_in_video(source=0):
    face_cascade = load_cascade()
    if face_cascade is None:
        return
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        return
    
    while True:
        ret, img = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow('Face Detection - Video', img)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:  # ESC key to exit
            break
            
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("\nFace Detection Program")
    print("1. Detect faces from webcam")
    print("2. Detect faces from predefined video file")
    print("3. Detect faces from predefined image file")
    print("4. Detect faces from custom video file")
    print("5. Detect faces from custom image file")
    print("0. Exit")
    
    while True:
        choice = input("\nEnter your choice (0-5): ").strip()
        
        if choice == '0':
            print("Exiting program...")
            break
            
        elif choice == '1':
            print("\nStarting webcam face detection... (Press ESC to stop)")
            detect_faces_in_video(0)  # 0 for default webcam
            
        elif choice == '2':
            if not os.path.isfile(PREDEFINED_VIDEO_PATH):
                print(f"\nError: Predefined video not found at {PREDEFINED_VIDEO_PATH}")
                print("Please update the PREDEFINED_VIDEO_PATH in the code.")
            else:
                print(f"\nProcessing predefined video: {PREDEFINED_VIDEO_PATH}")
                print("Press ESC to stop playback")
                detect_faces_in_video(PREDEFINED_VIDEO_PATH)
                
        elif choice == '3':
            if not os.path.isfile(PREDEFINED_IMAGE_PATH):
                print(f"\nError: Predefined image not found at {PREDEFINED_IMAGE_PATH}")
                print("Please update the PREDEFINED_IMAGE_PATH in the code.")
            else:
                print(f"\nProcessing predefined image: {PREDEFINED_IMAGE_PATH}")
                detect_faces_in_image(PREDEFINED_IMAGE_PATH)
                
        elif choice == '4':
            video_path = input("\nEnter video file path: ").strip('"\'')
            if not os.path.isfile(video_path):
                print("\nError: File not found. Please check the path.")
            else:
                print(f"\nProcessing video: {video_path}")
                print("Press ESC to stop playback")
                detect_faces_in_video(video_path)
                
        elif choice == '5':
            image_path = input("\nEnter image file path: ").strip('"\'')
            if not os.path.isfile(image_path):
                print("\nError: File not found. Please check the path.")
            else:
                print(f"\nProcessing image: {image_path}")
                detect_faces_in_image(image_path)
                
        else:
            print("\nInvalid choice! Please enter a number between 0-5")

if __name__ == "__main__":

    main()
