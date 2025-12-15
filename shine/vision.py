import cv2

# Specify which camera to use (0 = default)
CAM_ID = 0

# Path to the pre-trained model file for face recognition
CASCADE_FILE = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml"

# Reduced resolution for better Pi performance
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

def find_and_open():
    cap = None
    
    for i in range(5):
        print(f"Attempting to open camera {i}...")
        current_cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        
        if current_cap.isOpened():
            print(f"Camera {i} opened successfully!")
            cap = current_cap
            break
        else:
            current_cap.release() 
    
    if cap is None:
        print("No camera could be opened. Exiting.")
        return None, None

    # --- FIX #1: Resolution ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # --- FIX #2: BUFFER SIZE (CRITICAL FOR LAG) ---
    # This tells the camera to only keep the newest frame.
    # Note: Not all cameras support this, but it's the best command to try.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    cascade = cv2.CascadeClassifier(CASCADE_FILE)
    
    if cascade.empty():
        print(f"Error: Failed to load cascade classifier at {CASCADE_FILE}")
        cap.release()
        return None, None

    print("Starting face recognition... (Press 'q' to quit)")
    return cap, cascade


def camera(cap, cascade, debug=False):

    # --- FIX #2 (Backup): Manual Buffer Clear ---
    # Sometimes setting BUFFERSIZE to 1 doesn't work on all Pi setups.
    # We grab (but don't decode) one extra frame to flush the buffer slightly.
    # This is much faster than the old "loop 5 times" method.
    cap.grab()

    # Read one frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        return None
    
    center = (frame.shape[1] // 2, frame.shape[0] // 2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # --- FIX #3: OPTIMIZED DETECTION ---
    # scaleFactor: Increased from 1.1 to 1.2 (Faster scanning)
    # minSize: Increased from (30,30) to (60,60) (Ignores tiny false faces, speeds up CPU)
    faces = cascade.detectMultiScale(
        gray, 
        scaleFactor=1.15, 
        minNeighbors=3, 
        minSize=(40, 40) 
    )

    move = None

    if len(faces) > 0:
        # Find the largest face
        target_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = target_face
        
        if debug:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face_center = (x + w // 2, y + h // 2)
        move = (center[0] - face_center[0], center[1] - face_center[1])

    if debug:
        cv2.imshow('Face Recognition', frame)

    return move


def end_camera(cap):
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("Face recognition ended.")

if __name__ == '__main__':
    cap, cascade = find_and_open()
    
    if cap is not None:
        try:
            while True:
                resp = camera(cap, cascade, debug=True)
                if resp is not None:
                    print(f"Moving: {resp}")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            end_camera(cap)
