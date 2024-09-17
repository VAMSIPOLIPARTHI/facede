import threading
import cv2
from deepface import DeepFace

# Initialize webcam capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set webcam frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Counter and face match flag
face_match = False

# Load reference image for face comparison
reference_img = cv2.imread("reference.jpg")

# Threading flag to control execution
stop_threads = False

# Function to run face recognition in a separate thread
def face_recognition_task(frame):
    global face_match
    try:
        # Analyze the frame using DeepFace for face verification
        result = DeepFace.verify(frame, reference_img, model_name='VGG-Face')
        
        # Check if the faces match with a defined threshold
        if result["verified"]:
            face_match = True
        else:
            face_match = False

    except Exception as e:
        print(f"Error: {e}")

# Main loop to capture frames and start face recognition in a separate thread
while True:
    ret, frame = cap.read()

    if ret:
        # Start the face recognition in a separate thread
        recognition_thread = threading.Thread(target=face_recognition_task, args=(frame,))
        recognition_thread.start()

        # Display feedback based on face recognition result
        if face_match:
            cv2.putText(frame, "Face Matched!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Face Not Matched", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow("Face Recognition", frame)

    # Break the loop if 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        stop_threads = True
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
