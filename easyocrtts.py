import cv2
import easyocr
from gtts import gTTS
import os
import tempfile
import time

# Initialize the webcam
video = cv2.VideoCapture(0)
video.set(3, 1280)
video.set(4, 720)

# Create a temporary directory to store the mp3 files
temp_dir = tempfile.gettempdir()

def text_to_speech(text):
    speech = gTTS(text=text, lang='en', slow=False)
    temp_file = os.path.join(temp_dir, "temp_audio.mp3")
    speech.save(temp_file)
    os.system(f"start {temp_file}")

# Initialize the EasyOCR reader with GPU support
reader = easyocr.Reader(['en'], gpu=True)

while True:
    check, frame = video.read()
    if not check:
        break
    
    # Perform OCR using EasyOCR
    result = reader.readtext(frame)
    extracted_text = ""
    
    for (bbox, text, prob) in result:
        if prob > 0.5:  # Only consider high confidence results
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            
            cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
            cv2.putText(frame, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            extracted_text += text + " "
    
    if extracted_text.strip():
        text_to_speech(extracted_text)
    
    cv2.imshow('Video Capture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Adding a small delay to avoid excessive TTS calls
    time.sleep(2)

video.release()
cv2.destroyAllWindows()
