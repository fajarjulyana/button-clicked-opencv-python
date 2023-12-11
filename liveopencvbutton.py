import cv2
import mediapipe as mp

# Flag to track button click
button_clicked = False

# Function to be executed when the button is clicked
def on_button_click():
    global button_clicked
    button_clicked = not button_clicked

# Inisialisasi MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inisialisasi OpenCV
cap = cv2.VideoCapture(0)

# Membuat jendela OpenCV
cv2.namedWindow("Hand Tracking")

# Mendeteksi button
button_x, button_y, button_w, button_h = 100, 100, 100, 50

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mendeteksi tangan
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Menggambar landmark tangan
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                # Mendeteksi apakah tangan berada di atas tombol
                if button_x < landmarks.landmark[8].x * frame.shape[1] < button_x + button_w and \
                        button_y < landmarks.landmark[8].y * frame.shape[0] < button_y + button_h:
                    on_button_click()

    # Menggambar tombol
    if button_clicked:
        label = "Clicked"
    else:
        label = "Click Me!"

    cv2.rectangle(frame, (button_x, button_y), (button_x + button_w, button_y + button_h), (0, 255, 0), -1)
    cv2.putText(frame, label, (button_x + 10, button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Hand Tracking", frame)

    # Keluar dari loop jika tombol 'ESC' ditekan
    if cv2.waitKey(1) == 27:
        break

# Membersihkan dan menutup jendela
cap.release()
cv2.destroyAllWindows()
