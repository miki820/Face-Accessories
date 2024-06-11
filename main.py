import cv2

face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
glasses = cv2.imread('files/glasses.png', -1)
mustache = cv2.imread('files/moustache.png', -1)

# Variables to toggle between accessories
show_glasses = False
show_mustache = False


# Function to add glasses or moustache to the face
def add_accessories(face_img, img, x, y, w, h, is_glasses):
    img = cv2.resize(img, (w, h))
    x_offset = x
    # Calculate offset to layout accessories on face properly depending on glasses or moustache
    y_offset = y - 25 if is_glasses else y + 40
    # Add accessory to face
    for c in range(0, 3):
        face_img[y_offset:y_offset + h, x_offset:x_offset + w, c] = img[:, :, c] * (
                img[:, :, 3] / 255.0) + face_img[y_offset:y_offset + h, x_offset:x_offset + w, c] * (
                                                                            1.0 - img[:, :, 3] / 255.0)


cap = cv2.VideoCapture(0)

# List to store the last detected faces
last_faces = []

# Loop to capture frame from the camera
while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # If no faces detected in this frame, use the faces from the previous frame
    if len(faces) > 0:
        last_faces = faces
    elif len(last_faces) > 0:
        faces = last_faces

    # Loop through detected faces and add accessories
    for (x, y, w, h) in faces:
        # Add proper accessory dependent on toggled variable
        if show_glasses:
            add_accessories(frame, glasses, x, y, w, h, is_glasses=True)
        if show_mustache:
            add_accessories(frame, mustache, x, y, w, h, is_glasses=False)

    cv2.imshow('Face', frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    # Show glasses when 'a' is pressed
    elif key == ord('a'):
        show_glasses = not show_glasses
    # Show moustache when 's' is pressed
    elif key == ord('s'):
        show_mustache = not show_mustache

cap.release()
cv2.destroyAllWindows()
