import cv2

face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')

glasses = cv2.imread('files/glasses.png', -1)
mustache = cv2.imread('files/moustache.png', -1)


def add_accessories(face_img, img, x, y, w, h, is_glasses):
    img = cv2.resize(img, (w, h))
    x_offset = x
    y_offset = y - 15 if is_glasses else y + int(h/3)
    for c in range(0, 3):
        face_img[y_offset:y_offset + h, x_offset:x_offset + w, c] = img[:, :, c] * (
                img[:, :, 3] / 255.0) + face_img[y_offset:y_offset + h, x_offset:x_offset + w, c] * (
                                                                            1.0 - img[:, :, 3] / 255.0)


cap = cv2.VideoCapture(0)

last_faces = []

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        last_faces = faces
    elif len(last_faces) > 0:
        faces = last_faces

    for (x, y, w, h) in faces:
        add_accessories(frame, glasses, x, y, w, h, is_glasses=True)
        add_accessories(frame, mustache, x, y, w, h, is_glasses=False)

    cv2.imshow('Face', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
