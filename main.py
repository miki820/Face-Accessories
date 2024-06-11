import cv2

face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')

glasses = cv2.imread('files/glasses.png', -1)
mustache = cv2.imread('files/moustache.png', -1)

show_glasses = False
show_mustache = False


def add_accessories(face_img, img, x, y, w, h, is_glasses):
    img = cv2.resize(img, (w, h))
    x_offset = x
    y_offset = y - 25 if is_glasses else y + 40
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
        if show_glasses:
            add_accessories(frame, glasses, x, y, w, h, is_glasses=True)
        if show_mustache:
            add_accessories(frame, mustache, x, y, w, h, is_glasses=False)

    cv2.imshow('Face', frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('a'):
        show_glasses = not show_glasses
    elif key == ord('s'):
        show_mustache = not show_mustache

cap.release()
cv2.destroyAllWindows()
