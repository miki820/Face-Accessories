import cv2


face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')

glasses = cv2.imread('files/glasses.png', -1)
mustache = cv2.imread('files/moustache.png', -1)

def overlay_glasses(face_img, glasses_img, x, y, w, h):
    glasses_img = cv2.resize(glasses_img, (w, h))
    x_offset = x
    y_offset = y
    for c in range(0, 3):
        face_img[y_offset:y_offset+h, x_offset:x_offset+w, c] = glasses_img[:,:,c] * (glasses_img[:,:,3] / 255.0) +  face_img[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1.0 - glasses_img[:,:,3] / 255.0)


def overlay_mustache(face_img, mustache_img, x, y, w, h):
    mustache_img = cv2.resize(mustache_img, (w, h))
    x_offset = x
    y_offset = y + int(h/1.5)
    for c in range(0, 3):
        face_img[y_offset:y_offset+h, x_offset:x_offset+w, c] = mustache_img[:,:,c] * (mustache_img[:,:,2] / 255.0) +  face_img[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1.0 - mustache_img[:,:,2] / 255.0)

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
        overlay_glasses(frame, glasses, x, y, w, h)
        overlay_mustache(frame, mustache, x, y, w, h)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
