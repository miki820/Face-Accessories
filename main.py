import cv2

face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
glasses = cv2.imread('files/glasses.png', -1)
mustache = cv2.imread('files/moustache.png', -1)
current_elements = []

def overlay_glasses(face_img, glasses_img, x, y, w, h):
    glasses_img = cv2.resize(glasses_img, (w, h))
    x_offset = x
    y_offset = y + int(h/5)
    for c in range(0, 3):
        face_img[y_offset:y_offset+h, x_offset:x_offset+w, c] = glasses_img[:,:,c] * (glasses_img[:,:,3] / 255.0) +  face_img[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1.0 - glasses_img[:,:,3] / 255.0)
    current_elements.append((glasses_img, x_offset, y_offset, w, h))

def overlay_mustache(face_img, mustache_img, x, y, w, h):
    mustache_img = cv2.resize(mustache_img, (w, h))
    x_offset = x
    y_offset = y + int(h/1.5)
    for c in range(0, 3):
        face_img[y_offset:y_offset+h, x_offset:x_offset+w, c] = mustache_img[:,:,c] * (mustache_img[:,:,3] / 255.0) +  face_img[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1.0 - mustache_img[:,:,3] / 255.0)
    current_elements.append((mustache_img, x_offset, y_offset, w, h))

def draw_buttons(frame):
    button_color = (255, 255, 255)
    cv2.rectangle(frame, (10, 10), (110, 50), button_color, -1)
    cv2.putText(frame, 'Add Glasses', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (120, 10), (220, 50), button_color, -1)
    cv2.putText(frame, 'Add Mustache', (125, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

def check_button_clicks(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 < x < 110 and 10 < y < 50:
            overlay_glasses(frame, glasses, x, y, glasses.shape[1], glasses.shape[0])
        elif 120 < x < 220 and 10 < y < 50:
            overlay_mustache(frame, mustache, x, y, mustache.shape[1], mustache.shape[0])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        for element in current_elements:
            element_img, x_offset, y_offset, element_w, element_h = element
            for c in range(0, 3):
                frame[y_offset:y_offset+element_h, x_offset:x_offset+element_w, c] = element_img[:,:,c] * (element_img[:,:,3] / 255.0) +  frame[y_offset:y_offset+element_h, x_offset:x_offset+element_w, c] * (1.0 - element_img[:,:,3] / 255.0)
    draw_buttons(frame)
    cv2.imshow('Frame', frame)
    cv2.setMouseCallback('Frame', check_button_clicks)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
