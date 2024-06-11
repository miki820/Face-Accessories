import cv2

# Załaduj kaskadę Haara do detekcji twarzy
face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')

# Załaduj obrazki elementów, np. okulary i wąsy
glasses = cv2.imread('files/glasses.png', -1)
mustache = cv2.imread('files/moustache.png', -1)

# Funkcja do nałożenia okularów na twarz
def overlay_glasses(face_img, glasses_img, x, y, w, h):
    # Dopasuj rozmiar obrazu okularów do obszaru twarzy
    glasses_img = cv2.resize(glasses_img, (w, h))
    # Ustal punkt referencyjny do nałożenia okularów na oczach
    x_offset = x
    y_offset = y
    # Nałóż okulary na twarz
    for c in range(0, 3):
        face_img[y_offset:y_offset+h, x_offset:x_offset+w, c] = glasses_img[:,:,c] * (glasses_img[:,:,3] / 255.0) +  face_img[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1.0 - glasses_img[:,:,3] / 255.0)

# Funkcja do nałożenia wąsów na twarz
def overlay_mustache(face_img, mustache_img, x, y, w, h):
    # Dopasuj rozmiar obrazu wąsów do obszaru twarzy
    mustache_img = cv2.resize(mustache_img, (w, h))
    # Ustal punkt referencyjny do nałożenia wąsów na ustach
    x_offset = x
    y_offset = y + int(h/1.5)
    # Nałóż wąsy na twarz
    for c in range(0, 3):
        face_img[y_offset:y_offset+h, x_offset:x_offset+w, c] = mustache_img[:,:,c] * (mustache_img[:,:,2] / 255.0) +  face_img[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1.0 - mustache_img[:,:,2] / 255.0)

# Uruchom kamerę
cap = cv2.VideoCapture(0)

# Inicjalizuj zmienne przechowujące ostatnie pozycje twarzy
last_faces = []

while True:
    # Odczytaj klatkę z kamery
    ret, frame = cap.read()

    # Konwertuj klatkę na odcień szarości
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Wykryj twarze na klatce
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        last_faces = faces
    elif len(last_faces) > 0:
        faces = last_faces

    # Dla każdej wykrytej twarzy
    for (x, y, w, h) in faces:
        overlay_glasses(frame, glasses, x, y, w, h)
        overlay_mustache(frame, mustache, x, y, w, h)

    # Wyświetl klatkę
    cv2.imshow('Frame', frame)

    # Przerwij pętlę, jeśli naciśnięto klawisz 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnij zasoby
cap.release()
cv2.destroyAllWindows()
