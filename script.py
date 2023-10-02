import cv2
import face_recognition

# encode an image of my face uwa
known_person_image = face_recognition.load_image_file("me.jpg")
known_person_face_encoding = face_recognition.face_encodings(known_person_image)[0]

# array of face encodings
known_face_encodings = [known_person_face_encoding]
known_face_names = ["me_uwa"]

# init webcam
video_capture = cv2.VideoCapture(0)

while True:
    # read webcam
    ret, frame = video_capture.read()

    # get location encodings on face
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(
        face_locations, face_encodings
    ):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(
            frame,
            name,
            (left, bottom + 20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
