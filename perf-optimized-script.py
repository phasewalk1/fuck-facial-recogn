import cv2
import face_recognition
import numpy as np


# encode an image of me
known_person_image = face_recognition.load_image_file("me.jpg")
known_person_face_encoding = face_recognition.face_encodings(known_person_image)[0]

# array of face encodings and names
known_face_encodings = [known_person_face_encoding]
known_face_names = ["me_uwa"]

face_locations = []
face_encodings = []
proc_this_frame = True

# init webcam
video_capture = cv2.VideoCapture(0)

while True:
  ret, frame = video_capture.read()

  # only process every other frame to save processing time
  if proc_this_frame:
    # resize the frame to 1/4 for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # convert BGR (OpenCv) to RGB (face_recognition)
    rgb_small_frame = small_frame[:, :, ::-1]

    # find face locations and encodings in small frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
      matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
      name = "Unknown"

      face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
        name = known_face_names[best_match_index]

      face_names.append(name)

  proc_this_frame = not proc_this_frame

  for (top, right, bottom, left), name in zip(face_locations, face_names):
    # rescale the small frame
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4

    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

  cv2.imshow("Video", frame)

  if cv2.waitKey(1) & 0xFF = ord('q'):
    break

video_capture.release()
cv2.destroyAllWindows()
