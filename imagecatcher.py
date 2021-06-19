import cv2
import numpy as np
import requests

cam = cv2.VideoCapture(0)

cv2.namedWindow("Waste")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        response = requests.post('http://0.0.0.0:5432/predict', json={
            "name": "string",
            "img": np.array(frame).astype('int8').flatten().tolist(),
            "shape": frame.shape
        })
        print(response.text)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
