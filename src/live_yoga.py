# import the opencv library
import cv2
import joblib
import skimage

from src.utility.keypoints_utilities import *

svm_hog = joblib.load("../models/1669842054_LinearSVC_0.918")

# define a video capture object
vid = cv2.VideoCapture(0)

while 1:

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # oy, ox, oc = frame.shape

    frame = frame[0:480, 100:580]

    oy, ox, oc = frame.shape

    im = cv2.resize(frame, (256, 256))

    fd, hog_image = skimage.feature.hog(im,
                             visualize=True,
                             channel_axis=-1)
    fd = fd.reshape(1, -1)
    pred = svm_hog.predict(fd)[0]
    print(pred)

    hog_image = cv2.resize(hog_image, (720, 720))

    hog_image = cv2.putText(hog_image, str(pred),
                            (80, 80), 2, 3, (255, 255, 255), 5)

    # Display the resulting frame
    cv2.imshow('frame', hog_image)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
