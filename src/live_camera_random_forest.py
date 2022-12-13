# import the opencv library
import cv2
import joblib
import skimage

from src.utility.keypoints_utilities import *

forest_list = []

for i in range(16):
    forest_list.append(joblib.load("../models/" + str(i) + "_RandomForestRegressor"))

# define a video capture object
vid = cv2.VideoCapture(0)

while 1:

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # oy, ox, oc = frame.shape

    frame = frame[0:480, 100:580]

    oy, ox, oc = frame.shape

    im = cv2.resize(frame, (128, 128))

    fd = skimage.feature.hog(im,
                             visualize=False,
                             channel_axis=-1)

    keypoints = []

    for classifier in forest_list:
        keypoints.append(classifier.predict([fd])[0])

    scaled_keypoints = scale_points((720, 720), (128, 128), keypoints)

    # Display the resulting frame
    cv2.imshow('frame', draw_skeleton(cv2.resize(frame, (720, 720)), scaled_keypoints, radius=1, color=(255, 0, 0)))

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
