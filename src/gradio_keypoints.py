import gradio as gr
import numpy as np
import skimage
import cv2
import joblib

from src.utility.keypoints_utilities import *

title = "Analiza cyfrowa obrazow - Projekt"
description = "<center>Showcase of classifiers trained on different approach of feature extraction</center>"
article = """

"""

forest_list = []

for i in range(16):
    forest_list.append(joblib.load("../models/" + str(i) + "_RandomForestRegressor"))


svr_list = []
for i in range(32):
    svr_list.append(joblib.load("../models/" + str(i) + "_LinearSVR_scaled"))

std = joblib.load("../models/" + "scaler")

# examples = [
#     [None, "../data/yoga/DATASET/TEST/downdog/00000000.jpg"],
#     # [None, "../data/yoga/DATASET/TEST/goddess/00000052.jpg"]
# ]


def fn(model_choice, img):
    if model_choice == "random_forest":
        oy, ox, oc = img.shape

        im = cv2.resize(img, (128, 128))
        im = np.asarray(im)

        fd = skimage.feature.hog(im,
                                       visualize=False,
                                       channel_axis=-1)

        keypoints = []

        for classifier in forest_list:
            keypoints.append(classifier.predict([fd])[0])

        scaled_keypoints = scale_points((oy, ox), (128, 128), keypoints)

        return draw_skeleton(img, scaled_keypoints, radius=1, color=(255, 0, 0))

    elif model_choice == "svr":
        oy, ox, oc = img.shape

        im = cv2.resize(img, (128, 128))
        im = np.asarray(im)

        fd = skimage.feature.hog(im,
                                 visualize=False,
                                 channel_axis=-1)

        fd = std.transform([fd])

        keypoints = []

        for i in range(0, len(svr_list), 2):
            keypoints.append([svr_list[i].predict(fd)[0], svr_list[i+1].predict(fd)[0]])

        scaled_keypoints = scale_points((oy, ox), (128, 128), keypoints)

        return draw_skeleton(img, scaled_keypoints, radius=1, color=(255, 0, 0))


gr.Interface(
    fn,
    [gr.inputs.Dropdown(["svr", "random_forest"], default='random_forest'), gr.Image()],
    ["image"],
    # examples=examples,
    title=title,
    description=description,
    article=article,
    allow_flagging='never'
).launch(share=False)
