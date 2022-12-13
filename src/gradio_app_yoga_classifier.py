import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import skimage
import cv2
import joblib

title = "Analiza cyfrowa obrazow - Projekt"
description = "<center>Showcase of classifiers trained on different approach of feature extraction</center>"
article = """

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla vitae sapien purus. Sed a leo ac leo hendrerit porta vel non augue. Nulla congue, nulla vitae eleifend consequat, leo quam porta odio, id posuere elit nunc et massa. Donec euismod velit ipsum, eu fermentum lorem pulvinar nec. In ullamcorper rhoncus sapien at auctor. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Duis blandit blandit urna et placerat. Quisque suscipit odio fringilla urna interdum, id scelerisque nisl blandit. Integer lacinia arcu posuere lectus euismod ornare. Sed nunc dolor, volutpat vitae massa quis, porta ultricies ex. Nulla blandit, tellus ac sagittis varius, lorem mi condimentum lorem, quis lobortis nisl enim sed dui. Nunc eu ullamcorper tellus. Fusce sapien diam, ultrices id felis a, gravida faucibus ipsum. Etiam in dolor suscipit, porttitor erat vitae, mollis justo. Donec ac lorem in purus aliquet imperdiet vel a lacus. Nam lacinia sit amet felis in lacinia.

Proin venenatis blandit nunc quis rhoncus. Fusce sollicitudin magna nulla, ac aliquet ex laoreet sit amet. Duis quam neque, luctus venenatis massa sit amet, volutpat sodales nisl. Fusce in felis nec sapien consequat dictum vel id elit. Phasellus eu magna fermentum, finibus est venenatis, lobortis leo. Donec imperdiet, tellus at porta eleifend, lorem neque feugiat orci, ac cursus diam tellus ac felis. Maecenas id magna et metus molestie venenatis. Morbi nec neque tempus, auctor ex non, sollicitudin orci. Nulla cursus auctor quam nec tincidunt. Praesent sit amet magna ac nisl molestie tincidunt et in ante. Maecenas vestibulum non nulla vel rutrum. Aenean et sapien turpis. Proin vestibulum sodales lacus. Fusce sed erat non est egestas tempor eu eu magna. Ut a arcu id nisi convallis eleifend. Curabitur turpis mi, varius nec lacus ut, imperdiet sodales tellus.

Nunc facilisis tempor convallis. Ut id turpis non velit fermentum gravida ut eget ipsum. Nulla ante velit, porttitor ac finibus porttitor, lacinia pharetra nulla. Curabitur maximus eros nisi, vitae maximus mi imperdiet in. Quisque non fermentum odio, a dictum turpis. Vestibulum eget varius lorem. Nunc condimentum luctus porttitor. Donec risus sapien, sagittis vitae purus eu, vulputate sagittis magna.

Nunc lobortis mauris quis quam finibus, et elementum ligula sodales. Sed ac viverra dui. Nulla facilisi. Vivamus nec quam eu odio ullamcorper hendrerit. Curabitur id lobortis ante. Ut vel pretium massa. Donec dapibus urna et tellus maximus rhoncus.

Proin ornare vel tellus eu mattis. Sed mollis elit mauris. Morbi vel eleifend sem. Pellentesque rhoncus fermentum pellentesque. Nullam sagittis ante eget ultrices egestas. Curabitur est arcu, vehicula vel dignissim eget, aliquet nec odio. Nulla eget iaculis sapien. Nam fringilla dui sed odio cursus, non posuere lectus sollicitudin. Quisque hendrerit tellus vel ex euismod, ac fringilla eros malesuada. 

"""

svm_hog = joblib.load("../models/1669842054_LinearSVC_0.918")
random_forest_hog = joblib.load("../models/1669842100_RandomForestClassifier_0.878")

# examples = [
#     [None, "../data/yoga/DATASET/TEST/downdog/00000000.jpg"],
#     # [None, "../data/yoga/DATASET/TEST/goddess/00000052.jpg"]
# ]


def fn(model_choice, img):
    if model_choice == "hog_svm":
        im = cv2.resize(img, (256, 256))
        im = np.asarray(im)

        fd, hog_image = skimage.feature.hog(im,
                                       visualize=True,
                                       channel_axis=-1)
        fd = fd.reshape(1, -1)
        pred = svm_hog.predict(fd)[0]
        return pred, hog_image / 255
    elif model_choice == "random_forest_hog":
        im = cv2.resize(img, (256, 256))
        im = np.asarray(im)

        fd, hog_image = skimage.feature.hog(im,
                                            visualize=True,
                                            channel_axis=-1)
        fd = fd.reshape(1, -1)
        pred = random_forest_hog.predict_proba(fd)
        classes = random_forest_hog.classes_
        out = {k: v for k, v in zip(classes, pred[0])}

        # print(pred)
        return out, hog_image / 255
    elif model_choice == "some crazy model":
        return gptj6B(0)[0], img


gr.Interface(
    fn,
    [gr.inputs.Dropdown(["hog_svm", "random_forest_hog", "some crazy model"], default='hog_svm'), gr.Image()],
    ["label", "image"],
    # examples=examples,
    title=title,
    description=description,
    article=article,
    allow_flagging='never'
).launch(share=False)
