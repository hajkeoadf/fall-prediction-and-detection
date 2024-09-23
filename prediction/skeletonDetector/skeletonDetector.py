import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import cv2
import tensorflow_io as tfio

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Some modules to display an animation using imageio.
import imageio
from IPython.display import HTML, display
from cropping import *
from vis import *

# Initialize the TFLite interpreter
interpreter = tf.lite.Interpreter(model_path="movenet_float16.tflite")
interpreter.allocate_tensors()
input_size = 256


def movenet(input_image):
    """Runs detection on an input image.

    Args:
        input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
        A [1, 1, 17, 3] float numpy array representing the predicted keypoint
        coordinates and scores.
    """
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores


def get_skeleton(video_path):
    # Load the input image.
    video_path = video_path
    cap = cv2.VideoCapture(video_path)
    image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    crop_region = init_crop_region(image_height, image_width)

    all_keypoints = []
    # output_images = []
    # bar = display(progress(0, num_frames-1), display_id=True)
    for frame_idx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
        keypoints_with_scores = run_inference(
            movenet, image, crop_region,
            crop_size=[input_size, input_size])
        all_keypoints.append(keypoints_with_scores[0][0])
        # # Visualization code
        # output_images.append(draw_prediction_on_image(
        #     image.numpy().astype(np.int32),
        #     keypoints_with_scores, crop_region=None,
        #     close_figure=True, output_image_height=300))
        # crop_region = determine_crop_region(
        #     keypoints_with_scores, image_height, image_width)
        # bar.update(progress(frame_idx, num_frames-1))

    return all_keypoints
    # # Prepare gif visualization.
    # output = np.stack(output_images, axis=0)
    # to_gif(output, fps=10)
