import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

# General instructions: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python
#
# Available models: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models
#
# requires downloading model to root of repo i.e.
# `wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task -O pose_landmarker.task`

detector = None

_init_mediapipe_pose_called = False


def init_mediapipe_pose():
    global _init_mediapipe_pose_called, detector

    if _init_mediapipe_pose_called:
        raise ValueError("init_mediapipe_pose() called twice")

    _init_mediapipe_pose_called = True

    base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options, output_segmentation_masks=False
    )
    detector = vision.PoseLandmarker.create_from_options(options)


def mediapipe_pose_adapter_image(input_image):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(input_image))
    detection_result = detector.detect(mp_image)
    pose_landmarks = detection_result.pose_landmarks

    if len(pose_landmarks) == 0:
        return None

    pose = np.zeros((input_image.height, input_image.width, 3), dtype=np.uint8)
    draw_landmarks_on_image(pose, pose_landmarks)

    pose = Image.fromarray(pose)

    return pose


def draw_landmarks_on_image(pose, pose_landmarks):
    for idx in range(len(pose_landmarks)):
        pose_landmarks = pose_landmarks[idx]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            pose,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
