from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

def main():
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    detector = vision.PoseLandmarker.create_from_options(options)

    for i in range(4):
        image = Image.open(f'./sdxl_t2i_mediapipe_pose_validation/{i}_input.png')

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image))
        detection_result = detector.detect(mp_image)
        pose_landmarks = detection_result.pose_landmarks


        pose = np.zeros((image.height, image.width, 3), dtype=np.uint8)
        draw_landmarks_on_image(pose, pose_landmarks)

        pose = Image.fromarray(pose)
        pose = pose.resize((1024, 1024))

        pose.save(f'./sdxl_t2i_mediapipe_pose_validation/{i}.png')

def draw_landmarks_on_image(pose, pose_landmarks):
  for idx in range(len(pose_landmarks)):
    pose_landmarks = pose_landmarks[idx]

    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      pose,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())

if __name__ == "__main__":
    main()