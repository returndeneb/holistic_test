from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions, HolisticLandmarkerResult, RunningMode ,FaceLandmarkerResult ,FaceLandmarkerOptions, FaceLandmarker
from mediapipe.tasks.python import BaseOptions
import mediapipe as mp
import time
import cv2

fps_avg_frame_count = 10
DETECTION_RESULT = None
COUNTER, FPS = 0, 0
START_TIME = time.time()

def result_callback(result: HolisticLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, DETECTION_RESULT
        
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        DETECTION_RESULT = result
        COUNTER += 1

base_options = BaseOptions(model_asset_path='holistic_landmarker.task')
options = HolisticLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.LIVE_STREAM,
    # output_face_blendshapes=True,
    result_callback=result_callback)

detector = HolisticLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failed to grab a frame or frame is None")
        continue 
    
    img = cv2.flip(img, 1)
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    try:
        pass
        detector.detect_async(mp_img, time.time_ns() // 1000000)
    except Exception as e:
        print(f"Error during detection: {e}")

    # if DETECTION_RESULT:
    #      if DETECTION_RESULT.face_blendshapes:
    #           face_blendshapes = DETECTION_RESULT.face_blendshapes[0]
    #           print(face_blendshapes.shape)