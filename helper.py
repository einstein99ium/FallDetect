from ultralytics import YOLO
import time
import streamlit as st
import cv2
from pytube import YouTube
import numpy as np

import settings


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def detect(model, frame, conf):
    print("detect@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    det_params = {
        'source': frame,
        'imgsz': 640,  # 输入图像的大小
        'conf': conf,
        'iou': 0.45
    }
    results = model(**det_params)
    annotated_frame = results[0].plot()
    # 绘制跌倒
    fall = [0] * results[0].boxes.shape[0]
    for i in range(results[0].boxes.shape[0]):
        kpts = results[0].keypoints[i].xy
        if (kpts[0][5][1] + kpts[0][6][1]) / 2 >= (kpts[0][11][1] + kpts[0][12][1]) / 2:
            if kpts[0][5][1] > 0 and kpts[0][6][1] > 0 and kpts[0][11][1] > 0 and kpts[0][12][1] > 0:
                fall[i] = 1
        else:
            dy = (kpts[0][11][1] + kpts[0][12][1]) / 2 - (kpts[0][5][1] + kpts[0][6][1]) / 2
            dx = abs((kpts[0][11][0] + kpts[0][12][0]) / 2 - (kpts[0][5][0] + kpts[0][6][0]) / 2)
            if dx > 0 and dy > 0:
                deg = np.arctan((dy / dx).item()) * 180 / np.pi
                if deg < 50:
                    fall[i] = 1
        loc_xy = results[0].boxes[i].xyxy[0].to(int).tolist()[:2]
        loc_xy[1] -= 30
        if fall[i] == 0:
            cv2.putText(annotated_frame, f'state: normal.', tuple(loc_xy), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 255, 0), 2)
        else:
            print("fall###########################################")
            cv2.putText(annotated_frame, f'state: fall!!', tuple(loc_xy), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    return annotated_frame


def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))
    
    # Detect fall
    annotated_image = detect(model, image, conf)

    # Plot the detected objects on the video frame
    st_frame.image(annotated_image,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def play_youtube_video(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_youtube = st.sidebar.text_input("YouTube Video url")

    if st.sidebar.button('Detect Objects'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_rtsp_stream(conf, model):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image
                                             )
                else:
                    vid_cap.release()
                    vid_cap = cv2.VideoCapture(source_rtsp)
                    time.sleep(0.1)
                    continue
                    #break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
