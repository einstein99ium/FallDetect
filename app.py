# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st
import cv2
# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Fall Detection using YOLOv8")

# Sidebar
st.sidebar.header("ML Model Config")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

model_path = 'ultralytics/yolov8x-pose.pt'

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

def process_frame(model, frame):
    det_params = {
        'source': frame,
        'imgsz': 640,
        'conf': 0.5,
        'iou': 0.45
    }
    results = model(**det_params)
    annotated_frame = results[0].plot()

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
            cv2.putText(annotated_frame, f'state: fall!!', tuple(loc_xy), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    return annotated_frame

def process_image(model, image_path, frame_container):
    frame = cv2.imread(image_path)
    annotated_frame = process_frame(model, frame)
    frame_container.image(annotated_frame, '')

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                


                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")