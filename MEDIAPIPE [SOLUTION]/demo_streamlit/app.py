import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from typing import Union, BinaryIO
import io
import tempfile
from io import StringIO
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
exercise_dicc = {
    'Pull-up':          'pullup',
    'Squat':            'squat',
    'Push-up':          'pushup',
    'Triceps Pushdown': 'triceps_pushdown',
    'Bench Press':      'bench_press',
    'Biceps Curl':      'biceps_curl',
    'Cable Row':        'cable_row',
    'Lat Pulldown':     'lat_pulldown',
    'Lateral Raise':    'lateral_raise',
    'Abs Crunch':       'abs_crunch'
}
if "load_state" not in st.session_state:
     st.session_state.load_state = False



def video_detect(uploaded_video: Union[None, io.BytesIO], exercise_aux: str) -> str:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    stframe = st.empty()
    cap = cv2.VideoCapture(tfile.name)
    counter = 0
    stage = None  # "down" or "up"
    exercise = exercise_aux

    # Decide whether to resize frames
    resize_frames = True  # Set to False if you don't want to resize frames
    resize_width = 500

    # Initial frame size
    frame_rate = 20.0
    if resize_frames:
        ret, test_frame = cap.read()
        scale_ratio = resize_width / test_frame.shape[1]
        frame_size = (resize_width, int(test_frame.shape[0] * scale_ratio))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset cap to first frame after test read
    else:
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_filename = 'video.mp4'
    out = cv2.VideoWriter(output_filename, fourcc, frame_rate, frame_size)

    skip_frames = 2  # Process every 2nd frame
    frame_count = 0
    # Initialize Pose detection
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Ignoring empty camera frame.")
                break
            frame_count += 1
            if frame_count % skip_frames != 0:
                continue  # Skip this frame
              # Skip this frame
            if resize_frames:
                # Apply resizing here
                frame = cv2.resize(frame, frame_size)
            # Recolor the frame to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Detect the pose
            results = pose.process(image)
            image_height, image_width, _ = image.shape
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                if exercise == 'squat':

                    # Get coordinates for the left hip, knee, and ankle
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    # Calculate the angle
                    angle = calculate_angle(hip, knee, ankle)

                    # Squat logic
                    if angle > 160:  # Threshold for standing
                        if stage == 'down':
                            counter += 1
                            print(counter)  # Print the squat count
                        stage = 'up'
                    if angle < 100:  # Threshold for squat
                        stage = 'down'
                elif exercise == 'pushup':

                    # Get coordinates for the left hip, knee, and ankle
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    # Calculate the angle
                    angle = calculate_angle(shoulder, elbow, wrist)
                    if angle > 160:  # Threshold for standing
                        if stage == 'down':
                            counter += 1
                            print(counter)  # Print the squat count
                        stage = 'up'
                    if angle < 70:  # Threshold for squat
                        stage = 'down'
                elif exercise == 'pullup':
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    angle = calculate_angle(shoulder, elbow, wrist)
                    if angle < 90:  # Threshold for "up" phase of a pull-up (body lifted)
                        if stage == 'down':
                            counter += 1
                            print(counter)  # Print the pull-up count
                        stage = 'up'
                    elif angle > 160:  # Threshold for "down" phase (body lowered)
                        stage = 'down'
                elif exercise == 'triceps_pushdown':
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    angle = calculate_angle(shoulder, elbow, wrist)
                    print(angle)
                    if angle > 135:
                        if stage == 'close':
                            counter += 1
                            print(counter)
                        stage = 'open'
                    elif angle < 90:
                        stage = 'close'
                elif exercise == 'bench_press':
                    # Get coordinates for both the right and left shoulders, elbows, and wrists
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    # Calculate the angles for both arms
                    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                    # Average the angles of both arms for a more unified analysis
                    angle = (right_angle + left_angle) / 2

                    # Logic for detecting the bench press motion
                    if angle > 160:  # Threshold for "up" phase (arms extended)
                        if stage == 'down':  # Transition from down to up
                            counter += 1
                            print(counter)  # Print the bench press count
                        stage = 'up'
                    elif angle < 90:  # Threshold for "down" phase (starting or bottom position)
                        stage = 'down'
                elif exercise == 'biceps_curl':
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    angle = calculate_angle(shoulder, elbow, wrist)
                    if angle < 90:
                        if stage == 'open':
                            counter += 1
                            print(counter)
                        stage = 'close'
                    elif angle > 135:
                        stage = 'open'
                elif exercise == 'cable_row':
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    # Calculate the angle
                    angle = calculate_angle(shoulder, elbow, wrist)

                    if angle < 100:
                        if stage == 'extend':
                            counter += 1
                            print(counter)
                        stage = 'pull'
                    elif angle > 130:
                        stage = 'extend'
                elif exercise == 'lat_pulldown':
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    # Calculate the angles for both arms
                    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                    # Average the angles of both arms for a more unified analysis
                    angle = (right_angle + left_angle) / 2
                    if angle < 40:
                        if stage == 'extend':  # Transition from extend to pull
                            counter += 1
                            print(counter)  # Print the lat pull-down count
                        stage = 'pull'
                    elif angle > 120:
                        stage = 'extend'
                elif exercise == 'lateral_raise':

                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                    right_angle = calculate_angle(right_shoulder, right_elbow, right_hip)
                    left_angle = calculate_angle(left_shoulder, left_elbow, left_hip)
                    angle = (right_angle + left_angle) / 2

                    if angle < 60:
                        if stage == 'down':
                            counter += 1
                            print(counter)
                        stage = 'up'
                    elif angle > 100:
                        stage = 'down'
                elif exercise == 'abs_crunch':
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    angle = calculate_angle(shoulder, hip, knee)

                    if angle < 70:
                        if stage == 'down':
                            counter += 1
                            print(counter)
                        stage = 'up'
                    elif angle > 70:
                        stage = 'down'
            except:
                pass

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.namedWindow('Exercise Counter', cv2.WINDOW_NORMAL)

            # Annotation
            counter_text = f"Reps: {counter} {stage}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 1
            color = (255, 255, 255)
            background_color = (50, 50, 50)
            margin = 10

            text_width, text_height = cv2.getTextSize(counter_text, font, font_scale, thickness)[0]
            text_x = image.shape[1] - text_width - margin * 3
            text_y = text_height + margin * 2
            position = (text_x, text_y)
            cv2.rectangle(image, (text_x - margin, text_y - text_height - margin),
                          (text_x + text_width + margin, text_y + margin), background_color, -1)

            cv2.putText(image, counter_text, position, font, font_scale, color, thickness)


            image_height, image_width, _ = image.shape

            # Check if the image is vertical
            if image_height > image_width:
                # If vertical, set the width to 500
                stframe.image(image, channels="BGR", width=500)
            else:
                # If horizontal, display it without setting the width
                stframe.image(image, channels="BGR")
            out.write(image)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # with open(output_filename, 'rb') as f:
    #     download_file(f)
    return output_filename

def handle_video(uploaded_video, exercise_aux):
    if not st.session_state.load_state:
        res = video_detect(uploaded_video, exercise_aux)
        st.session_state.load_state = True
        if st.session_state.load_state:
            with open(res, 'rb') as f:
                download_file(f)
    else:
        st.success("Video already processed. Ready for download!")

def download_file(f):
    st.download_button('Download Video', f, file_name='processed_video.mp4')

st.set_page_config(
    page_title="Fitizens Exercise Counter",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# st.title("Welcome to Fitizens Exercise Counter")
st.markdown("<h1 style='text-align: center; color: black;'>Welcome to Fitizens Exercise Counter</h1>", unsafe_allow_html=True)
exercise_radio = st.sidebar.radio('Select Exercise', (
    'Pull-up',       'Squat',
    'Push-up',       'Triceps Pushdown',
    'Bench Press',   'Biceps Curl',
    'Cable Row',     'Lat Pulldown',
    'Lateral Raise', 'Abs Crunch')
)

exercise_aux = exercise_dicc[exercise_radio]



uploaded_video = st.sidebar.file_uploader("Choose a video...", type=["mp4"])


if uploaded_video is not None and not st.session_state.load_state:
    handle_video(uploaded_video, exercise_aux)

