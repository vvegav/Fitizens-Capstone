import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
import os
import subprocess
css = """
h1 {
    text-align: center;
    display:block;
}
h2 {
    text-align: center;
    display:block;
}
h3 {
    text-align: center;
    display:block;
}
"""
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def mediapipe_process(video):
    cap = cv2.VideoCapture(video)
    counter = 0
    stage = None  # "down" or "up"
    exercise = 'abs_crunch'
    frame_count = 0
    # get the video frames' width and height for proper saving of videos
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    output_video = "output_recorded.mp4"
    out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Ignoring empty camera frame.")
                break

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
            font_scale = 3
            thickness = 5
            color = (255, 255, 255)
            background_color = (50, 50, 50)
            margin = 20

            text_width, text_height = cv2.getTextSize(counter_text, font, font_scale, thickness)[0]
            text_x = image.shape[1] - text_width - margin * 3
            text_y = text_height + margin * 2
            position = (text_x, text_y)
            cv2.rectangle(image, (text_x - margin, text_y - text_height - margin),
                          (text_x + text_width + margin, text_y + margin), background_color, -1)

            cv2.putText(image, counter_text, position, font, font_scale, color, thickness)
            frame_out_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # cv2.imshow('Exercise Counter', image)
            out.write(image)
            if not frame_count % 10:
                yield frame_out_display, None
            frame_count += 1

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    yield None, output_video


# gradio interface
input_video = gr.Video(label="Input Video")
output_frames = gr.Image(label="Output Frames")
output_video_file = gr.Video()
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Interface(
        fn=mediapipe_process,
        inputs=[input_video],
        outputs=[output_frames, output_video_file],
        title=f"Welcome to Fitizens Exercise Counter",
        allow_flagging="never",
        examples=[["../../VIDEOS [TESTING]/abs_crunch.mp4"]],
    )

if __name__ == "__main__":
    interface.queue().launch(debug=True)
