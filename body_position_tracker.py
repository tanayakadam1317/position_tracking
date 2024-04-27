
import mediapipe as mp
import cv2
import numpy as np
import time
# import math
from reportlab.pdfgen import canvas
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter




mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Initialize variables
overlays = []
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
follow_through_done = 0
follow_through_not_done = 0
monitoring_follow_through = False
baseline_pose = None
start_time = None
smoothed_landmarks = []


# Smoothing parameters
SMOOTHING_ALPHA = 0.9


def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point


    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)


    if angle > 180.0:
        angle = 360 - angle


    return angle


cap = cv2.VideoCapture(0)
cv2.namedWindow('Mediapipe Feed', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Mediapipe Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)  # Flip horizontally to mirror the image
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


            # Smooth the landmark positions
            landmark_positions = np.array([[landmark.x, landmark.y] for landmark in results.pose_landmarks.landmark])
            if len(smoothed_landmarks) == 0:
                smoothed_landmarks = landmark_positions
            else:
                smoothed_landmarks = SMOOTHING_ALPHA * landmark_positions + (1 - SMOOTHING_ALPHA) * smoothed_landmarks


            for i, landmark in enumerate(results.pose_landmarks.landmark):
                landmark.x = smoothed_landmarks[i][0]
                landmark.y = smoothed_landmarks[i][1]


            # Calculate angles at the joints
            if len(results.pose_landmarks.landmark) >= 3:
                angle_left_shoulder = calculate_angle([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
                                                       [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                                                       [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y])


                angle_left_elbow = calculate_angle([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y],
                                                   [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
                                                   [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])


                angle_right_shoulder = calculate_angle([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
                                                        [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                                                        [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y])


                angle_right_elbow = calculate_angle([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y],
                                                    [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
                                                    [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
               
                angle_left_hip = calculate_angle([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
                                                [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                                                [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])


                angle_right_hip = calculate_angle([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
                                                [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                                                [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])






                # Display angles next to the joints
                cv2.putText(image, f"{int(angle_left_shoulder)}", (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image.shape[1]),
                                                                  int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image.shape[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.putText(image, f"{int(angle_left_elbow)}", (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image.shape[1]),
                                                                int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image.shape[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.putText(image, f"{int(angle_right_shoulder)}", (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * image.shape[1]),
                                                                   int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * image.shape[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.putText(image, f"{int(angle_right_elbow)}", (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * image.shape[1]),
                                                                int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * image.shape[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
               
                cv2.putText(image, f"{int(angle_left_hip)}", (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x * image.shape[1]),
                                                                int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y * image.shape[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
               
                cv2.putText(image, f"{int(angle_right_hip)}", (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x * image.shape[1]),
                                                                int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y * image.shape[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)


        if cv2.waitKey(1) & 0xFF == 13:  # Enter key is pressed
            overlays.append((results.pose_landmarks, colors[len(overlays) % len(colors)]))
            baseline_pose = results.pose_landmarks
            start_time = time.time()
            monitoring_follow_through = True
            pose_change_detected = False  # Initialize pose change detection


        # Logic to detect significant pose changes
        if monitoring_follow_through and baseline_pose:
            current_time = time.time()
            if current_time - start_time <= 5:
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    baseline_landmark = baseline_pose.landmark[i]
                    if (abs(landmark.x - baseline_landmark.x) > 0.1 or abs(landmark.y - baseline_landmark.y) > 0.1):
                        pose_change_detected = True
                        break
                if pose_change_detected:
                    follow_through_not_done += 1
                    monitoring_follow_through = False  # Stop monitoring
            else:
                # If 5 seconds pass without significant pose change, increment "done" counter
                if not pose_change_detected:
                    follow_through_done += 1
                monitoring_follow_through = False  # Reset for next check


        # Display overlays
        for idx, (overlay_landmarks, color) in enumerate(overlays):
            if overlay_landmarks:
                mp_drawing.draw_landmarks(
                    image, overlay_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
                )


        # Display the follow-through counters
        cv2.putText(image, f"FT Done: {follow_through_done}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.putText(image, f"FT Not Done: {follow_through_not_done}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)


        cv2.imshow('Mediapipe Feed', image)


        # Stop the script by pressing 'e' key
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

        # Create a PDF report
        def create_pdf(filename, follow_through_done, follow_through_not_done):
            c = canvas.Canvas(filename)
            c.drawString(100, 800, "Follow-Through Done: {}".format(follow_through_done))
            c.drawString(100, 780, "Follow-Through Not Done: {}".format(follow_through_not_done))
            c.save()

        # Place this at the end of your code, after releasing the VideoWriter and destroying the OpenCV windows
        pdf_filename = 'report.pdf'
        create_pdf(pdf_filename, follow_through_done, follow_through_not_done)
        print("PDF report generated: {}".format(pdf_filename))




        # Function to create PDF report
        def create_pdf(pdf_filename, project_name, follow_through_done, follow_through_not_done, num_shots):
            # Calculate percentage of consistency
            total_shots = follow_through_done + follow_through_not_done
            consistency_percentage = (follow_through_done / total_shots) * 100 if total_shots > 0 else 0

            # Create a PDF document
            c = canvas.Canvas(pdf_filename, pagesize=letter)

            # Set font size and center align the project name
            c.setFont("Helvetica", 26)
            text_width = c.stringWidth(project_name, "Helvetica", 26)
            c.drawString((letter[0] - text_width) / 2, 750, project_name)

            # Add follow-through information
            c.setFont("Helvetica", 12)
            c.drawString(100, 730, f"Follow-throughs: {follow_through_done}")
            c.drawString(100, 710, f"Follow-throughs not done: {follow_through_not_done}")

            # Add total number of shots
            c.drawString(100, 690, f"Total shots: {num_shots}")

            # Add percentage of consistency
            c.drawString(100, 670, f"Consistency: {consistency_percentage:.2f}%")

            # Add border to the whole page
            c.rect(0, 0, letter[0], letter[1])

            # Save the PDF file
            c.save()

        # Usage
        project_name = "Position Tracking for Shooters"
        pdf_filename = "report.pdf"
        num_shots = 60  # Example total number of shots
        create_pdf(pdf_filename, project_name, follow_through_done, follow_through_not_done, num_shots)



cap.release()
cv2.destroyAllWindows()