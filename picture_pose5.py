import cv2
import mediapipe as mp
import sys
import csv
import os

def process_image(image_path, mark_positions=False):
    # Mediapipeの初期化
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # 画像の読み込み
    image = cv2.imread(image_path)

    detected_points = []

    if mark_positions:
        # 事前に描画された赤い丸を検出するための設定
        red_lower = (0, 0, 150)
        red_upper = (50, 50, 255)
        mask = cv2.inRange(image, red_lower, red_upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 検出された赤い丸の中心を計算
        for contour in contours:
            if cv2.contourArea(contour) > 10:  # ノイズを除去するための閾値
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    detected_points.append((cX, cY))

        # 検出されたポイントを描画
        for idx, point in enumerate(detected_points):
            cv2.circle(image, point, 5, (0, 255, 0), -1)
            cv2.putText(image, f"Mark_{idx}", (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mediapipe Poseの初期化
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        # 画像をRGBに変換
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # ポーズの検出
        results = pose.process(image_rgb)

        # 検出されたポーズを描画
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 座標をCSVファイルに書き出し
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            output_csv = os.path.join(os.path.dirname(image_path), f"{name}.csv")
            with open(output_csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Joint", "Index", "X", "Y"])
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    writer.writerow([mp_pose.PoseLandmark(idx).name, idx, landmark.x, landmark.y])

            # 画像を保存
            output_image_path = os.path.join(os.path.dirname(image_path), f"p_{base_name}")
            cv2.imwrite(output_image_path, image)

            # 結果の表示
            cv2.imshow('Pose Detection', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            if mark_positions and detected_points:
                print("Pose could not be detected, but red marks were found.")
                # 座標をCSVファイルに書き出し
                base_name = os.path.basename(image_path)
                name, ext = os.path.splitext(base_name)
                output_csv = os.path.join(os.path.dirname(image_path), f"{name}.csv")
                with open(output_csv, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Joint", "X", "Y"])
                    for idx, point in enumerate(detected_points):
                        writer.writerow([f"Mark_{idx}", point[0], point[1]])

                # 画像を保存
                output_image_path = os.path.join(os.path.dirname(image_path), f"p_{base_name}")
                cv2.imwrite(output_image_path, image)

                # 結果の表示
                cv2.imshow('Pose Detection', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Pose could not be detected in the image.")
                return

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python script_name.py path_to_image [-p]")
    else:
        image_path = sys.argv[1]
        mark_positions = len(sys.argv) == 3 and sys.argv[2] == '-p'
        process_image(image_path, mark_positions)

