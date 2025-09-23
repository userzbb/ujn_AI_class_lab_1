import argparse
import math
from typing import List, Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp


# 将归一化坐标转换为像素坐标
def _to_pixel_coords(landmarks_norm, width: int, height: int) -> List[Tuple[int, int]]:
    pts = []
    for lm in landmarks_norm:
        x = min(max(int(lm.x * width), 0), width - 1)
        y = min(max(int(lm.y * height), 0), height - 1)
        pts.append((x, y))
    return pts


# 计算左右对称度分数（0~1），1为最对称
def _symmetry_score(pts: List[Tuple[int, int]]) -> float:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    face_w = max(1, max_x - min_x)
    center_x = (min_x + max_x) / 2.0

    left = [(x, y) for (x, y) in pts if x < center_x]
    right = [(x, y) for (x, y) in pts if x >= center_x]

    if len(left) < 5 or len(right) < 5:
        return 0.5  # 兜底

    # 按y排序，对应行进行左右比较
    left.sort(key=lambda p: p[1])
    right.sort(key=lambda p: p[1])
    k = min(len(left), len(right))
    diffs = []
    for i in range(k):
        xl, yl = left[i]
        xr, yr = right[i]
        dx_l = center_x - xl
        dx_r = xr - center_x
        diffs.append(abs(dx_l - dx_r) / face_w)
    mean_diff = float(np.mean(diffs)) if diffs else 0.5
    # 差异越小越好，经验缩放
    score = max(0.0, 1.0 - mean_diff * 3.0)
    return float(min(1.0, score))


# 计算脸宽高比例分数（0~1），目标比例约0.75
def _ratio_score_face_wh(pts: List[Tuple[int, int]]) -> float:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w = max(1, max(xs) - min(xs))
    h = max(1, max(ys) - min(ys))
    ratio = w / h
    target = 0.75
    diff_norm = abs(ratio - target) / target
    return float(max(0.0, 1.0 - diff_norm))


# 估计双眼中心（基于FaceMesh refine_landmarks下的虹膜点）
# 返回((xL,yL),(xR,yR))，若不可用返回None
def _eye_centers_from_iris(landmarks_norm, width: int, height: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    # MediaPipe FaceMesh 在 refine_landmarks=True 时，通常包含：
    # 右眼虹膜: 469,470,471,472；左眼虹膜: 474,475,476,477
    idx_right = [469, 470, 471, 472]
    idx_left = [474, 475, 476, 477]
    total = len(landmarks_norm)
    if max(idx_right + idx_left) >= total:
        return None
    # 取质心作为眼睛中心
    def centroid(idxs):
        xs, ys = [], []
        for i in idxs:
            xs.append(landmarks_norm[i].x * width)
            ys.append(landmarks_norm[i].y * height)
        return (int(np.mean(xs)), int(np.mean(ys)))
    cR = centroid(idx_right)
    cL = centroid(idx_left)
    return (cL, cR)


# 计算双眼间距比例分数（0~1），以脸宽归一化，目标约0.46
def _ratio_score_eye_distance(eyeL: Tuple[int, int], eyeR: Tuple[int, int], pts: List[Tuple[int, int]]) -> float:
    xs = [p[0] for p in pts]
    w = max(1, max(xs) - min(xs))
    dx = math.dist(eyeL, eyeR)
    ratio = dx / w
    target = 0.46
    diff_norm = abs(ratio - target) / target
    return float(max(0.0, 1.0 - diff_norm))


# 综合打分（0~100）
def beauty_score(landmarks_norm, width: int, height: int) -> int:
    pts = _to_pixel_coords(landmarks_norm, width, height)

    s_sym = _symmetry_score(pts)
    s_wh = _ratio_score_face_wh(pts)

    eye_centers = _eye_centers_from_iris(landmarks_norm, width, height)
    if eye_centers is not None:
        s_eye = _ratio_score_eye_distance(eye_centers[0], eye_centers[1], pts)
        # 加权（可按需调整）
        score01 = 0.5 * s_sym + 0.25 * s_wh + 0.25 * s_eye
    else:
        score01 = 0.65 * s_sym + 0.35 * s_wh

    return int(np.clip(score01 * 100.0, 0, 100))


def draw_points_and_lines(image, face_landmarks, draw_tesselation=True, draw_points=True):
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    if draw_tesselation:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
        )
    # 关键点圆点
    if draw_points:
        h, w = image.shape[:2]
        for lm in face_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 1, (0, 255, 0), -1)


def process_frame(frame_bgr, face_mesh) -> Tuple[np.ndarray, Optional[int]]:
    image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    out = frame_bgr.copy()

    if not results.multi_face_landmarks:
        return out, None

    # 仅使用第一张脸
    face_landmarks = results.multi_face_landmarks[0]
    draw_points_and_lines(out, face_landmarks, draw_tesselation=True, draw_points=True)

    score = beauty_score(face_landmarks.landmark, out.shape[1], out.shape[0])

    # 注意：OpenCV默认不支持中文字体，避免乱码，这里使用英文标注
    cv2.rectangle(out, (10, 10), (220, 50), (0, 0, 0), -1)
    cv2.putText(out, f"Score: {score}/100", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

    # 若能检出虹膜则标记双眼中心
    eye_centers = _eye_centers_from_iris(face_landmarks.landmark, out.shape[1], out.shape[0])
    if eye_centers:
        cv2.circle(out, eye_centers[0], 3, (255, 0, 0), -1)
        cv2.circle(out, eye_centers[1], 3, (255, 0, 0), -1)
        cv2.line(out, eye_centers[0], eye_centers[1], (255, 0, 0), 1)

    return out, score


def run_camera(device: int = 0):
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out, score = process_frame(frame, face_mesh)
            cv2.imshow("FaceMesh + Score (press q to quit)", out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


def run_image(path: str, save: Optional[str] = None):
    mp_face_mesh = mp.solutions.face_mesh
    img = cv2.imread(path)
    if img is None:
        print(f"无法读取图片: {path}")
        return
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        out, score = process_frame(img, face_mesh)
        cv2.imshow("FaceMesh + Score (press any key to close)", out)
        if save:
            cv2.imwrite(save, out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video(path: str):
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"无法打开视频: {path}")
        return
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out, score = process_frame(frame, face_mesh)
            cv2.imshow("FaceMesh + Score (press q to quit)", out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    ap = argparse.ArgumentParser(description="人脸关键点可视化与简易颜值评分（MediaPipe FaceMesh）")
    ap.add_argument("--image", type=str, default=None, help="图片路径")
    ap.add_argument("--video", type=str, default=None, help="视频路径")
    ap.add_argument("--camera", type=int, default=None, help="摄像头设备号，默认0")
    ap.add_argument("--save", type=str, default=None, help="保存输出图片路径（仅image模式）")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.image:
        run_image(args.image, args.save)
    elif args.video:
        run_video(args.video)
    else:
        cam_id = 0 if args.camera is None else args.camera
        run_camera(cam_id)


if __name__ == "__main__":
    main()

