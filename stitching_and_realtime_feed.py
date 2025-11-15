import cv2
import numpy as np
import time
import os

LEFT_ID = 0
RIGHT_ID = 1
ORB_FEATURES = 3000
TOP_MATCHES = 300
BLEND_POWER = 1.0

SAVE_DIR = "stitched_images"
os.makedirs(SAVE_DIR, exist_ok=True)


def set_cam_size(cam, w, h):
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    try:
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except:
        pass



def compute_homography_from_frames(left_frame, right_frame, n_features=ORB_FEATURES, top_matches=TOP_MATCHES):
    grayL = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(n_features)
    kp1, des1 = orb.detectAndCompute(grayL, None)
    kp2, des2 = orb.detectAndCompute(grayR, None)

    if des1 is None or des2 is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 8:
        return None

    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:min(len(matches), top_matches)]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    return H


def find_overlap_bounds(mask_left_padded, mask_right):
    overlap = cv2.bitwise_and(mask_left_padded, mask_right)
    ys, xs = np.where(overlap > 0)
    if len(xs) == 0:
        return None, None
    return int(xs.min()), int(xs.max())



def blend_overlap(left_frame, warped_right, x_min, x_max, blend_power=BLEND_POWER):
    stitched = warped_right.copy()
    h, w = left_frame.shape[:2]

    stitched[:, :w] = left_frame

    if x_max <= x_min:
        return stitched

    width = x_max - x_min
    alpha = np.linspace(0.0, 1.0, width, dtype=np.float32)
    if blend_power != 1.0:
        alpha = np.power(alpha, blend_power)

    for i in range(x_min, x_max):
        a = float(alpha[i - x_min])

        L = left_frame[:, min(i, w - 1)].astype(np.float32)
        R = warped_right[:, i].astype(np.float32)
        stitched[:, i] = (L * (1 - a) + R * a).astype(np.uint8)

    return stitched


def main():

    # fullscreen window
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    screen_width = 1920
    screen_height = 1080

    left_cam = cv2.VideoCapture(LEFT_ID)
    right_cam = cv2.VideoCapture(RIGHT_ID)

    if not left_cam.isOpened() or not right_cam.isOpened():
        print("Camera error")
        return

    print("Capturing frames to compute homography...")
    retL, left_sample = left_cam.read()
    retR, right_sample = right_cam.read()

    left_sample = cv2.flip(left_sample, 1)
    right_sample = cv2.flip(right_sample, 1)

    H = compute_homography_from_frames(left_sample, right_sample)
    if H is None:
        print("Homography computation failed")
        return

    print("Homography matrix computed.")

    frame_count = 0

    while True:
        retL, frameL = left_cam.read()
        retR, frameR = right_cam.read()
        if not retL or not retR:
            break

        frameL = cv2.flip(frameL, 1)
        frameR = cv2.flip(frameR, 1)

        # natural dims
        hL, wL = frameL.shape[:2]
        hR, wR = frameR.shape[:2]

        # warp right frame
        canvas_w = wL + wR + 400  
        warped_right = cv2.warpPerspective(frameR, H, (canvas_w, max(hL, hR)))

        # create stitched
        mask_left = (cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
        mask_left_padded = np.zeros_like(warped_right[:, :, 0])
        mask_left_padded[:, :wL] = mask_left

        mask_right = (cv2.cvtColor(warped_right, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)

        x_min, x_max = find_overlap_bounds(mask_left_padded, mask_right)

        if x_min is None:
            stitched = warped_right.copy()
            stitched[:, :wL] = frameL
        else:
            stitched = blend_overlap(frameL, warped_right, x_min, x_max)

        #here we are bulding the canvas for putting in the three images in one frame for comparison
        canvas = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255

        # top row = stitched
        sh, sw = stitched.shape[:2]
        stitched_resized = stitched.copy()
        stitched_resized = cv2.resize(stitched_resized, (screen_width, int(screen_height * 0.6)))

        top_h = stitched_resized.shape[0]
        canvas[0:top_h, :] = stitched_resized

        # bottom row
        bottom_y = top_h
        left_resized = cv2.resize(frameL, (screen_width // 2, screen_height - top_h))
        right_resized = cv2.resize(frameR, (screen_width // 2, screen_height - top_h))

        canvas[bottom_y:, 0:screen_width // 2] = left_resized
        canvas[bottom_y:, screen_width // 2:] = right_resized

        cv2.imshow("Output", canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('s'):
            cv2.imwrite(f"{SAVE_DIR}/stitched_{frame_count}.jpg", stitched)
            print("Saved stitched image:", frame_count)
            frame_count += 1

        if key == ord('h'):
            print("Recomputing homography...")
            H2 = compute_homography_from_frames(frameL, frameR)
            if H2 is not None:
                H = H2
                print("Updated homography.")
            else:
                print("Failed, keeping old H")

    left_cam.release()
    right_cam.release()
    cv2.destroyAllWindows()


main()
