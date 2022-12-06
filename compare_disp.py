from itertools import product
import os

import cv2
import numpy as np


if __name__ == "__main__":
    phone_type = "p30pro"
    sgbm_root_folder = f"images/{phone_type}"
    anynet_root_folder = f"output/{phone_type}"
    madnet_root_folder = f"../Real-time-self-adaptive-deep-stereo/output/{phone_type}"
    root_output_folder = "output/compare_disp"

    depth_folders = ["0.5", "1", "3", "5"]
    modes = ["disparity_static", "disparity_slow", "disparity_fast"]

    for depth_folder, mode in product(depth_folders, modes):
        sgbm_folder = os.path.join(sgbm_root_folder, depth_folder, mode)
        anynet_folder = os.path.join(anynet_root_folder, depth_folder, mode)
        madnet_folder = os.path.join(madnet_root_folder, depth_folder, mode)
        output_folder = os.path.join(root_output_folder, depth_folder, mode)
        os.makedirs(output_folder, exist_ok=True)

        sgbm_disparity_results = sorted([os.path.join(sgbm_folder, i) for i in os.listdir(sgbm_folder)])
        anynet_disparity_results = sorted([os.path.join(anynet_folder, i) for i in os.listdir(anynet_folder)])
        madnet_disparity_results = sorted([os.path.join(madnet_folder, i) for i in os.listdir(madnet_folder)])

        assert len(sgbm_disparity_results) == len(anynet_disparity_results) == len(madnet_disparity_results)
        for sgbm_disparity_result, anynet_disparity_result, madnet_disparity_result in \
                zip(sgbm_disparity_results, anynet_disparity_results, madnet_disparity_results):
            sgbm_disp = np.load(sgbm_disparity_result)
            sgbm_disp = np.clip(sgbm_disp, -16, None)

            anynet_disp = np.load(anynet_disparity_result)
            madnet_disp = np.load(madnet_disparity_result)

            delta_sgbm_anynet = np.mean(np.abs(sgbm_disp - anynet_disp))
            delta_sgbm_madnet = np.mean(np.abs(sgbm_disp - madnet_disp))
            delta_anynet_madnet = np.mean(np.abs(anynet_disp - madnet_disp))
            print(f"{os.path.basename(sgbm_disparity_result)} | {delta_sgbm_anynet} | {delta_sgbm_madnet} | {delta_anynet_madnet}")

            sgbm_disp_vis = 255 - cv2.normalize(sgbm_disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            sgbm_disp_vis = cv2.applyColorMap(sgbm_disp_vis, cv2.COLORMAP_TURBO)
            anynet_disp_vis = 255 - cv2.normalize(anynet_disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            anynet_disp_vis = cv2.applyColorMap(anynet_disp_vis, cv2.COLORMAP_TURBO)
            madnet_disp_vis = 255 - cv2.normalize(madnet_disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            madnet_disp_vis = cv2.applyColorMap(madnet_disp_vis, cv2.COLORMAP_TURBO)

            output_path = os.path.join(output_folder, os.path.basename(sgbm_disparity_result).replace("_sgbm.npy", ".png"))
            cv2.imwrite(output_path, np.hstack([sgbm_disp, anynet_disp, madnet_disp]))
            output_vis_path = os.path.join(output_folder, os.path.basename(sgbm_disparity_result).replace("_sgbm.npy", "_vis.png"))
            cv2.imwrite(output_vis_path, np.hstack([sgbm_disp_vis, anynet_disp_vis, madnet_disp_vis]))
