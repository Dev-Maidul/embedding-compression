"""Part A validation: compare our from-scratch NumPy implementations
against OpenCV, on real images, and write the results to a CSV.

This script produces results/a_validation.csv. Every number in that file
comes from running this script; nothing here is typed by hand into README
or report tables (those are generated from this CSV by report/make_tables.py).
"""

import time

import cv2
import numpy as np
import pandas as pd
from skimage import data as skimage_data

from src.ip.canny import canny_edge_detector
from src.ip.convolution import correlate2d
from src.ip.gaussian import gaussian_blur, gaussian_kernel
from src.ip.histeq import histogram_equalize
from src.ip.sobel import sobel_gradients

SEED = 42
GAUSSIAN_SIZE = 5
GAUSSIAN_SIGMA = 1.4
CANNY_LOW = 0.1
CANNY_HIGH = 0.2


def load_validation_images():
    """A handful of skimage.data sample images (no download needed),
    converted to grayscale float64 in [0, 255]."""
    names_and_arrays = {
        "camera": skimage_data.camera(),
        "astronaut_gray": cv2.cvtColor(skimage_data.astronaut(), cv2.COLOR_RGB2GRAY),
        "checkerboard": skimage_data.checkerboard(),
        "coins": skimage_data.coins(),
    }
    return {name: arr.astype(np.float64) for name, arr in names_and_arrays.items()}


def validate_filter2d(images: dict) -> list[dict]:
    rows = []
    kernel = gaussian_kernel(GAUSSIAN_SIZE, GAUSSIAN_SIGMA)
    for name, img in images.items():
        t0 = time.perf_counter()
        ours = correlate2d(img, kernel, pad_mode="reflect")
        t_ours = time.perf_counter() - t0

        t0 = time.perf_counter()
        ref = cv2.filter2D(img, cv2.CV_64F, kernel, borderType=cv2.BORDER_REFLECT_101)
        t_ref = time.perf_counter() - t0

        err = np.abs(ours - ref)
        rows.append({
            "check": "filter2D_vs_correlate2d",
            "image": name,
            "image_shape": f"{img.shape[0]}x{img.shape[1]}",
            "max_abs_error": float(err.max()),
            "mean_abs_error": float(err.mean()),
            "time_ours_sec": t_ours,
            "time_opencv_sec": t_ref,
        })
    return rows


def validate_gaussian_blur(images: dict) -> list[dict]:
    rows = []
    for name, img in images.items():
        t0 = time.perf_counter()
        ours = gaussian_blur(img, size=GAUSSIAN_SIZE, sigma=GAUSSIAN_SIGMA, pad_mode="reflect")
        t_ours = time.perf_counter() - t0

        t0 = time.perf_counter()
        ref = cv2.GaussianBlur(img, (GAUSSIAN_SIZE, GAUSSIAN_SIZE), sigmaX=GAUSSIAN_SIGMA,
                                borderType=cv2.BORDER_REFLECT_101)
        t_ref = time.perf_counter() - t0

        err = np.abs(ours - ref)
        rows.append({
            "check": "GaussianBlur_vs_gaussian_blur",
            "image": name,
            "image_shape": f"{img.shape[0]}x{img.shape[1]}",
            "max_abs_error": float(err.max()),
            "mean_abs_error": float(err.mean()),
            "time_ours_sec": t_ours,
            "time_opencv_sec": t_ref,
        })
    return rows


def validate_sobel(images: dict) -> list[dict]:
    rows = []
    for name, img in images.items():
        t0 = time.perf_counter()
        gx, gy, mag, _ = sobel_gradients(img, pad_mode="reflect")
        t_ours = time.perf_counter() - t0

        t0 = time.perf_counter()
        ref_gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
        ref_gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
        ref_mag = np.sqrt(ref_gx ** 2 + ref_gy ** 2)
        t_ref = time.perf_counter() - t0

        err_mag = np.abs(mag - ref_mag)
        rows.append({
            "check": "Sobel_vs_sobel_gradients",
            "image": name,
            "image_shape": f"{img.shape[0]}x{img.shape[1]}",
            "max_abs_error": float(err_mag.max()),
            "mean_abs_error": float(err_mag.mean()),
            "time_ours_sec": t_ours,
            "time_opencv_sec": t_ref,
        })
    return rows


def validate_histeq(images: dict) -> list[dict]:
    rows = []
    for name, img in images.items():
        img_u8 = np.clip(np.round(img), 0, 255).astype(np.uint8)

        t0 = time.perf_counter()
        ours = histogram_equalize(img_u8)
        t_ours = time.perf_counter() - t0

        t0 = time.perf_counter()
        ref = cv2.equalizeHist(img_u8)
        t_ref = time.perf_counter() - t0

        err = np.abs(ours.astype(int) - ref.astype(int))
        rows.append({
            "check": "equalizeHist_vs_histogram_equalize",
            "image": name,
            "image_shape": f"{img.shape[0]}x{img.shape[1]}",
            "max_abs_error": float(err.max()),
            "mean_abs_error": float(err.mean()),
            "time_ours_sec": t_ours,
            "time_opencv_sec": t_ref,
            "pixel_agreement": float(np.mean(err == 0)),
        })
    return rows


def validate_canny(images: dict) -> list[dict]:
    """Canny edge maps are binary, and our from-scratch pipeline (sector-
    quantized NMS, fraction-of-max thresholds) is not numerically identical
    to cv2.Canny's (interpolated NMS, absolute gradient thresholds), so we
    do not expect a pixel-perfect match. We report pixel agreement (what
    fraction of pixels get the same edge/non-edge decision) rather than a
    max abs error, which is the honest way to compare two different, valid
    implementations of the same algorithm family."""
    rows = []
    for name, img in images.items():
        t0 = time.perf_counter()
        ours = canny_edge_detector(img, sigma=GAUSSIAN_SIGMA, gaussian_size=GAUSSIAN_SIZE,
                                    low_threshold=CANNY_LOW, high_threshold=CANNY_HIGH,
                                    pad_mode="reflect")
        t_ours = time.perf_counter() - t0

        img_u8 = np.clip(np.round(img), 0, 255).astype(np.uint8)
        blurred_u8 = cv2.GaussianBlur(img_u8, (GAUSSIAN_SIZE, GAUSSIAN_SIZE), sigmaX=GAUSSIAN_SIGMA)
        # cv2.Canny takes absolute gradient thresholds; Sobel magnitude on
        # uint8 images with ksize=3 is typically a few hundred at edges,
        # so use thresholds in that same ballpark rather than our 0-1
        # fractions (which are relative to this image's own max gradient).
        t0 = time.perf_counter()
        ref = cv2.Canny(blurred_u8, threshold1=50, threshold2=150) > 0
        t_ref = time.perf_counter() - t0

        agreement = float(np.mean(ours == ref))
        disagreement = np.abs(ours.astype(int) - ref.astype(int))
        rows.append({
            "check": "Canny_vs_canny_edge_detector",
            "image": name,
            "image_shape": f"{img.shape[0]}x{img.shape[1]}",
            "max_abs_error": float(disagreement.max()),
            "mean_abs_error": float(disagreement.mean()),
            "time_ours_sec": t_ours,
            "time_opencv_sec": t_ref,
            "pixel_agreement": agreement,
        })
    return rows


def main():
    np.random.seed(SEED)
    images = load_validation_images()

    rows = []
    rows += validate_filter2d(images)
    rows += validate_gaussian_blur(images)
    rows += validate_sobel(images)
    rows += validate_histeq(images)
    rows += validate_canny(images)

    df = pd.DataFrame(rows)
    out_path = "results/a_validation.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
