from __future__ import annotations
import os
from collections import Counter
from pathlib import Path

import cv2            # OpenCV
import numpy as np
from PyQt5.QtGui import QPixmap, QImage

def ensure_rgba(img: np.ndarray) -> np.ndarray:
    """Return a copy that always has 4 channels in RGBA order (OpenCV is BGR[A])."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)


def most_common_rgb(pixels: np.ndarray) -> tuple[int, int, int]:
    """Return the RGB triplet that occurs most often in *pixels* (shape = (-1, 3))."""
    rgb_tuples = map(tuple, pixels)
    return Counter(rgb_tuples).most_common(1)[0][0]


def get_real_rect(img_rgba: np.ndarray, bg: tuple[int, int, int] | None = None) -> tuple[int, int, int, int]:
    """
    Compute the tight bounding box around all pixels that are
    (a) **opaque**  _and_  (b) **not** the background colour.
    Returns (x, y, w, h).  If the whole image is background, returns (0,0,0,0).
    """
    h, w, _ = img_rgba.shape
    rgb = img_rgba[:, :, :3]
    alpha = img_rgba[:, :, 3]

    if bg is None:
        bg = most_common_rgb(rgb.reshape(-1, 3))

    mask = (alpha > 0) & np.any(rgb != bg, axis=2)

    if not mask.any():
        return 0, 0, 0, 0

    ys, xs = np.where(mask)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)


def crop_rgba(img_rgba: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    """Return the sub‑image defined by *rect* (x, y, w, h) from RGBA image."""
    x, y, w, h = rect
    return img_rgba[y:y + h, x:x + w].copy()


def centre_on_canvas(img_rgba: np.ndarray, canvas_size: tuple[int, int]) -> np.ndarray:
    """Paste *img_rgba* in the centre of a transparent canvas with *canvas_size* (w, h)."""
    cw, ch = canvas_size
    canvas = np.zeros((ch, cw, 4), dtype=np.uint8)
    h, w, _ = img_rgba.shape
    x = (cw - w) // 2
    y = (ch - h) // 2
    canvas[y:y + h, x:x + w] = img_rgba
    return canvas

def qpixmap_to_numpy(px: QPixmap) -> np.ndarray:
    """
    Convert *px* → np.ndarray in RGBA order (h, w, 4).

    The returned array is a **copy**; modifying it will **not** affect the QPixmap
    and you can safely use it outside the Qt GUI thread.
    """
    if px.isNull():
        raise ValueError("Cannot convert a null QPixmap.")

    fmt = QImage.Format_RGBA8888

    img = px.toImage().convertToFormat(fmt)

    w, h = img.width(), img.height()

    ptr = img.bits()
    ptr.setsize(img.byteCount())

    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4))

    return arr.copy()


def process_folder(folder: Path, show: bool = False) -> None: # used to clean up the sprites for use
    cropped: list[tuple[str, np.ndarray]] = []

    print(f"Scanning {folder} …")
    for file in folder.iterdir():
        if not file.suffix.lower() in {".png", ".bmp", ".tga"}:
            continue

        img_rgba = ensure_rgba(cv2.imread(str(file), cv2.IMREAD_UNCHANGED))

        rect = get_real_rect(img_rgba, bg=tuple(img_rgba[0, 0, :3]))  # use corner pixel as bg
        cropped_img = crop_rgba(img_rgba, rect)
        cropped.append((file.name, cropped_img))

        if show:
            cv2.imshow("cropped preview", cv2.cvtColor(cropped_img, cv2.COLOR_RGBA2BGRA))
            cv2.waitKey(0)

    max_w = max(img.shape[1] for _, img in cropped)
    max_h = max(img.shape[0] for _, img in cropped)
    print(f"Common canvas size: {max_w} × {max_h}")

    for fname, img in cropped:
        centred = centre_on_canvas(img, (max_w, max_h))
        out_path = folder / fname
        cv2.imwrite(str(out_path), cv2.cvtColor(centred, cv2.COLOR_RGBA2BGRA))

        if show:
            cv2.imshow("centred", cv2.cvtColor(centred, cv2.COLOR_RGBA2BGRA))
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("✓ All sprites processed.")
