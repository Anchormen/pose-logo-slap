"""

@author: Jeroen Vlek <j.vlek@anchormen.nl>
"""

import cv2


def get_camera_streaming(width, height, cam_id=0, fps=30):
    """ Sets up capture with width, height and frames per second parameters """
    capture = cv2.VideoCapture(cam_id)
    # previous cv2 version was opencv-python   3.3.0.10    <pip>
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    if not capture:
        raise Exception("Failed to initialize camera")

    return capture
