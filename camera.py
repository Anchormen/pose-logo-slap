"""

@author: Jeroen Vlek <j.vlek@anchormen.nl>
"""

import logging
import queue
import threading

import cv2


def setup_camera_streaming(width, height, cam_id=0, fps=30):
    """ Sets up capture with width, height and frames per second parameters """
    capture = cv2.VideoCapture(cam_id)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    if not capture:
        raise Exception("Failed to initialize camera")

    return capture


class FrameGrabber(threading.Thread):
    """
    Grabs the camera by the frame.
    """

    def __init__(self, width, height, cam_id=0, fps=30):
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.queue = queue.Queue()
        self.camera = setup_camera_streaming(width, height, cam_id, fps)
        self.running = True

    def run(self):
        self.logger.info("Starting FrameGrabber")

        while self.running:
            _, frame = self.camera.read()
            flipped = cv2.flip(frame, 1)
            self.queue.put(flipped)

        self.camera.release()

    def stop(self):
        self.logger.info("Stopping FrameGrabber")
        self.running = False

    def pop_frame(self):
        frame = None
        if not self.queue.empty():
            frame = self.queue.get_nowait()

        return frame

