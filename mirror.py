import os

import argparse
import sys
import cv2
import itertools
from openpose import pyopenpose

TOP_LEFT = (0, 1)
TOP_RIGHT = (2, 1)
BOTTOM_RIGHT = (2, 3)
OPENPOSE_ROOT = os.environ["OPENPOSE_ROOT"]

sys.path.append("./")


class IntelligentMirror:

    def __init__(self, cam_id, w, h, fps, fullscreen, dual, model_path, wname='Big Data Expo Demo'):
        """ Intelligent mirror
        """
        params = dict()
        params["model_folder"] = model_path
        op = pyopenpose.WrapperPython()
        op.configure(params)
        op.start()
        self.op = op

        self.camera = self.get_camera_streaming(cam_id, w, h, fps)
        self.window_name = wname
        self.setup_window(fullscreen, dual)

        self.dual_display = dual

    def run(self):

        # Main loop, continues until CTRL-C is pressed
        for _ in itertools.count():

            _, frame = self.camera.read()
            datum = pyopenpose.Datum()
            datum.cvInputData = frame
            self.op.emplaceAndPop([datum])

            try:
                out = datum.cvOutputData
                # Show on smaller window
                if self.dual_display:
                    out = cv2.resize(out, (960, 540))

                # Show on main window
                cv2.imshow(self.window_name, out)
            except cv2.error as e:
                print(e)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

    @staticmethod
    def get_camera_streaming(cam_id, w, h, fps):
        """ Sets up capture with width, height and frames per second parameters """
        capture = cv2.VideoCapture(cam_id)
        # previous cv2 version was opencv-python   3.3.0.10    <pip>
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        capture.set(cv2.CAP_PROP_FPS, fps)
        if not capture:
            print("Failed to initialize camera")
            sys.exit(1)
        return capture

    def setup_window(self, fullscreen, dual):
        """
        Sets up window. If fullscreen, no exit bar is displayed. If double, additional smaller screen is drawn on the
        left monitor while the fullscreen display is drawn on the right monitor
        """
        # cv2.startWindowThread()
        if fullscreen:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        else:
            cv2.namedWindow(self.window_name)
        cv2.namedWindow(self.window_name)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        if dual:
            # Move is to make sure it's on the right monitor
            cv2.moveWindow(self.window_name, 1920, 0)
            cv2.namedWindow(self.window_name + ' Small View')
            cv2.resizeWindow(self.window_name + ' Small View', 960, 540)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A live emotion recognition from webcam')
    parser.add_argument('--cam_id', default=0, type=int, choices=[0, 1, 2],
                        help='Camera ID, 0 = built-in, 1 = external')
    parser.add_argument('--fps', type=int, default=15, help='Frames per second')
    parser.add_argument('--width', type=int, default=1920, help='Capture and display width')
    parser.add_argument('--height', type=int, default=1080, help='Capture and display height')
    parser.add_argument('--fullscreen', action='store_true', dest='fullscreen',
                        help='If provided, displays in fullscreen')
    parser.add_argument('--dual', action='store_true', dest='dual',
                        help='If provided creates a double display, one for code view and the other for fullscreen'
                             'mirror.')
    parser.add_argument("--model_path", default="/opt/openpose/models/", help="Path to the model directory")
    args = parser.parse_args()

    mind_mirror = IntelligentMirror(args.cam_id, args.width, args.height, args.fps, args.fullscreen, args.dual,
                                    args.model_path)
    mind_mirror.run()
