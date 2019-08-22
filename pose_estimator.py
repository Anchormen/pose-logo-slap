"""

@author: Jeroen Vlek <j.vlek@anchormen.nl>
"""

from openpose import pyopenpose


class PoseEstimator(object):
    def __init__(self, model_path, camera):
        op = pyopenpose.WrapperPython()
        op.configure({"model_folder": model_path})
        op.start()
        self.op = op
        self.camera = camera

    def grab_pose(self):
        _, frame = self.camera.read()
        datum = pyopenpose.Datum()
        datum.cvInputData = frame
        self.op.emplaceAndPop([datum])
        return datum
