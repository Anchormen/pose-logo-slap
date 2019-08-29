"""

@author: Jeroen Vlek <j.vlek@anchormen.nl>
"""

from openpose import pyopenpose


class PoseEstimator(object):
    def __init__(self, model_path):
        op = pyopenpose.WrapperPython()
        op.configure({"model_folder": model_path})
        op.start()
        self.op = op

    def grab_pose(self, frame):
        datum = pyopenpose.Datum()
        datum.cvInputData = frame
        self.op.emplaceAndPop([datum])
        return datum
