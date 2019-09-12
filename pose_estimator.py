"""

@author: Jeroen Vlek <j.vlek@anchormen.nl>
"""

from openpose import pyopenpose


class PoseEstimator(object):
    def __init__(self, model_path, net_resolution):
        op = pyopenpose.WrapperPython()

        params = dict()
        params["model_folder"] = model_path
        params["face"] = False
        params["body"] = 1
        params["render_pose"] = 1
        params["net_resolution"] = net_resolution

        op.configure(params)
        op.start()
        self.op = op

    def grab_pose(self, frame):
        datum = pyopenpose.Datum()
        datum.cvInputData = frame
        self.op.emplaceAndPop([datum])
        return datum
