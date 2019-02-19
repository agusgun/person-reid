from app.deep_sort import nn_matching
from app.deep_sort.tools import generate_detections as gdet
from app.deep_sort.tracker import Tracker
import os


class Tracking:
    def __init__(self):
        max_cosine_distance = 0.3
        nn_budget = None
        package_path = os.path.abspath(os.path.dirname(__file__))

        model_filename = os.path.join(package_path, '../model_data/mars-small128.pb')
        self.encoder = gdet.create_box_encoder(model_filename)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)
