import unittest

from data_prepare import SENETRawDataBuilder, DataPrepare
from feature_extractors import SENETFeaturePipe


class TestDataPrepare(unittest.TestCase):
    def test_raw_prepare(self):
        rb = SENETRawDataBuilder("test.db")

    def test_feature_building(self):
        rb = SENETRawDataBuilder("test.db")
        pipeline = SENETFeaturePipe()
        data = DataPrepare("test_dataset.pickle", feature_pipe=pipeline, raw_materials=rb.raws, rebuild=True)
