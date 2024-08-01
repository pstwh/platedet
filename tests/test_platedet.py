import os
import unittest

import cv2
import numpy as np

from platedet import Platedet


class Testplatedet(unittest.TestCase):
    def setUp(self):
        self.platedet = Platedet()

    def test_platedet(self):
        example_path = os.path.join(os.path.dirname(__file__), "../examples/1.jpg")
        example = cv2.imread(example_path)

        output = self.platedet.inference(example, return_types=["boxes"])

        boxes = output["boxes"]["boxes"]
        result = np.array([[490, 521, 922, 652]])

        dist = np.linalg.norm(boxes - result)

        self.assertTrue(dist < 10)


if __name__ == "__main__":
    unittest.main()
