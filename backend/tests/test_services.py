import unittest

from app.services.georeference import map_to_pixel, pixel_to_map
from app.services.geometry import mask_to_polygon_features
from app.services.topology import validate_features


class ServiceTests(unittest.TestCase):
    def test_georeference_round_trip(self):
        transform = (2.0, 0.0, 100.0, 0.0, -2.0, 200.0)
        mapped = pixel_to_map(10.0, 15.0, transform)
        self.assertEqual(mapped, (120.0, 170.0))
        self.assertEqual(map_to_pixel(*mapped, transform), (10.0, 15.0))

    def test_polygon_feature_is_closed(self):
        features = mask_to_polygon_features([[[0, 0], [10, 0], [10, 10]]], 100, 100)
        ring = features[0]["geometry"]["coordinates"][0]
        self.assertEqual(ring[0], ring[-1])

    def test_topology_accepts_polygon_feature(self):
        features = mask_to_polygon_features([[[0, 0], [10, 0], [10, 10]]], 100, 100)
        result = validate_features(features)
        self.assertTrue(result["valid"])
        self.assertEqual(result["feature_count"], 1)


if __name__ == "__main__":
    unittest.main()
