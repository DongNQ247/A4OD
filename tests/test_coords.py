import unittest
from src.coords import xyxy_pixel_to_yolo_norm, yolo_norm_to_xyxy_pixel


class TestCoords(unittest.TestCase):
    def test_center_box(self):
        # 1000x1000 image, box from (200, 200) to (800, 800)
        # width = 600, height = 600, center = (500, 500)
        xc, yc, w, h = xyxy_pixel_to_yolo_norm(200, 200, 800, 800, 1000, 1000)
        self.assertAlmostEqual(xc, 0.5)
        self.assertAlmostEqual(yc, 0.5)
        self.assertAlmostEqual(w, 0.6)
        self.assertAlmostEqual(h, 0.6)

        # Inverse
        xmin, ymin, xmax, ymax = yolo_norm_to_xyxy_pixel(xc, yc, w, h, 1000, 1000)
        self.assertEqual((xmin, ymin, xmax, ymax), (200, 200, 800, 800))

    def test_reversed_coords(self):
        # Passed xmax first, xmin second
        xc1, yc1, w1, h1 = xyxy_pixel_to_yolo_norm(800, 800, 200, 200, 1000, 1000)
        xc2, yc2, w2, h2 = xyxy_pixel_to_yolo_norm(200, 200, 800, 800, 1000, 1000)
        self.assertEqual((xc1, yc1, w1, h1), (xc2, yc2, w2, h2))

    def test_clamping(self):
        # Outside image bounds [-50, -50] to [1200, 1200]
        xc, yc, w, h = xyxy_pixel_to_yolo_norm(-50, -50, 1200, 1200, 1000, 1000)
        self.assertAlmostEqual(xc, 0.5)
        self.assertAlmostEqual(yc, 0.5)
        self.assertAlmostEqual(w, 1.0)
        self.assertAlmostEqual(h, 1.0)

    def test_zero_area_raises(self):
        with self.assertRaises(ValueError):
            xyxy_pixel_to_yolo_norm(100, 100, 100, 100, 1000, 1000)


if __name__ == "__main__":
    unittest.main()
