import tempfile
import unittest
from pathlib import Path
from PIL import Image

from src.corner_inspector import render_corner_inspection


class TestCornerInspector(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.img_path = self.test_dir / "sample.png"
        self.out_corners = self.test_dir / "sample_corners.png"

        img = Image.new("RGB", (1000, 800), color=(100, 120, 140))
        img.save(self.img_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_render_corner_inspection(self):
        res = render_corner_inspection(
            image_path=self.img_path,
            output_path=self.out_corners,
            xmin=300,
            ymin=200,
            xmax=600,
            ymax=500,
            patch_size=60
        )
        self.assertTrue(self.out_corners.exists())
        self.assertEqual(res["pixel_bbox"], [300, 200, 600, 500])
        self.assertEqual(res["box_width"], 300)
        self.assertEqual(res["box_height"], 300)
        self.assertEqual(res["patch_size"], 60)


if __name__ == "__main__":
    unittest.main()
