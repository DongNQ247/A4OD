import tempfile
import unittest
from pathlib import Path
from PIL import Image

from src.zoom_renderer import render_zoom_image


class TestZoomRenderer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.img_path = self.test_dir / "sample.png"
        self.out_zoom = self.test_dir / "sample_zoom.png"

        # Create dummy image 1000x800
        img = Image.new("RGB", (1000, 800), color=(150, 180, 210))
        img.save(self.img_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_render_zoom_basic(self):
        res = render_zoom_image(
            image_path=self.img_path,
            output_path=self.out_zoom,
            xmin=200,
            ymin=300,
            xmax=600,
            ymax=700,
            cell_size=50
        )
        self.assertTrue(self.out_zoom.exists())
        self.assertEqual(res["crop_bbox"], [200, 300, 600, 700])
        self.assertEqual(res["crop_width"], 400)
        self.assertEqual(res["crop_height"], 400)
        self.assertEqual(res["cell_size"], 50)

    def test_render_zoom_clamping(self):
        # Coordinates exceeding image boundaries
        res = render_zoom_image(
            image_path=self.img_path,
            output_path=self.out_zoom,
            xmin=-100,
            ymin=-50,
            xmax=1200,
            ymax=900,
            cell_size=100
        )
        self.assertTrue(self.out_zoom.exists())
        self.assertEqual(res["crop_bbox"], [0, 0, 1000, 800])


if __name__ == "__main__":
    unittest.main()
