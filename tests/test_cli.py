import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from PIL import Image


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.img_path = self.test_dir / "test_img.png"
        self.label_path = self.test_dir / "test_img.txt"

        # Create dummy 800x600 image
        img = Image.new("RGB", (800, 600), color=(100, 150, 200))
        img.save(self.img_path)

        # Create dummy dataset config
        self.yaml_path = self.test_dir / "data.yaml"
        with open(self.yaml_path, "w", encoding="utf-8") as f:
            f.write("names:\n  0: person\n  1: car\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_cli(self, args):
        cmd = [".venv/bin/python", "annotation.py"] + args
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res

    def test_cli_grid(self):
        res = self.run_cli(["grid", str(self.img_path), "--cell-size", "200", "--data", str(self.yaml_path)])
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        data = json.loads(res.stdout)
        self.assertEqual(data["width"], 800)
        self.assertEqual(data["height"], 600)
        self.assertEqual(data["cell_size"], 200)
        self.assertTrue(Path(data["grid_image_path"]).exists())
        self.assertEqual(Path(data["grid_image_path"]).parent.name, self.img_path.stem)

    def test_cli_visual(self):
        res = self.run_cli(["visual", str(self.img_path), "car", "100", "100", "300", "300", "--data", str(self.yaml_path)])
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        data = json.loads(res.stdout)
        self.assertEqual(data["candidate"]["pixel_bbox"], [100, 100, 300, 300])
        self.assertTrue(Path(data["visual_image_path"]).exists())
        self.assertEqual(Path(data["visual_image_path"]).parent.name, self.img_path.stem)

    def test_cli_bbox_workflow(self):
        # 1. Add bbox
        res = self.run_cli(["bbox", str(self.img_path), "car", "200", "150", "600", "450", "--data", str(self.yaml_path)])
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        data = json.loads(res.stdout)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["class_id"], 1)

        # Verify .txt file exists
        self.assertTrue(self.label_path.exists())

        # 2. List bbox
        res = self.run_cli(["bbox", str(self.img_path), "--action", "list", "--data", str(self.yaml_path)])
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertEqual(data["total_boxes"], 1)
        self.assertEqual(data["boxes"][0]["class_name"], "car")

        # 3. Delete bbox
        res = self.run_cli(["bbox", str(self.img_path), "--action", "delete", "--index", "0", "--data", str(self.yaml_path)])
        self.assertEqual(res.returncode, 0)

        # 4. Verify list empty
        res = self.run_cli(["bbox", str(self.img_path), "--action", "list", "--data", str(self.yaml_path)])
        data = json.loads(res.stdout)
    def test_cli_zoom(self):
        res = self.run_cli(["zoom", str(self.img_path), "100", "100", "500", "400", "--cell-size", "50", "--data", str(self.yaml_path)])
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        data = json.loads(res.stdout)
        self.assertEqual(data["crop_bbox"], [100, 100, 500, 400])
        self.assertEqual(data["cell_size"], 50)
        self.assertTrue(Path(data["zoom_image_path"]).exists())
        self.assertEqual(Path(data["zoom_image_path"]).parent.name, self.img_path.stem)

    def test_cli_corners(self):
        res = self.run_cli(["corners", str(self.img_path), "150", "120", "450", "380", "--patch-size", "60"])
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        data = json.loads(res.stdout)
        self.assertEqual(data["pixel_bbox"], [150, 120, 450, 380])
        self.assertTrue(Path(data["corners_image_path"]).exists())
        self.assertEqual(Path(data["corners_image_path"]).parent.name, self.img_path.stem)


if __name__ == "__main__":
    unittest.main()
