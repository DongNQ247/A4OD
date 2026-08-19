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

    def run_a4od(self, args):
        cmd = ["./a4od"] + args
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
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["candidate"]["pixel_bbox"], [100, 100, 300, 300])
        self.assertTrue(Path(data["visual_image_path"]).exists())
        self.assertEqual(Path(data["visual_image_path"]).parent.name, self.img_path.stem)

    def test_cli_visual_crop_context_and_reversed_bbox(self):
        res = self.run_cli(["visual", str(self.img_path), "car", "300", "300", "100", "100", "--crop-context", "20", "--data", str(self.yaml_path)])
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        data = json.loads(res.stdout)
        self.assertEqual(data["candidate"]["pixel_bbox"], [100, 100, 300, 300])
        self.assertEqual(data["view_bbox"], [80, 80, 320, 320])
        self.assertEqual(data["view_size"], [240, 240])

    def test_cli_visual_rejects_unknown_class(self):
        res = self.run_cli(["visual", str(self.img_path), "airplane", "100", "100", "300", "300", "--data", str(self.yaml_path)])
        self.assertNotEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error"]["code"], "UNKNOWN_CLASS")

    def test_cli_bbox_workflow(self):
        # 1. Verify candidate, then add with verification id.
        res = self.run_cli(["verify", str(self.img_path), "car", "200", "150", "600", "450", "--data", str(self.yaml_path)])
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        verify_data = json.loads(res.stdout)
        verification_id = verify_data["verification_id"]

        res = self.run_cli(["bbox", str(self.img_path), "car", "200", "150", "600", "450", "--verification-id", verification_id, "--data", str(self.yaml_path)])
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        data = json.loads(res.stdout)
        self.assertTrue(data["ok"])
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
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertEqual(data["total_boxes"], 0)

    def test_cli_bbox_dry_run_does_not_write(self):
        res = self.run_cli(["bbox", str(self.img_path), "car", "200", "150", "600", "450", "--dry-run", "--data", str(self.yaml_path)])
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        data = json.loads(res.stdout)
        self.assertTrue(data["dry_run"])
        self.assertIn("verification_id", data)
        self.assertFalse(self.label_path.exists())

    def test_cli_bbox_rejects_unknown_class(self):
        res = self.run_cli(["bbox", str(self.img_path), "airplane", "200", "150", "600", "450", "--data", str(self.yaml_path)])
        self.assertNotEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error"]["code"], "UNKNOWN_CLASS")

    def test_cli_bbox_requires_verification_id(self):
        res = self.run_cli(["bbox", str(self.img_path), "car", "200", "150", "600", "450", "--data", str(self.yaml_path)])
        self.assertNotEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"]["code"], "VERIFICATION_REQUIRED")

    def test_cli_bbox_rejects_bad_verification_id(self):
        res = self.run_cli(["bbox", str(self.img_path), "car", "200", "150", "600", "450", "--verification-id", "bad", "--data", str(self.yaml_path)])
        self.assertNotEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertEqual(data["error"]["code"], "VERIFICATION_MISMATCH")
        self.assertIn("expected_verification_id", data["error"]["details"])
        self.assertFalse(self.label_path.exists())

    def test_cli_bbox_force_writes_with_warning(self):
        res = self.run_cli(["bbox", str(self.img_path), "car", "200", "150", "600", "450", "--force", "--data", str(self.yaml_path)])
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        data = json.loads(res.stdout)
        self.assertTrue(self.label_path.exists())
        self.assertEqual(data["warnings"][-1]["code"], "VERIFICATION_BYPASSED")

    def test_cli_verify_stable_for_unchanged_label_state(self):
        args = ["verify", str(self.img_path), "car", "200", "150", "600", "450", "--data", str(self.yaml_path)]
        first = self.run_cli(args)
        second = self.run_cli(args)
        self.assertEqual(first.returncode, 0, msg=first.stderr)
        self.assertEqual(second.returncode, 0, msg=second.stderr)
        self.assertEqual(json.loads(first.stdout)["verification_id"], json.loads(second.stdout)["verification_id"])

    def test_cli_verify_invalid_bbox(self):
        res = self.run_cli(["verify", str(self.img_path), "car", "200", "150", "200", "450", "--data", str(self.yaml_path)])
        self.assertNotEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertEqual(data["error"]["code"], "INVALID_BBOX")

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

    def test_cli_schema(self):
        res = self.run_cli(["schema"])
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        data = json.loads(res.stdout)
        self.assertEqual(data["status"], "success")
        self.assertIn("doctor", data["commands"])
        self.assertIn("verify", data["commands"])
        self.assertIn("verify", data["schema_files"])

    def test_a4od_version(self):
        res = self.run_a4od(["--version"])
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        data = json.loads(res.stdout)
        self.assertEqual(data["api_version"], "1")
        self.assertEqual(data["implementation"], "annotation.py")

    def test_a4od_capabilities(self):
        res = self.run_a4od(["capabilities"])
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        data = json.loads(res.stdout)
        self.assertEqual(data["api_version"], "1")
        self.assertEqual(data["preferred_cli"], "a4od")
        self.assertTrue(data["mutation_rules"]["bbox_add_requires_verification_id"])
        self.assertEqual(data["coordinate_contract"]["bbox_input_format"], "xyxy")

    def test_cli_doctor(self):
        res = self.run_cli(["doctor", "--data", str(self.yaml_path), "--run-smoke"])
        self.assertEqual(res.returncode, 0, msg=res.stdout + res.stderr)
        data = json.loads(res.stdout)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["class_count"], 2)

    def test_cli_inspect(self):
        res = self.run_cli(["inspect", str(self.img_path), "car", "100", "100", "300", "300", "--data", str(self.yaml_path)])
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        data = json.loads(res.stdout)
        self.assertEqual(data["status"], "success")
        self.assertTrue(Path(data["inspect_image_path"]).exists())


if __name__ == "__main__":
    unittest.main()
