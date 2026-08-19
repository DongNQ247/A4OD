import tempfile
import unittest
from pathlib import Path
import os
import yaml
from src.config import find_label_path, load_dataset_config, resolve_class


class TestConfig(unittest.TestCase):
    def test_load_dataset_config_list(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump({"names": ["person", "car", "dog"]}, f)
            f_path = f.name

        try:
            cfg = load_dataset_config(f_path)
            self.assertEqual(cfg[0], "person")
            self.assertEqual(cfg[1], "car")
            self.assertEqual(cfg[2], "dog")
        finally:
            Path(f_path).unlink()

    def test_load_dataset_config_dict(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump({"names": {0: "person", 5: "truck"}}, f)
            f_path = f.name

        try:
            cfg = load_dataset_config(f_path)
            self.assertEqual(cfg[0], "person")
            self.assertEqual(cfg[5], "truck")
        finally:
            Path(f_path).unlink()

    def test_load_dataset_config_defaults_to_dataset_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_dir = tmp_path / "dataset"
            dataset_dir.mkdir()
            with open(dataset_dir / "data.yaml", "w", encoding="utf-8") as f:
                yaml.dump({"names": {0: "traffic_sign"}}, f)

            old_cwd = Path.cwd()
            try:
                os.chdir(tmp_path)
                cfg = load_dataset_config()
            finally:
                os.chdir(old_cwd)

            self.assertEqual(cfg[0], "traffic_sign")

    def test_resolve_class(self):
        class_map = {0: "person", 1: "car", 2: "bicycle"}

        # By name
        cid, cname = resolve_class("car", class_map)
        self.assertEqual(cid, 1)
        self.assertEqual(cname, "car")

        # Case-insensitive
        cid, cname = resolve_class("CAR", class_map)
        self.assertEqual(cid, 1)
        self.assertEqual(cname, "car")

        # By ID string
        cid, cname = resolve_class("2", class_map)
        self.assertEqual(cid, 2)
        self.assertEqual(cname, "bicycle")

        # Unknown name
        with self.assertRaises(ValueError):
            resolve_class("airplane", class_map)

        # Unknown id
        with self.assertRaises(ValueError):
            resolve_class("9", class_map)

    def test_find_label_path(self):
        # Same dir
        img_path = Path("/tmp/dataset/image1.jpg")
        self.assertEqual(find_label_path(img_path), Path("/tmp/dataset/image1.txt"))

        # YOLO images/labels structure
        img_yolo = Path("/tmp/dataset/images/train/sample.png")
        self.assertEqual(find_label_path(img_yolo), Path("/tmp/dataset/labels/train/sample.txt"))

        # A4OD data/dataset sibling structure
        img_data = Path("/tmp/a4od/data/12447.png")
        self.assertEqual(find_label_path(img_data), Path("/tmp/a4od/dataset/labels/12447.txt"))


if __name__ == "__main__":
    unittest.main()
