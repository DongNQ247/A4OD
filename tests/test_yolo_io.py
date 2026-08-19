import tempfile
import unittest
from pathlib import Path
from src.yolo_io import add_yolo_label, delete_yolo_label, read_yolo_labels


class TestYoloIO(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.label_file = Path(self.temp_dir.name) / "sample.txt"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_add_and_read_labels(self):
        class_map = {0: "person", 1: "car"}
        # Add box 1: class 0 (person), xc=0.5, yc=0.5, w=0.2, h=0.4
        add_yolo_label(self.label_file, 0, 0.5, 0.5, 0.2, 0.4)
        # Add box 2: class 1 (car), xc=0.8, yc=0.8, w=0.1, h=0.1
        add_yolo_label(self.label_file, 1, 0.8, 0.8, 0.1, 0.1)

        boxes = read_yolo_labels(self.label_file, img_w=1000, img_h=1000, class_map=class_map)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0]["class_name"], "person")
        self.assertEqual(boxes[0]["pixel_bbox"], [400, 300, 600, 700])
        self.assertEqual(boxes[1]["class_name"], "car")
        self.assertEqual(boxes[1]["pixel_bbox"], [750, 750, 850, 850])

    def test_delete_label(self):
        add_yolo_label(self.label_file, 0, 0.5, 0.5, 0.2, 0.4)
        add_yolo_label(self.label_file, 1, 0.8, 0.8, 0.1, 0.1)

        # Delete first
        success = delete_yolo_label(self.label_file, 0)
        self.assertTrue(success)

        boxes = read_yolo_labels(self.label_file, img_w=1000, img_h=1000)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["class_id"], 1)


if __name__ == "__main__":
    unittest.main()
