# Implementation Plan - AI-Assisted YOLO Annotation Tool (`annotation.py`)

Kế hoạch xây dựng công cụ CLI và module Python hỗ trợ quy trình AI tự động phát hiện đối tượng (Object Detection) và gán nhãn định dạng YOLO dựa trên phương pháp **Visual Prompting / Grid Overlay** kết hợp **Vòng lặp tự kiểm chứng (Self-verification Loop)**.

---

## 1. Yêu cầu & Quyết định Thiết kế

- **Cấu hình dataset**: Sử dụng chuẩn `dataset/data.yaml` (hỗ trợ `names: [...]` hoặc `names: {0: 'class0', 1: 'class1'}`).
- **Hệ tọa độ đầu vào**: AI truyền trực tiếp tọa độ **Pixel** `(xmin, ymin, xmax, ymax)` thu được từ ảnh lưới. Tool tự động chuyển đổi sang chuẩn YOLO normalized `(x_center, y_center, w, h)` khi lưu file nhãn `.txt`.
- **Hỗ trợ nhãn có sẵn**: Tự động nhận diện file `.txt` có sẵn (ở cùng thư mục hoặc trong `dataset/labels/`) và vẽ nhãn cũ (màu xanh lá) trên ảnh grid/visual để AI không đánh nhãn trùng lặp.
- **Thư viện sử dụng**: Pillow (`PIL`) + `PyYAML` (thuần Python, nhẹ, nhanh, không phụ thuộc C/OpenCV).

---

## 2. Cấu trúc Source Code

```
A4OD/
├── annotation.py              # CLI Entrypoint (subcommands: grid, visual, bbox)
├── src/
│   ├── __init__.py
│   ├── config.py              # Xử lý dataset/data.yaml và mapping đường dẫn ảnh <-> nhãn
│   ├── coords.py              # Chuyển đổi tọa độ (pixel xyxy <-> yolo xywh normalized)
│   ├── yolo_io.py             # Đọc/ghi/quản lý file label .txt chuẩn YOLO
│   ├── grid_renderer.py       # Render lưới pixel + thước đo tọa độ + nhãn cũ (như dataset/images/1.png)
│   └── visualizer.py          # Render bbox ứng viên (màu đỏ) + nhãn cũ (màu xanh) để AI review
├── tmp/                       # Thư mục chứa ảnh tạm thời (_grid.png, _visual.png)
├── tests/
│   ├── test_coords.py         # Unit tests chuyển đổi tọa độ
│   ├── test_yolo_io.py        # Unit tests đọc ghi YOLO format
│   └── test_cli.py            # Integration tests cho CLI subcommands
├── docs/
│   └── plan_annotation_tool.md
└── dataset/
    ├── data.yaml              # File cấu hình class mẫu
    ├── images/                # Ảnh nguồn
    └── labels/                # Nhãn YOLO
```

---

## 3. Chi tiết các Subcommand của `annotation.py`

### 1. `annotation grid`
```bash
python annotation.py grid <image_path> [--cell-size 200] [--data dataset/data.yaml]
```
- **Chức năng**:
  - Vẽ thước đo tọa độ pixel trên và trái (origin `(0, 0)` màu đỏ).
  - Vẽ các đường lưới pixel cách nhau `cell_size` px kèm tọa độ `(x, y)` tại góc mỗi ô.
  - Vẽ các bounding box **đã có sẵn** trong file `.txt` (nét xanh lá + tên class).
  - Lưu ảnh kết quả vào `tmp/<image_name>_grid.png`.
  - In ra metadata JSON: `width`, `height`, `cell_size`, `existing_boxes`.

### 2. `annotation visual`
```bash
python annotation.py visual <image_path> <class> <xmin> <ymin> <xmax> <ymax> [--data dataset/data.yaml]
```
- **Chức năng**:
  - Vẽ bbox ứng viên do AI đề xuất (viền đỏ đậm + nhãn) và các bbox đã có (viền xanh lá).
  - Lưu ảnh review vào `tmp/<image_name>_visual.png`.

### 3. `annotation bbox`
```bash
python annotation.py bbox <image_path> <class> <xmin> <ymin> <xmax> <ymax> [--action add|delete|list] [--data dataset/data.yaml]
```
- **Chức năng**:
  - `add`: Chuyển tọa độ pixel thành YOLO normalized và ghi vào file `.txt`.
  - `delete`: Xóa bbox theo index hoặc tọa độ.
  - `list`: Liệt kê các bbox hiện có của ảnh.

---

## 4. Kế hoạch Kiểm thử & Xác minh (Verification Plan)

### Automated Tests
```bash
python3 -m unittest discover tests
```
- Test chuyển đổi tọa độ `pixel_xyxy <-> yolo_xywh_norm` chính xác tới từng pixel và float precision.
- Test parse file `dataset/data.yaml` định dạng list và dict.
- Test đọc/ghi file `.txt` YOLO chuẩn.
- Test render ảnh `grid` và `visual`.

### Manual End-to-End Simulation
1. Tạo file `dataset/data.yaml` mẫu (chứa class `car`, `motorcycle`, `person`, `tree`...).
2. Chạy `python annotation.py grid dataset/images/1.png --cell-size 200 --data dataset/data.yaml` $\to$ kiểm tra `tmp/1_grid.png`.
3. Chạy `python annotation.py visual dataset/images/1.png car 890 630 1020 760 --data dataset/data.yaml` $\to$ kiểm tra `tmp/1_visual.png`.
4. Chạy `python annotation.py bbox dataset/images/1.png car 890 630 1020 760 --action add --data dataset/data.yaml` $\to$ kiểm tra `dataset/labels/1.txt`.
5. Chạy lại `grid` hoặc `visual` $\to$ xác nhận bbox vừa tạo xuất hiện ở trạng thái đã có (màu xanh).
