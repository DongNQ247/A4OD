# Walkthrough - AI-Assisted YOLO Annotation Tool (`annotation.py`)

Công cụ CLI và thư viện Python hỗ trợ AI tự động hóa quy trình Object Detection và gán nhãn dataset chuẩn YOLO sử dụng kỹ thuật **Coarse-to-Fine (Lưới thưa $\to$ Zoom mịn)** và **Corner Inspection (Soi 4 góc vi sai)** giúp tối ưu hóa độ chính xác từng pixel và tiết kiệm từ 60% – 85% Vision Tokens.

---

## 1. Cấu trúc Source Code & Thư mục Dự án

```
A4OD/
├── annotation.py              # CLI Entrypoint chính (grid, zoom, corners, visual, bbox)
├── dataset/
│   ├── data.yaml              # Cấu hình dataset mẫu (names: {0: traffic_sign...})
│   ├── images/                # Ảnh nguồn
│   └── labels/                # Nhãn YOLO tương ứng
├── requirements.txt           # Dependencies (Pillow, PyYAML)
├── src/
│   ├── __init__.py
│   ├── config.py              # Xử lý dataset/data.yaml và đường dẫn nhãn YOLO
│   ├── coords.py              # Chuyển đổi tọa độ (pixel xyxy <-> yolo xywh normalized)
│   ├── yolo_io.py             # Đọc, ghi và xóa nhãn .txt chuẩn YOLO
│   ├── grid_renderer.py       # Render lưới pixel, thước đo kèm vạch chia phụ 50px
│   ├── zoom_renderer.py       # [MỚI] Render ROI zoom với lưới mịn (50px/25px) & tọa độ Global
│   ├── corner_inspector.py    # [MỚI] Cắt ghép 4 góc BBox (TL, TR, BL, BR) siêu nhẹ token
│   └── visualizer.py          # Render preview bbox ứng viên toàn cảnh
├── tests/
│   ├── __init__.py
│   ├── test_coords.py         # Unit test chuyển đổi tọa độ
│   ├── test_config.py         # Unit test đọc dataset/data.yaml & resolve đường dẫn
│   ├── test_yolo_io.py        # Unit test đọc/ghi/xóa nhãn .txt
│   ├── test_zoom.py           # [MỚI] Unit test zoom_renderer
│   ├── test_corners.py        # [MỚI] Unit test corner_inspector
│   └── test_cli.py            # Integration test toàn bộ các lệnh CLI
├── prompt/
│   └── ai_annotation_prompt.md # [MỚI] System Prompt & Task Prompt tối ưu Coarse-to-Fine
└── docs/
    ├── plan_annotation_tool.md
    ├── plan_precision_and_token_optimization.md
    └── walkthrough.md
```

---

## 2. Quy trình & Lệnh CLI Coarse-to-Fine

### Bước 1: Khảo sát Toàn cảnh (`grid`)
```bash
.venv/bin/python annotation.py grid dataset/images/59.png --cell-size 200 --data dataset/data.yaml
```
- Hiển thị ảnh tổng quan với thước đo pixel và vạch chia phụ 50px.
- Tự động hiển thị các nhãn cũ (màu xanh lá) để không gán trùng lặp.

### Bước 2: Phóng to Vùng Chi tiết Giữ Tọa độ Global (`zoom`)
```bash
.venv/bin/python annotation.py zoom dataset/images/59.png 780 320 920 460 --cell-size 25 --data dataset/data.yaml
```
- Phóng to riêng vùng ROI, tự động căn chỉnh tỷ lệ (auto-scale) chống đè chữ số.
- Thước đo vẫn giữ nguyên tọa độ ảnh gốc (Global Coordinates), AI không cần tính toán cộng trừ offset.

### Bước 3: Soi 4 Góc Mép Viền Siêu Nhẹ Token (`corners`)
```bash
.venv/bin/python annotation.py corners dataset/images/59.png 830 380 895 430 --patch-size 60
```
- Ghép 4 góc `[TL]`, `[TR]`, `[BL]`, `[BR]` thành 1 ảnh composite siêu nhỏ (~200x200px, tiêu tốn chỉ ~60 tokens).
- Giúp AI kiểm tra mép biên chính xác tuyệt đối mà không cần tải lại toàn bộ ảnh lớn.

### Bước 4: Ghi Nhãn Chính thức vào Dataset (`bbox add`)
```bash
.venv/bin/python annotation.py bbox dataset/images/59.png traffic_sign 830 380 895 430 --data dataset/data.yaml
```
- Chuyển đổi pixel sang YOLO normalized format và ghi vào file `.txt`.

---

## 3. Kết quả Kiểm thử

- Chạy toàn bộ 18 test cases:
  ```bash
  .venv/bin/python -m unittest discover tests
  ```
  ```
  Ran 18 tests in 1.396s
  OK
  ```
- Đã kiểm tra thực tế trên ảnh thực tế `dataset/images/59.png` với nhãn có sẵn và nhãn mới được thêm thành công.
