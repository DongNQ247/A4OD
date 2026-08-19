# Kế hoạch Triển khai: Tối ưu Độ Chính Xác và Tiết Kiệm Token cho AI Object Detection

Tài liệu thiết kế và kế hoạch chi tiết bổ sung các công cụ định vị Coarse-to-Fine và kiểm tra vi sai mép biên siêu tiết kiệm token cho AI gán nhãn YOLO.

---

## 1. Mục tiêu Thiết kế

1. **Giải quyết triệt để vấn đề định vị sai / ảnh bị rối**:
   - Sử dụng mô hình **Coarse-to-Fine**: Ảnh toàn cảnh chỉ kẻ lưới thưa ($200\text{px}$), khi cần định vị chi tiết đối tượng cụ thể thì dùng lệnh `zoom` với lưới mịn ($50\text{px}$ hoặc $20\text{px}$).
   - Giữ nguyên hệ tọa độ toàn cục (**Global Coordinates**) trên toàn bộ thước đo và nhãn góc ô của ảnh zoom.
2. **Tiết kiệm tối đa Vision Tokens khi kiểm chứng (Verification)**:
   - Tool `corners` cắt ghép 4 góc viền của BBox (Top-Left, Top-Right, Bottom-Left, Bottom-Right) thành một ảnh nhỏ $120 \times 120\text{px}$ (~60-80 token) để AI kiểm tra độ khít của mép biên mà không cần tải lại toàn bộ ảnh lớn hàng ngàn pixel.
3. **Cải tiến trực quan trên Thước đo Grid**:
   - Thêm các vạch chia phụ ($50\text{px}$ sub-ticks) trên thước đo trục X & Y của ảnh `grid` tổng thể giúp dễ quan sát các mốc lẻ mà không làm rối ảnh bằng chữ số dày đặc.

---

## 2. Thiết kế Kỹ thuật Chi tiết

### A. Module `src/zoom_renderer.py` (Lệnh `zoom`)

```bash
python annotation.py zoom <image_path> <xmin> <ymin> <xmax> <ymax> [--cell-size 50] [--data dataset/data.yaml]
```

- **Đầu vào**:
  - `xmin, ymin, xmax, ymax`: Tọa độ pixel của vùng cần phóng to (ROI).
  - `cell_size`: Bước nhảy lưới mịn (mặc định $50\text{px}$, hoặc $20\text{px}$).
- **Xử lý**:
  - Cắt vùng ảnh gốc theo `[xmin, ymin, xmax, ymax]`.
  - Tạo canvas có thước đo trên (từ `xmin` đến `xmax`) và thước đo trái (từ `ymin` đến `ymax`).
  - Kẻ lưới với bước nhảy `cell_size` và ghi nhãn tọa độ thực `(x_global, y_global)` tại mỗi đỉnh ô.
  - Vẽ lại các bounding box đã có (màu xanh lá) nếu nằm trong vùng crop.
- **Đầu ra**:
  - Lưu ảnh vào `tmp/<image_stem>_zoom_<xmin>_<ymin>_<xmax>_<ymax>.png`.
  - In ra metadata JSON: `crop_box`, `width`, `height`, `cell_size`, `existing_boxes`.

---

### B. Module `src/corner_inspector.py` (Lệnh `corners`)

```bash
python annotation.py corners <image_path> <xmin> <ymin> <xmax> <ymax> [--patch-size 60] [--data dataset/data.yaml]
```

- **Đầu vào**:
  - `xmin, ymin, xmax, ymax`: Tọa độ BBox mà AI đang muốn kiểm tra.
  - `patch_size`: Kích thước mỗi miếng vuông góc (mặc định $60\text{px}$).
- **Xử lý**:
  - Cắt 4 miếng vuông $60 \times 60\text{px}$ quanh:
    1. **Top-Left (TL)**: tâm tại `(xmin, ymin)`
    2. **Top-Right (TR)**: tâm tại `(xmax, ymin)`
    3. **Bottom-Left (BL)**: tâm tại `(xmin, ymax)`
    4. **Bottom-Right (BR)**: tâm tại `(xmax, ymax)`
  - Vẽ đường biên BBox màu đỏ tương ứng ở mỗi góc.
  - Ghép thành ảnh composite $2 \times 2$ có nhãn phân biệt: `[TL]`, `[TR]`, `[BL]`, `[BR]`.
- **Đầu ra**:
  - Lưu ảnh vào `tmp/<image_stem>_corners_<xmin>_<ymin>_<xmax>_<ymax>.png`.
  - In ra metadata JSON.

---

### C. Cải tiến `src/grid_renderer.py`

- Bổ sung các vạch chia nhỏ (sub-ticks, cao 3px) cách nhau $50\text{px}$ trên trục X & Y của thước đo ảnh toàn cảnh.

---

## 3. Cập nhật Quy trình AI & Prompt (`docs/ai_annotation_prompt.md`)

Quy trình mới tối ưu độ chính xác và token:
1. **Bước 1 (Survey)**: Gọi `grid` $\to$ nhận diện vùng sơ bộ của vật thể unannotated.
2. **Bước 2 (Optional - High-Precision Zoom)**: Với vật thể nhỏ/xa $\to$ gọi `zoom` $\to$ đọc tọa độ chính xác từng pixel với lưới $50\text{px}$.
3. **Bước 3 (Ultra-Light Verification)**: Gọi `corners` để kiểm tra độ khít 4 mép biên (tiết kiệm 85% token so với `visual`). Hoặc gọi `visual` nếu muốn nhìn tổng thể.
4. **Bước 4 (Commit)**: Gọi `bbox --action add` để ghi vào dataset.

---

## 4. Kế hoạch Kiểm thử (Verification Plan)

1. **Automated Unit Tests**:
   - `tests/test_zoom.py`: Test crop, tính toán thước đo global coordinates, xử lý biên.
   - `tests/test_corners.py`: Test cắt ghép 4 góc, xử lý bounding box sát mép ảnh.
   - `tests/test_cli.py`: Test các lệnh CLI `zoom` và `corners`.
   - Chạy: `.venv/bin/python -m unittest discover tests` (đảm bảo 100% pass).
2. **Manual & Visual Tests**:
   - Chạy thử `zoom` và `corners` trên ảnh thực tế `dataset/images/street_raw.png`.
   - Kiểm tra ảnh `tmp/*_zoom.png` và `tmp/*_corners.png`.
