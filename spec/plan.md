Tool để Object Detection sử dụng codex để tự động xác định vị trí bbox


Ý tưởng sơ khai
- Sử dụng Codex với prompt "xác định đối tượng A và đánh nhãn trực quan hóa kết quả trên ảnh" 
- Đánh giá kết quả tương đối tốt

Ý tưởng tự động hóa
- Input: Tập dataset định dạng YOLO
- Ouput: Tập dataset được cập nhật thêm các object

Chuẩn bị tool support AI tạm gọi là annotation.py

Giúp AI hỗ trợ tạo bbox
```c
# Tạo bbox
annotation bbox <image_path> <class> <x_center> <y_center> <width> <height> 
```
Thực hiện:
1. 
2.
3. 

- giúp AI trực quan hóa kết quả để đánh giá lại

```c
# Trực quan hóa
annotation visual <image_path> <class> <x_center> <y_center> <width> <height>
```
Kết quả sẽ là bbox vẽ lên ảnh tạo image_path lưu trong tmp, sau đó AI sẽ đọc lại ảnh và đánh giá.


```c
# Tạo lưới ảnh
annotation gird <image_path> <size_cell> 
```
Kết quả sẽ là một ảnh được kẻ các ô lưới có kích trước size_cell (tính bằng pixel) lưu trong tmp và meta data của ảnh:
- width: ... & height: ....
- size_cell: ....

Ảnh đầu ra cần phải chứa thông tin để AI dễ dàng suy ra được tọa độ của vật, tham khảo ảnh mẫu trong dataset/images/1.png

Về quy trình đánh nhãn của AI
1. Sử dụng tool gird để tạo lưới -> giúp AI phán đoán vị trí tốt hơn
2. Sử dụng tool trực quan để AI tự kiểm chứng
3. Nếu AI thấy phán đoán của mình là chính xác, sẽ sử dụng tool bbox

Quan trọng, AI không trực tiếp được sửa txt mà chỉ được gọi thông qua tool

Chúng ta sẽ thảo luận thêm để làm rõ một vài điểm về mặt kỹ thuật và nghiệp vụ nếu có

Quan trọng là càng đơn giản càng tốt, tool được dev sử dụng, không phải người dùng, dev phải chấp nhận tuân thủ theo một quy trình nào đó
