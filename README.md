# nlp_day_3
# Demo 
## Thành viên nhóm
1. Nguyễn Thanh Bình - MSHV: 2591302
2. Huỳnh Đình Hiệp - MSHV: 2591303
3. Trần Minh Sang - MSHV: 2591322
4. Lê Nguyễn Tuấn Kiệt - MSHV: 2591311

---

## Cách chạy

1. Cài thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
2. Tải dữ liệu
   ```bash
   python get_data_from_drive.py
3. Chạy API
   ```bash
   python api.py
4. Truy cập
   ```bash
   http://127.0.0.1:9999/
5. Nếu muốn train lại
   ```bash
   python train.py --model_name <tên model>