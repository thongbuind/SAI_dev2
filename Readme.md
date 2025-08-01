# Build a Pre-trained Language Model (LM) from scratch with Transformer architecture

## Thang điểm đánh giá bài kiểm tra tạo câu:
- **10**: Câu hay tuyệt đối.
- **9**: Câu đúng cấu trúc, đúng nội dung, khá ổn.
- **6-8**: Câu đúng cấu trúc, nội dung tạm đúng nhưng chung chung quá, không cụ thể hoặc nội dung chỉ gần đúng.
- **5**: Câu đúng cấu trúc, nhưng sai ngữ nghĩa, nội dung.
- **2-4**: Câu sai cấu trúc, nhưng nội dung vẫn có gì đó liên quan nhẹ
- **1**: Câu không liên quan, sai cấu trúc, các từ không có sự liên quan.

---

## Test pre-train

*Mô hình mới pre-train, chưa fine-tune nên chỉ có thể sinh văn bản dựa trên đầu vào, không yêu cầu tính chính xác của thông tin, chỉ xét cấu trúc và ngữ nghĩa.*

### Test 1
**Req:** bánh mì 

**Res:** bánh_mì không_chỉ là món ăn mà_còn là một phần quan_trọng trong văn_hoá ẩm_thực của nhiều quốc_gia.

---

### Test 2
**Req:** việt nam 

**Res:** việt_nam đã trải qua nhiều biến_động chính_trị và xã_hội, từ đó tạo ra những thay_đổi lớn trong cấu_trúc xã_hội.

---

### Test 3
**Req:** việt nam sở hữu 

**Res:** việt_nam sở_hữu nhiều sản_phẩm thủ_công mỹ_nghệ độc_đáo, từ dệt vải đến các sản_phẩm thủ_công mỹ_nghệ.

---

### Test 4
**Req:** phở 

**Res:** phở thường được ăn kèm với rau_sống và các món ăn khác, tạo nên sự hấp_dẫn cho bữa ăn.

---

### Test 5
**Req:** buổi sáng người việt nam thường

**Res:** buổi sáng người việt_nam thường mang lại những trải_nghiệm thú_vị cho người đọc.

---

### Test 6
**Req:** đám mây 

**Res:** đám mây là một phần quan_trọng trong chu_trình nước của trái_đất, ảnh_hưởng đến sự phát_triển của các loài sinh_vật.

---

### Test 7
**Req:** Đinh Tiên Hoàng 

**Res:** đinh tiên hoàng là một nhân_vật lịch_sử quan_trọng trong lịch_sử việt_nam, nổi_bật với những chiến_công lẫy_lừng.

---

### Test 8
**Req:** Lê Lợi đã 

**Res:** lê_lợi đã có nhiều đóng_góp cho sự_nghiệp của ông, góp_phần vào việc xây_dựng lòng tin và lòng yêu nước.

---

### Test 9
**Req:** sau khi lên ngôi

**Res:** sau khi lên_ngôi, nhiều người đã phải chịu_đựng những khó_khăn trong việc vượt qua khó_khăn.

---

### Test 10
**Req:** công thức 1 

**Res:** công_thức 1 không_chỉ là một môn thể_thao mà_còn là một phần quan_trọng trong văn_hoá và xã_hội của nhiều quốc_gia.

---

### Test 11
**Req:** sáng hôm ấy 

**Res:** sáng hôm ấy là một ngày dài mà tôi đã thích khám_phá.

---

### Test 12
**Req:** sau khi ăn xong, chúng tôi 

**Res:** sau khi ăn xong, chúng_tôi đã chuẩn_bị cho một bữa tiệc sinh_nhật, mang lại sự gắn_kết và gắn_kết.

---

### Test 13
**Req:** mặc dù

**Res:** mặc_dù có nhiều thách_thức, nhưng việc duy_trì sự ổn_định trong quản_lý tài_chính là rất quan_trọng.

---

### Test 14
**Req:** bởi vì trời mưa,

**Res:** bởi_vì trời mưa, không_khí trở_nên dễ_chịu hơn khi mọi người cảm_thấy thoải_mái hơn.
