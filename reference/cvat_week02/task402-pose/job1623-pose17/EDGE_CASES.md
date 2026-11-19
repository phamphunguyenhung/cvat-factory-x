# Job 1623 — edge case từng frame (f90–f119, file G06_091…120)

Đọc từ ảnh thu nhỏ ngày 16/11/2026; xác nhận lại khi vẽ. Quy tắc: `../../../weeks/guidelines/w2-face50-pose17.md` mục 5A (mục G cho trái/phải khi thân xoay). Q = che, O = ngoài biên.
Hai kiểu ảnh: **IR chính diện** (camera trước mặt, chân/cổ chân thường ra ngoài đáy ảnh → O), **ảnh màu nhìn nghiêng** (chân/gối thường ra mép phải → O), **IR fisheye nằm ngang** (nguy cơ nhầm trái/phải).

| Frame | Ảnh | Lưu ý khi vẽ |
|---|---|---|
| f90 | IR chính diện, cúi lấy túi nilon bên trái ảnh | cổ chân O (ra đáy); túi che tay thì cổ tay vẫn đặt ở khớp |
| f91 | IR chính diện, cúi cầm túi nilon, đầu nghiêng | túi che cẳng/cổ tay: Q; cổ chân O |
| f92 | IR, cúi sát camera, đầu sát mép trên | mũi/mắt/tai có thể vượt mép: O; thân co lại, vai gần đầu |
| f93 | màu nghiêng, đọc sách che ngực/tay | sách che tay-ngực: Q; chân ra mép: O |
| f94 | IR, báo trải trên đùi che hông-gối | hông: Q nếu suy ra được; gối, cổ chân không đoán nổi: O (v0) |
| f95 | IR, nghiêng người, tay xuống bên trái ảnh, áo sáng | cổ chân O |
| f96 | màu nghiêng, cúi đầu, cầm đồ | đùi/gối ra mép phải: O; đầu sát mép trái |
| f97 | màu nghiêng, gập người về trước, đầu sát mép trái | mặt cúi: mũi/mắt xác định theo hướng đầu; điểm ra mép: O |
| f98 | màu nghiêng, tay nâng lên vô-lăng | tay giơ cao; chân ra mép: O |
| f99 | **IR rất tối**, tay chồng nhau | tăng sáng; điểm không thấy: ước lượng + Q |
| f100 | màu nghiêng, **nhòe chuyển động mạnh** | đặt ở tâm khớp ước lượng, Q nếu không phân biệt khớp |
| f101 | màu nghiêng, cầm sách, laptop trên đùi | sách/laptop che đùi-gối: Q; chân ra mép: O |
| f102 | IR chính diện, viết giấy, cúi đầu, kính | cổ chân O |
| f103 | IR **nằm ngang**, gọi điện, đầu bên trái ảnh | **R/L theo ảnh** (mục G, UNCERTAIN_CLASS nếu hai bên cùng x); điện thoại che tai |
| f104 | IR nằm ngả, hai tay giơ lên vô-lăng | nguy cơ nhầm trái/phải; tay cầm vô-lăng: cổ tay ở khớp |
| f105 | màu nghiêng, đọc sách, kính | sách che tay: Q |
| f106 | IR fisheye nằm ngang, đầu bên phải, cầm túi | trái/phải khó; gối/cổ chân thường trên vô-lăng/ghế: Q hoặc O nếu ra ảnh. Nháp YOLO đã sai ở frame này |
| f107 | IR chính diện, cúi, cầm vật nhỏ | cổ chân O |
| f108 | màu nghiêng, đầu nghiêng, tay vươn trước cầm bút, laptop | chân ra mép: O |
| f109 | IR chính diện, bao/túi trước bụng che hông | hông: Q; cổ chân O |
| f110 | IR fisheye nằm ngang, mặt bị tay/tay khoanh che | mặt che: mũi/mắt/tai Q ở vị trí đầu ước lượng; trái/phải khó |
| f111 | IR chính diện, giấy/sách trên đùi, kính | hông-gối: Q; cổ chân O |
| f112 | IR nghiêng, tay giơ cao nắm vô-lăng | nguy cơ nhầm trái/phải |
| f113 | màu nghiêng, tay giơ lên bên trái ảnh | gập người; chân ra mép: O |
| f114 | IR chính diện, hai tay khép trước ngực cầm vật nhỏ | cổ tay đặt ở khớp; cổ chân O |
| f115 | màu nghiêng, cầm bút/giấy trước ngực | chân dài ra mép: O |
| f116 | IR chính diện, nghiêng sang phải ảnh, đầu sát mép phải | điểm ra mép: O; kính; có cốc trên bệ |
| f117 | màu nghiêng, viết giấy, kính | chân ra mép: O; túi gần chân |
| f118 | màu nghiêng, **gục mặt vào tay/vô-lăng**, đầu sát mép trên-trái | mặt khuất hẳn: mũi/mắt/tai đặt ở vị trí đầu ước lượng + Q; điểm ra mép: O |
| f119 | IR chính diện, ngồi thẳng, cười, tay đan trước bụng, kính | rõ nhất; cổ chân O |

Frame có nháp thô sẵn (chờ BTC xác nhận cách xử lý): f91, f92, f97, f98, f103, f104, f109, f110, f115, f116.
