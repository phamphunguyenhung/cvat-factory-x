# Edge case của task 473 (150 ảnh, đọc bằng mắt từ contact sheet 16/11/2026)

`fNN` = số frame CVAT (file `G06_<NN+1>.jpg`). Đọc từ ảnh thu nhỏ, **cần xác nhận lại khi vẽ từng ảnh**. Cách xử lý từng nhóm: `../../weeks/guidelines/w2-face50-pose17.md` mục 5A. Phân công job: f0–29 → 1972, f30–59 → 1973, f60–89 → 1974, f90–119 → 1975 (mình), f120–149 → 1976.

| Nhóm | Frame |
|---|---|
| Ảnh IR xám ban đêm (tối) | f5, f8, f9, f10, f13–f16, f18–f20, f22, f24, f29, f32, f34, f35, f37–f40, f43–f46, f48, f56, f58–f59, f61, f62, f64–f65, f67–f70, f75, f78, f80, f82, f84–f86, f88–f89, f92, f94–f95, f102, f104, f106, f110–f112, f116, f124, f128, f130, f142 |
| Fisheye, người nằm ngang/xoay ~90° (nguy cơ nhầm trái/phải) | f10, f20, f34, f44, f58, f67, f68, f92, f106, f116, f124, f142 |
| Kính râm hoặc tròng tối | f6, f45, f106, f116, f144, f147 |
| Kính trong (có phản sáng) | f1, f3, f7?, f11, f14, f15, f18–f21, f23, f24, f27, f29, f36–f38, f49, f51, f55, f60, f70, f75, f79, f86–f89, f93–f98, f100–f105, f108–f111, f113, f117–f118, f120, f122, f125, f132, f135, f137–f140, f146, f148 |
| Mắt nhắm | f17, f39, f40, f99, f112? |
| Miệng mở (nói, ngáp, cười) | f0, f99, f119, f120, f133, f134?, f147 |
| Mặt nghiêng/profile mạnh | f4, f7, f27, f31, f47, f50, f52, f57, f66, f74, f76, f79, f98, f100, f105, f118, f122, f136, f146, f148 |
| Ngửa mặt lên | f2, f28, f57, f66, f96, f97, f100, f105, f112, f118 |
| Cúi mặt nhìn xuống (điện thoại, giấy, đồ vật) | f30, f35, f37, f38, f46, f51, f54, f61, f62, f84, f85, f103, f111, f114, f115, f132, f138 |
| Tay che mặt / sát mặt | f2, f75, f102, f107, f113, f126, f129, f135, f143, f149 |
| Điện thoại áp tai/gần mặt | f1, f8, f9, f22, f73, f103, f117, f131, f138 |
| Chai, đồ ăn, đồ uống sát miệng | f19, f55, f59, f96, f54 |
| Sách, báo, laptop che mặt | f11, f34, f37, f44, f58, f67, f77, f92, f148 |
| Ba lô che đầu | f64 |
| Tóc dài che mày/mắt/má | f9, f46, f115, f125? |
| Mũ lưỡi trai, mũ len | f25, f81, f123, f127 |
| Khăn trùm đầu | f139 |
| Râu, ria che viền môi | f0, f27, f31, f61, f84, f85, f103, f109, f121, f141, f144, f145 |
| Đầu sát/vượt mép ảnh (khả năng điểm O) | f2, f20, f35, f46, f68, f92, f110, f142 |
| Hành khách nhìn thấy (không gán) | f0, f21, f25, f30, f60, f73, f81, f99, f103, f107, f113, f117, f123, f127, f133, f139, f141, f143, f145 |
| Mặt rất nhỏ trong ảnh | f5, f10, f13, f14, f24, f37, f38, f61, f84–f86, f94, f95, f110, f111 |

Dấu `?` = chưa chắc khi xem ảnh nhỏ.
