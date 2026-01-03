# Template Báo Cáo Khóa Luận Tốt Nghiệp

Template LaTeX cho báo cáo khóa luận tốt nghiệp của Trường Đại học Khoa học Tự nhiên - ĐHQG TP.HCM, Khoa Công nghệ Thông tin.

## Mục đích

Template này cung cấp:
- Cấu trúc chuẩn cho báo cáo khóa luận
- Các ví dụ về cách sử dụng LaTeX trong báo cáo
- Định dạng và style đã được cấu hình sẵn
- Hướng dẫn sử dụng các tính năng phổ biến

## Cấu trúc thư mục

```
Thesis_Report_2025/
├── main.tex                 # File chính, chứa preamble và cấu trúc tài liệu
├── Title/
│   └── title.tex           # Trang bìa
├── Chapter1/               # Chương 1: Giới thiệu
│   └── chapter1.tex
├── Chapter2/               # Chương 2: Tổng quan tài liệu
│   └── chapter2.tex
├── Chapter3/               # Chương 3: Phương pháp đề xuất
│   └── chapter3.tex
├── Chapter4/               # Chương 4: Thực nghiệm
│   └── chapter4.tex
├── Chapter5/               # Chương 5: Kết luận
│   └── chapter5.tex
├── Appendix/               # Các phụ lục
│   ├── commitment.tex      # Lời cam đoan
│   ├── thanks.tex          # Lời cảm ơn
│   ├── summary.tex        # Tóm tắt
│   ├── proposal.tex       # Đề cương
│   └── appendix1.tex       # Phụ lục 1
├── References/
│   └── references.bib     # Tài liệu tham khảo (BibTeX)
├── figures/                # Thư mục chứa hình ảnh
└── out/                    # Thư mục output (PDF, aux files, etc.)
```

## Hướng dẫn sử dụng

### Bước 1: Cài đặt

1. Cài đặt LaTeX distribution (TeX Live, MiKTeX, hoặc MacTeX)
2. Cài đặt editor hỗ trợ LaTeX (TeXstudio, Overleaf, VS Code với LaTeX extension)
3. Đảm bảo có các gói cần thiết (thường được cài tự động)

### Bước 2: Điền thông tin cá nhân

Mở file `main.tex` và điền thông tin vào các biến ở dòng 139-143:

```latex
\newcommand{\tenSV}{Họ~và~Tên~Sinh~Viên}
\newcommand{\mssv}{MSSV1~MSSV2}
\newcommand{\tenKL}{Tên~Đề~Tài~Khóa~Luận}
\newcommand{\tenGVHD}{Họ~Tên~Giảng~Viên~Hướng~Dẫn}
\newcommand{\tenBM}{Tên~Bộ~Môn}
```

**Lưu ý:** Dấu `~` tạo khoảng trắng không bị ngắt dòng (các từ nối bằng `~` sẽ luôn cùng một dòng).

### Bước 3: Thay thế nội dung mẫu

1. Mở từng file chapter (Chapter1/chapter1.tex, ...)
2. Thay thế nội dung mẫu bằng nội dung thực tế của bạn
3. Giữ lại các ví dụ nếu cần tham khảo

### Bước 4: Thêm tài liệu tham khảo

1. Mở file `References/references.bib`
2. Thêm các entry mới theo format BibTeX
3. Sử dụng `\cite{key}` trong nội dung để trích dẫn

### Bước 5: Build tài liệu

**Quan trọng:** Build theo thứ tự sau:

1. **BIB** - Xử lý tài liệu tham khảo (BibTeX/Biber)
2. **PDF** - Compile LaTeX lần 1
3. **PDF** - Compile LaTeX lần 2 (để cập nhật references)

Hoặc sử dụng build chain tự động trong editor của bạn.

## Các tính năng chính

### 1. Trích dẫn tài liệu tham khảo

```latex
% Trích dẫn đơn
Nghiên cứu của Zhang \cite{Zhang_Isola_Efros_2016}...

% Trích dẫn nhiều tài liệu
Các mô hình \cite{Ho_Jain_Abbeel_2020, LDM_Rombach_Blattmann_Lorenz_Esser_Ommer_2022}...
```

### 2. Chèn hình ảnh

```latex
\begin{figure}[htp]
    \centering
    \includegraphics[width=0.8\linewidth]{figures/main_model.png}
    \caption{Mô tả hình ảnh}
    \label{fig:example}
\end{figure}
```

**Tùy chỉnh kích thước:**
- `width=0.5\linewidth` - 50% chiều rộng trang
- `height=5cm` - Chiều cao cố định
- `scale=0.8` - 80% kích thước gốc

### 3. Tạo bảng

```latex
\begin{table}[ht]
    \centering
    \begin{tabular}{lcc}
        \toprule
        Cột 1 & Cột 2 & Cột 3 \\
        \midrule
        Dòng 1 & 10 & 100 \\
        Dòng 2 & 20 & 200 \\
        \bottomrule
    \end{tabular}
    \caption{Mô tả bảng}
    \label{tab:example}
\end{table}
```

**Căn chỉnh cột:**
- `l` - Căn trái
- `c` - Căn giữa
- `r` - Căn phải
- `p{width}` - Cột với chiều rộng cố định

### 4. Công thức toán học

```latex
% Công thức inline
Phương trình $E = mc^2$ là...

% Công thức đánh số
\begin{equation} \label{eqn:example}
    f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
\end{equation}

% Nhiều công thức liên tiếp
\begin{align}
    x &= a + b \label{eqn:add} \\
    y &= c \cdot d \label{eqn:mult}
\end{align}
```

### 5. Điều chỉnh khoảng cách

```latex
% Khoảng cách dọc
\vspace{0.5cm}

% Khoảng cách ngang
\hspace{1cm}

% Khoảng cách không bị ngắt dòng
Từ~này~sẽ~luôn~cùng~một~dòng
```

### 6. Định dạng văn bản

```latex
\textbf{In đậm}
\textit{In nghiêng}
\underline{Gạch chân}
\texttt{Mã nguồn}
\emph{Nhấn mạnh}
```

## Các file quan trọng

### main.tex
- Chứa tất cả các package và cấu hình
- Định nghĩa các lệnh tùy chỉnh
- Cấu trúc tài liệu (thứ tự các chapter)

### Title/title.tex
- Trang bìa của báo cáo
- Sử dụng các biến đã định nghĩa trong main.tex

### References/references.bib
- Tất cả tài liệu tham khảo
- Format BibTeX chuẩn
- Key phải là duy nhất

## Lưu ý quan trọng

1. **Build order:** Luôn build theo thứ tự BIB > PDF > PDF
2. **Encoding:** File phải được lưu với encoding UTF-8
3. **Hình ảnh:** Đặt trong thư mục `figures/`, hỗ trợ PNG, JPG, PDF
4. **Tham chiếu:** Luôn sử dụng `\label` và `\ref` để tham chiếu
5. **Trích dẫn:** Key trong `\cite{key}` phải khớp với key trong `references.bib`

## Troubleshooting

### Lỗi "Undefined citation"
- Chạy lại BIB build
- Kiểm tra key trong `\cite{}` có khớp với `references.bib`

### Hình ảnh không hiển thị
- Kiểm tra đường dẫn file
- Đảm bảo file tồn tại trong thư mục `figures/`
- Kiểm tra extension file (phải viết đúng: .png, .jpg, .pdf)

### Bảng/hình bị tràn trang
- Sử dụng `\resizebox` cho bảng lớn
- Điều chỉnh kích thước hình ảnh
- Sử dụng `[H]` thay vì `[htp]` để buộc vị trí

### Công thức không hiển thị đúng
- Kiểm tra các package toán học đã được load
- Đảm bảo đang ở trong môi trường math (`$...$` hoặc `\[...\]`)

## Tài liệu tham khảo

- [LaTeX Wikibook](https://en.wikibooks.org/wiki/LaTeX)
- [Overleaf Documentation](https://www.overleaf.com/learn)
- [BibTeX/Biblatex Guide](https://www.overleaf.com/learn/latex/Bibliography_management_in_LaTeX)

## Liên hệ

Nếu có thắc mắc về template, vui lòng liên hệ:
- Email: [email của khoa/bộ môn]
- Hoặc tham khảo tài liệu hướng dẫn của khoa

## License

Template này được cung cấp miễn phí cho sinh viên sử dụng trong việc viết báo cáo khóa luận.

---

**Chúc bạn hoàn thành báo cáo khóa luận thành công!**

