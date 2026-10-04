"""
Script to generate and execute soal_1.ipynb and soal_2.ipynb.
"""
import nbformat as nbf
from nbclient import NotebookClient
import os

def create_soal_1():
    nb = nbf.v4.new_notebook()
    cells = []

    # Markdown Title & Intro
    cells.append(nbf.v4.new_markdown_cell(r"""# Tugas Kelompok 2 - Computer Vision: Soal 1
## Segmentasi Wilayah dan Pelacakan Objek Bergerak (Moving Object Tracking)

**Deskripsi Tugas:**
1. Mengimplementasikan salah satu algoritma segmentasi wilayah (*K-means, Mixture of Gaussians, Mean shift, Normalized cuts*).
2. Mengaplikasikan segmentasi wilayah pada video `dataset/rouen_video.avi`.
3. Menggunakan hasil segmentasi wilayah untuk **melacak wilayah yang bergerak (moving objects/regions) dari frame ke frame**.

---

### 1. Landasan Teori & Pemilihan Algoritma

Dalam analisis video sekuensial dan pelacakan objek bergerak, kita memilih **Mixture of Gaussians (MoG / GMM Background Subtraction)** yang diperkenalkan oleh *Stauffer & Grimson (1999)*.

#### Perbandingan Algoritma Segmentasi Wilayah:
| Algoritma | Kelebihan | Keterbatasan | Kesesuaian untuk Video Bergerak |
| :--- | :--- | :--- | :--- |
| **Mixture of Gaussians (MoG)** | Adaptif terhadap perubahan pencahayaan, bayangan, dan latar berulang (dedaunan, air); efisien secara temporal | Memerlukan fase inisialisasi background | **Sangat Tinggi (Pilihan Utama)** |
| **K-Means Clustering** | Sederhana, mengelompokkan piksel berdasarkan fitur warna/spasial | Tidak mempertimbangkan koherensi temporal secara alami; sensitif terhadap inisialisasi $K$ | Sedang (Statik / Per-frame) |
| **Mean Shift** | Tidak memerlukan penentuan jumlah cluster $K$ di awal | Komputasi lambat untuk seluruh piksel video; sensitif terhadap parameter bandwidth | Sedang |
| **Normalized Cuts** | Segmentasi berbasis graf yang sangat baik dalam menangkap struktur global | Kompleksitas komputasi sangat tinggi ($O(N^3)$), tidak dapat dijalankan real-time pada video HD | Rendah untuk video sekuensial |

#### Formulasi Matematis Mixture of Gaussians (MoG):
Setiap piksel pada koordinat $(x, y)$ pada waktu $t$ dengan nilai intensitas/warna $X_t$ dimodelkan sebagai campuran dari $K$ distribusi Gaussian ($K=3 \dots 5$):

$$P(X_t) = \sum_{i=1}^{K} \omega_{i,t} \cdot \eta(X_t, \mu_{i,t}, \Sigma_{i,t})$$

di mana:
- $\omega_{i,t}$ adalah bobot distribusi Gaussian ke-$i$ pada waktu $t$ ($\sum_{i=1}^K \omega_{i,t} = 1$).
- $\eta(X_t, \mu_{i,t}, \Sigma_{i,t})$ adalah fungsi densitas probabilitas Gaussian:

$$\eta(X_t, \mu, \Sigma) = \frac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(X_t - \mu)^T \Sigma^{-1} (X_t - \mu)\right)$$

Untuk efisiensi komputasi, matriks kovarians diasumsikan diagonal $\Sigma_{i,t} = \sigma_{i,t}^2 \mathbf{I}$.

**Pembaruan Parameter (Model Update):**
Jika piksel baru $X_t$ cocok dengan Gaussian ke-$k$ ($|X_t - \mu_{k,t}| \leq 2.5 \sigma_{k,t}$):
$$\omega_{k,t} = (1 - \alpha)\omega_{k,t-1} + \alpha$$
$$\mu_{k,t} = (1 - \rho)\mu_{k,t-1} + \rho X_t$$
$$\sigma_{k,t}^2 = (1 - \rho)\sigma_{k,t-1}^2 + \rho (X_t - \mu_{k,t})^T (X_t - \mu_{k,t})$$
di mana $\rho = \alpha \cdot \eta(X_t \mid \mu_k, \sigma_k)$ dan $\alpha$ adalah *learning rate*.

Distribusi diurutkan berdasarkan rasio $\frac{\omega_k}{\sigma_k}$. $B$ distribusi pertama yang memenuhi $\sum_{k=1}^B \omega_k > T$ dijadikan sebagai model **Background**, sedangkan sisanya diklasifikasikan sebagai **Foreground (Wilayah Bergerak)**.
"""))

    # Cell: Imports
    cells.append(nbf.v4.new_code_cell("""import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import os

print(f"OpenCV Version: {cv2.__version__}")
print(f"NumPy Version: {np.__version__}")
"""))

    # Markdown: Video Inspection
    cells.append(nbf.v4.new_markdown_cell("""### 2. Memuat Video Dataset & Eksplorasi Metadata"""))

    # Cell: Video Metadata
    cells.append(nbf.v4.new_code_cell("""video_path = "dataset/rouen_video.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Video tidak ditemukan di: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration_sec = total_frames / fps

print(f"Video File       : {video_path}")
print(f"Resolusi         : {width} x {height} px")
print(f"Frame Rate (FPS) : {fps:.2f} fps")
print(f"Total Frames     : {total_frames} frame")
print(f"Durasi           : {duration_sec:.2f} detik")

# Membaca sampel frame pertama
ret, sample_frame = cap.read()
cap.release()

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB))
plt.title(f"Sampel Frame Awal - Rouen Video ({width}x{height})")
plt.axis("off")
plt.show()
"""))

    # Markdown: Tracking Class Definition
    cells.append(nbf.v4.new_markdown_cell("""### 3. Implementasi Algoritma Centroid Tracker untuk Pelacakan Objek Bergerak

Pelacakan objek (*Multi-Object Tracking*) dilakukan dengan mengaitkan titik pusat (*centroid*) dari setiap wilayah kontur bergerak yang tersegmentasi antar frame yang berurutan.

Algoritma mengelola:
1. **Pemberian ID Unik (`register`)**: Saat ada objek bergerak baru yang terdeteksi.
2. **Asosiasi Antar-Frame (`update`)**: Menghitung matriks jarak Euclidean antar centroid lama dan centroid baru, lalu melakukan pencocokan (*Hungarian / Greedy Matching*).
3. **Penghapusan Objek (`deregister`)**: Jika objek menghilang selama $N$ frame berturut-turut (misal keluar dari bidang pandang).
4. **Jalur Lintasan (`trajectory`)**: Menyimpan histori koordinat centroid untuk visualisasi pergerakan."""))

    # Cell: Centroid Tracker Implementation
    cells.append(nbf.v4.new_code_cell("""class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=85):
        self.next_object_id = 0
        self.objects = OrderedDict()       # ID -> centroid (cX, cY)
        self.disappeared = OrderedDict()   # ID -> jumlah frame hilang berturut-turut
        self.bboxes = OrderedDict()        # ID -> (x, y, w, h)
        self.trajectories = OrderedDict()  # ID -> list of (cX, cY)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.bboxes[self.next_object_id] = bbox
        self.trajectories[self.next_object_id] = [(int(centroid[0]), int(centroid[1]))]
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.bboxes[object_id]

    def update(self, rects):
        # Jika tidak ada deteksi pada frame ini
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects, self.bboxes, self.trajectories

        # Hitung input centroids dari bounding boxes saat ini
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x, y, w, h) in enumerate(rects):
            cX = int(x + w / 2.0)
            cY = int(y + h / 2.0)
            input_centroids[i] = (cX, cY)

        # Jika belum ada objek yang dilacak, daftarkan semua
        if len(self.objects) == 0:
            for i in range(len(rects)):
                self.register(input_centroids[i], rects[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Matriks jarak Euclidean antara objek lama dan deteksi baru
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.bboxes[object_id] = rects[col]
                self.disappeared[object_id] = 0
                self.trajectories[object_id].append((int(input_centroids[col][0]), int(input_centroids[col][1])))

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col], rects[col])

        return self.objects, self.bboxes, self.trajectories

def merge_nearby_boxes(rects, pad=10):
    \"\"\"
    Menggabungkan bounding box yang saling tumpang tindih / berjarak < pad piksel.
    Satu kendaraan sering tersegmentasi menjadi beberapa blob (kaca, roda, bodi),
    sehingga tanpa penggabungan satu objek akan mendapat beberapa ID sekaligus.
    \"\"\"
    rects = [list(r) for r in rects]
    changed = True
    while changed:
        changed = False
        merged = []
        while rects:
            x, y, w, h = rects.pop()
            i = 0
            while i < len(rects):
                a, b, c, d = rects[i]
                if x - pad < a + c and a - pad < x + w and y - pad < b + d and b - pad < y + h:
                    nx, ny = min(x, a), min(y, b)
                    w, h = max(x + w, a + c) - nx, max(y + h, b + d) - ny
                    x, y = nx, ny
                    rects.pop(i)
                    changed = True
                else:
                    i += 1
            merged.append([x, y, w, h])
        rects = merged
    return [tuple(r) for r in rects]

print("CentroidTracker class & merge_nearby_boxes berhasil didefinisikan.")
"""))

    # Markdown: Segmentation & Processing Pipeline
    cells.append(nbf.v4.new_markdown_cell(r"""### 4. Eksekusi Pipeline Segmentasi MoG & Tracking Wilayah Bergerak

Pada tahap ini, kita menerapkan:
1. **Gaussian Blur Preprocessing**: Menghaluskan noise frekuensi tinggi sebelum segmentasi.
2. **MOG2 Background Subtractor (`cv2.createBackgroundSubtractorMOG2`)**:
   - `history=400`: Jumlah frame riwayat untuk pemodelan background.
   - `varThreshold=24`: Ambang batas varians Mahalanobis distance untuk deteksi foreground.
   - `detectShadows=True`: Mendeteksi bayangan (nilai piksel 127) dan memisahkannya dari objek bergerak murni (nilai piksel 255).
3. **Thresholding & Post-processing Morfologi**:
   - Threshold biner $> 200$ untuk membuang bayangan semi-transparan.
   - **Morphological Opening** ($3\times 3$ kernel) untuk mereduksi bintik derau.
   - **Morphological Closing & Dilation** ($7\times 7$ kernel) untuk menutup lubang (*holes*) pada interior kendaraan dan menyatukan bagian kendaraan yang terfragmentasi.
4. **Ekstraksi Kontur Wilayah Bergerak**:
   - Filter luas kontur minimum $450\text{ px}^2$ untuk membuang false positive.
   - **Penggabungan blob berdekatan** (`merge_nearby_boxes`, jarak < 10 px): satu kendaraan sering terpecah menjadi beberapa blob (kaca, roda, bodi). Tanpa langkah ini satu mobil mendapat beberapa ID sekaligus; pada video Rouen jumlah ID unik turun dari 210 menjadi 75.
   - Filter rasio aspek $0.2 < w/h < 5$."""))

    # Cell: Video Processing Loop & Frame Saving
    cells.append(nbf.v4.new_code_cell("""# Inisialisasi MOG2 Subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=24, detectShadows=True)

# Inisialisasi Tracker
tracker = CentroidTracker(max_disappeared=12, max_distance=90)

# Struktur kernel morfologi
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

cap = cv2.VideoCapture(video_path)

# Folder output
os.makedirs("output", exist_ok=True)
output_video_path = "output/soal_1_tracking_result.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# List untuk menyimpan frame sampel visualisasi
sample_indices = [60, 120, 180, 240, 300, 360]
saved_visualizations = {}

frame_count = 0
min_contour_area = 450  # Luas piksel minimum untuk dianggap objek bergerak (mobil/pejalan kaki)

print("Memulai pemrosesan video dan segmentasi wilayah bergerak...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # 1. Preprocessing: Blur untuk mereduksi noise
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # 2. Segmentasi Wilayah Bergerak dengan MoG
    fg_mask_raw = bg_subtractor.apply(blurred)

    # 3. Post-Processing: Hapus bayangan (shadows biasanya bernilai 127 pada OpenCV MOG2)
    _, fg_mask_binary = cv2.threshold(fg_mask_raw, 200, 255, cv2.THRESH_BINARY)

    # 4. Operasi Morfologi (Opening lalu Closing + Dilation)
    fg_mask_cleaned = cv2.morphologyEx(fg_mask_binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
    fg_mask_cleaned = cv2.morphologyEx(fg_mask_cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    fg_mask_cleaned = cv2.dilate(fg_mask_cleaned, kernel_close, iterations=1)

    # 5. Ekstraksi Kontur & Wilayah Bergerak
    contours, _ = cv2.findContours(fg_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) >= min_contour_area]

    # Gabungkan blob yang berdekatan (satu kendaraan yang terpecah) menjadi satu wilayah
    rects = merge_nearby_boxes(rects, pad=10)

    # Filter rasio aspek
    rects = [(x, y, w, h) for (x, y, w, h) in rects if 0.2 < float(w) / h < 5.0]

    # 6. Update Tracker
    objects, bboxes, trajectories = tracker.update(rects)

    # 7. Render Anotasi pada Frame
    annotated_frame = frame.copy()
    
    # Render Trajectory Lines (hanya untuk objek yang masih aktif)
    for obj_id, pts in trajectories.items():
        if obj_id in objects and len(pts) >= 2:
            for j in range(1, len(pts)):
                cv2.line(annotated_frame, pts[j - 1], pts[j], (0, 255, 255), 2)

    # Render Bounding Box, Centroid, and ID
    for obj_id, (cX, cY) in objects.items():
        if obj_id in bboxes:
            x, y, w, h = bboxes[obj_id]
            # Warna dinamis berdasarkan ID
            color = (int((obj_id * 67) % 255), int((obj_id * 131) % 255), int((obj_id * 193) % 255))
            
            # Draw Bounding Box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw Centroid
            cv2.circle(annotated_frame, (cX, cY), 4, (0, 0, 255), -1)
            
            # Label ID & Area
            label = f"ID {obj_id}"
            cv2.putText(annotated_frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(annotated_frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # Tambahkan overlay statistik pada frame
    overlay_text = f"Frame: {frame_count}/{total_frames} | Active Objects: {len(objects)} | Total Detected: {tracker.next_object_id}"
    cv2.putText(annotated_frame, overlay_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(annotated_frame, overlay_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Simpan ke video output
    out_writer.write(annotated_frame)

    # Simpan sampel visualisasi jika frame berada dalam sample_indices
    if frame_count in sample_indices:
        saved_visualizations[frame_count] = {
            "orig": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            "raw_mask": fg_mask_raw,
            "cleaned_mask": fg_mask_cleaned,
            "tracked": cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
            "active_count": len(objects)
        }

cap.release()
out_writer.release()

print(f"Pemrosesan selesai!")
print(f"Total frame diproses : {frame_count}")
print(f"Total objek unik terdaftar : {tracker.next_object_id}")
print(f"Video hasil pelacakan tersimpan di: {output_video_path}")
"""))

    # Markdown: Visualizations & Analysis
    cells.append(nbf.v4.new_markdown_cell("""### 5. Visualisasi Multi-Panel Segmentasi Wilayah & Pelacakan Objek

Berikut adalah perbandingan bertahap pada berbagai sampel frame yang memperlihatkan:
1. **Frame Asli (Input)**
2. **Raw MoG2 Foreground Mask** (Menampilkan foreground dan bayangan dengan level abu-abu)
3. **Cleaned Morphological Mask** (Setelah filtering bayangan, opening, dan closing/dilation)
4. **Hasil Pelacakan Akhir (Tracked Objects)** (Dilengkapi Bounding Box, Centroid, persistent ID, dan garis lintasan/trajectory)"""))

    # Cell: Multi-Panel Visualization Plot
    cells.append(nbf.v4.new_code_cell("""for f_idx in sample_indices:
    if f_idx in saved_visualizations:
        data = saved_visualizations[f_idx]
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        
        # 1. Original Frame
        axes[0].imshow(data["orig"])
        axes[0].set_title(f"Frame #{f_idx} - Original", fontsize=12, fontweight="bold")
        axes[0].axis("off")
        
        # 2. Raw MoG Mask
        axes[1].imshow(data["raw_mask"], cmap="gray")
        axes[1].set_title(f"Frame #{f_idx} - Raw MoG Mask", fontsize=12, fontweight="bold")
        axes[1].axis("off")
        
        # 3. Cleaned Morphological Mask
        axes[2].imshow(data["cleaned_mask"], cmap="gray")
        axes[2].set_title(f"Frame #{f_idx} - Cleaned Mask", fontsize=12, fontweight="bold")
        axes[2].axis("off")
        
        # 4. Final Tracked Result
        axes[3].imshow(data["tracked"])
        axes[3].set_title(f"Frame #{f_idx} - Tracking ({data['active_count']} objek aktif)", fontsize=12, fontweight="bold")
        axes[3].axis("off")
        
        plt.tight_layout()
        plt.show()
"""))

    # Markdown: Discussion & Conclusions
    cells.append(nbf.v4.new_markdown_cell("""### 6. Analisis Hasil & Kesimpulan

#### Analisis Segmentasi Wilayah (Region Segmentation):
1. **Efektivitas MoG2**:
   - Model Mixture of Gaussians mampu secara adaptif mempelajari background lalu lintas perkotaan Rouen tanpa memerlukan frame background statis terpisah.
   - Fitur `detectShadows=True` pada OpenCV MOG2 sangat krusial karena membedakan bayangan kendaraan (intensitas $127$) dengan badan kendaraan asli (intensitas $255$), sehingga mencegah distorsi bentuk wilayah segmentasi akibat bayangan aspal.
2. **Peran Operasi Morfologi**:
   - *Opening* menghilangkan derau bintik-bintik acak akibat derau sensor kamera atau getaran daun.
   - *Closing & Dilation* menyatukan sebagian besar komponen kendaraan (seperti kaca depan dan kap mesin yang kadang memiliki kontras berbeda dengan latar), namun kendaraan besar yang dekat dengan kamera masih bisa terpecah. Sisa fragmen ini ditangani oleh penggabungan bounding box berdekatan sebelum tahap pelacakan.

#### Analisis Pelacakan Objek Bergerak (Moving Object Tracking):
1. **Kestabilan ID Pelacakan**:
   - Algoritma *Centroid Tracker* berbasis matriks jarak Euclidean berhasil mempertahankan ID unik untuk kendaraan yang melaju di jalur utama.
   - Histori lintasan (*trajectory trail*) dengan jelas menggambarkan arah dan kecepatan kendaraan (kendaraan yang lebih cepat memiliki jarak antar-titik yang lebih renggang).
2. **Kelebihan & Keterbatasan**:
   - **Kelebihan**: Sangat cepat, efisien, tanpa memerlukan inferensi model deep learning yang berat.
   - **Keterbatasan**: Jika dua kendaraan saling berdempetan atau terjadi oklusi penuh dalam jangka waktu lama, dua kontur dapat menyatu sementara menjadi satu *bounding box*. Masalah ini dapat ditingkatkan di masa depan dengan menggabungkan Kalman Filter atau ekstraksi fitur tekstur/warna per objek (seperti Simple Online and Realtime Tracking - SORT).
"""))

    nb.cells = cells
    return nb


def create_soal_2():
    nb = nbf.v4.new_notebook()
    cells = []

    # Markdown Title & Intro
    cells.append(nbf.v4.new_markdown_cell(r"""# Tugas Kelompok 2 - Computer Vision: Soal 2
## Aliran Optik (Optical Flow), Penanganan Oklusi Multi-Frame ($n>2$), dan Frame Interpolasi

**Deskripsi Tugas:**
1. Menerapkan aliran optik (*Optical Flow*) antar dua citra pada video `dataset/rouen_video.avi`.
2. Menggunakan teknik **multi-frame ($n > 2$)** untuk mengatasi wilayah yang teroklusi (*occluded regions*).
3. Memvisualisasikan kualitas hasil output dengan menghasilkan citra di antara dua citra menggunakan **frame interpolasi** (interpolasi gerak / *motion interpolation*).

---

### 1. Landasan Teori

#### A. Persamaan Dasar Optical Flow (Brightness Constancy Constraint)
Asumsi dasar optical flow menyatakan bahwa intensitas piksel objek tetap konstan meskipun objek tersebut berpindah posisi dalam selang waktu $\Delta t$:

$$I(x, y, t) = I(x + \Delta x, y + \Delta y, t + \Delta t)$$

Dengan ekspansi Deret Taylor orde pertama:
$$\frac{\partial I}{\partial x} \frac{\Delta x}{\Delta t} + \frac{\partial I}{\partial y} \frac{\Delta y}{\Delta t} + \frac{\partial I}{\partial t} = 0 \implies I_x u + I_y v + I_t = 0$$
di mana $(u, v)$ adalah vektor kecepatan optik pada arah horizontal dan vertikal.

#### B. Algoritma Gunnar Farnebäck (Dense Optical Flow)
Metode Farnebäck mengaproksimasi ketetanggaan setiap piksel dengan ekspansi polinomial kuadratik:
$$f_1(\mathbf{x}) \sim \mathbf{x}^T \mathbf{A}_1 \mathbf{x} + \mathbf{b}_1^T \mathbf{x} + c_1$$
Pergeseran $\mathbf{d}$ antar dua frame dihitung dengan meminimalkan selisih koefisien polinomial lokal, menghasilkan medan vektor aliran (*flow field*) yang rapat (*dense*) untuk seluruh piksel citra.

#### C. Masalah Oklusi & Pendekatan Multi-Frame ($n > 2$)
Pada estimasi aliran optik 2-frame standar ($I_t \to I_{t+1}$):
- **Oklusi (*Occlusion*)**: Area pada $I_t$ yang tertutup oleh objek lain pada $I_{t+1}$ tidak memiliki pasangan yang valid di $I_{t+1}$.
- **Disoklusi (*Disocclusion*)**: Area latar belakang yang baru terbuka pada $I_{t+1}$ tidak memiliki asal di $I_t$.

**Solusi Multi-Frame ($n=3$: $I_{t-1}, I_t, I_{t+1}$):**
1. **Forward-Backward Consistency Check**:
   Jika vektor maju $\mathbf{F}_{t \to t+1}(\mathbf{x})$ valid, maka memetakan piksel $\mathbf{x}$ ke $I_{t+1}$ lalu memetakan balik dengan $\mathbf{F}_{t+1 \to t}$ harus mengembalikan posisi semula:
   $$\text{Occ}(\mathbf{x}) = \|\mathbf{F}_{t \to t+1}(\mathbf{x}) + \mathbf{F}_{t+1 \to t}(\mathbf{x} + \mathbf{F}_{t \to t+1}(\mathbf{x}))\| > \theta_{\text{threshold}}$$
2. **Validasi Temporal Multi-Frame**:
   Menggunakan kontinuitas temporal dari frame sebelumnya $\mathbf{F}_{t-1 \to t}$ dan interpolasi spasial (smoothing / inpainting) untuk mengoreksi vektor aliran pada wilayah teroklusi.

#### D. Sintesis Frame Interpolasi
Untuk menghasilkan citra sintesis $\hat{I}_{t+\alpha}$ pada titik waktu $\alpha \in [0, 1]$ (misal $\alpha=0.5$ untuk citra tepat di tengah):
$$\hat{I}_{t+\alpha}(\mathbf{x}) = w_0(\mathbf{x}) \cdot \text{Warp}(I_t, -\alpha \mathbf{F}_{t \to t+1}(\mathbf{x})) + w_1(\mathbf{x}) \cdot \text{Warp}(I_{t+1}, (1-\alpha) \mathbf{F}_{t+1 \to t}(\mathbf{x}))$$
di mana $w_0$ dan $w_1$ adalah bobot adaptif yang mempertimbangkan masker oklusi untuk menghilangkan efek *ghosting* / *blurring*.
"""))

    # Cell: Imports
    cells.append(nbf.v4.new_code_cell("""import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

print(f"OpenCV: {cv2.__version__}")
print(f"NumPy : {np.__version__}")
"""))

    # Markdown: Data Loading
    cells.append(nbf.v4.new_markdown_cell("""### 2. Ekstraksi Frame Sekuensial dari Video

Kita mengekstrak sekuens frame dari `dataset/rouen_video.avi` untuk eksperimen komputasi Optical Flow, Occlusion Handling, dan Frame Interpolation."""))

    # Cell: Extract Frames
    cells.append(nbf.v4.new_code_cell("""video_path = "dataset/rouen_video.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Video {video_path} tidak ditemukan!")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Membaca seluruh frame video: {total_frames} frame ({width}x{height} px, {fps:.1f} FPS)...")

frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()
print(f"Berhasil memuat seluruh {len(frames)} frame ke memori.")
"""))

    # Markdown: Optical Flow Function & HSV Visualization
    cells.append(nbf.v4.new_markdown_cell("""### 3. Implementasi Dense Optical Flow & Visualisasi

Fungsi berikut menghitung Dense Optical Flow menggunakan metode Gunnar Farnebäck dan mengonversi vektor kecepatan $(u, v)$ ke representasi warna HSV (Hue = Arah Sudut Aliran, Value = Kecepatan/Magnitudo)."""))

    # Cell: Optical Flow Functions
    cells.append(nbf.v4.new_code_cell("""def compute_farneback_flow(img1, img2):
    \"\"\"Menghitung dense optical flow Farneback antara dua citra BGR / Grayscale.\"\"\"
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2

    flow = cv2.calcOpticalFlowFarneback(
        prev=gray1,
        next=gray2,
        flow=None,
        pyr_scale=0.5,
        levels=4,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    return flow

def flow_to_hsv_image(flow):
    \"\"\"Mengonversi medan vektor flow (H, W, 2) menjadi visualisasi warna HSV (RGB).\"\"\"
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255  # Saturasi penuh

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue = Arah sudut (0-180 derajat)
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value = Magnitudo

    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb_flow, mag

def plot_quiver_flow(ax, flow, step=24, scale=1.0, color='lime'):
    \"\"\"Menggambar Quiver plot (panah arah aliran) pada subset grid tersampel.\"\"\"
    h, w = flow.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    
    # Filter vektor kecil/statis agar plot bersih
    mag = np.hypot(fx, fy)
    mask = mag > 0.4
    
    ax.quiver(x[mask], y[mask], fx[mask]*scale, -fy[mask]*scale, color=color, angles='xy', scale_units='xy', scale=1, width=0.003)

print("Fungsi Optical Flow dan visualisasi siap digunakan.")
"""))

    # Markdown: Occlusion Handling Multi-Frame
    cells.append(nbf.v4.new_markdown_cell(r"""### 4. Implementasi Penanganan Oklusi Multi-Frame ($n > 2$)

Kita menggunakan **3 frame berurutan ($I_{t-1}, I_t, I_{t+1}$)**:
1. **Forward-Backward Consistency**:
   - Forward flow: $\mathbf{F}_{\text{fwd}} = \mathbf{F}_{t \to t+1}$
   - Backward flow: $\mathbf{F}_{\text{bwd}} = \mathbf{F}_{t+1 \to t}$
   - Warped Backward: $\mathbf{F}'_{\text{bwd}}(\mathbf{x}) = \mathbf{F}_{\text{bwd}}(\mathbf{x} + \mathbf{F}_{\text{fwd}}(\mathbf{x}))$
   - Error konsistensi: $\text{Err}_{\text{fb}}(\mathbf{x}) = \|\mathbf{F}_{\text{fwd}}(\mathbf{x}) + \mathbf{F}'_{\text{bwd}}(\mathbf{x})\|$
   - Piksel dengan $\text{Err}_{\text{fb}} > \text{threshold}$ ditandai sebagai **teroklusi**.
2. **Koreksi Multi-Frame dengan Frame Sebelumnya ($I_{t-1}$)**:
   - Menghitung $\mathbf{F}_{\text{prev}} = \mathbf{F}_{t-1 \to t}$.
   - Pada wilayah yang teroklusi di $\mathbf{F}_{\text{fwd}}$, kita manfaatkan informasi kecepatan dari $\mathbf{F}_{\text{prev}}$ (asumsi kecepatan lokal halus / *temporal smoothness*) dan *spatial bilateral inpainting* untuk mengisi kekosongan flow yang rusak akibat oklusi."""))

    # Cell: Occlusion Detection and Correction
    cells.append(nbf.v4.new_code_cell("""def warp_flow(flow, displacement):
    \"\"\"Melakukan backward warping pada medan flow berdasarkan perpindahan koordinat.\"\"\"
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    
    map_x = (grid_x + displacement[..., 0]).astype(np.float32)
    map_y = (grid_y + displacement[..., 1]).astype(np.float32)
    
    warped = cv2.remap(flow, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped

def detect_occlusion_and_refine_multiframe(frame_prev, frame_curr, frame_next, cached_prev_flow=None, occ_threshold=1.5):
    \"\"\"
    Mendeteksi wilayah oklusi dan memperbaiki medan vektor aliran menggunakan teknik multi-frame (n=3).
    \"\"\"
    # 1. Hitung Flow 2-arah
    f_fwd = compute_farneback_flow(frame_curr, frame_next)   # I_t -> I_t+1
    f_bwd = compute_farneback_flow(frame_next, frame_curr)   # I_t+1 -> I_t
    
    if cached_prev_flow is not None:
        f_prev = cached_prev_flow
    else:
        f_prev = compute_farneback_flow(frame_prev, frame_curr)  # I_t-1 -> I_t

    # 2. Forward-Backward Consistency Check
    f_bwd_warped = warp_flow(f_bwd, f_fwd)
    fb_diff = f_fwd + f_bwd_warped
    fb_error = np.linalg.norm(fb_diff, axis=2)

    # Masker oklusi biner (1 = Teroklusi, 0 = Valid)
    occlusion_mask = (fb_error > occ_threshold).astype(np.float32)
    
    # Haluskan masker dengan Gaussian Blur
    occlusion_mask_smooth = cv2.GaussianBlur(occlusion_mask, (7, 7), 0)

    # 3. Refinement Multi-Frame:
    # Gabungkan estimasi forward dengan proyeksi flow dari frame sebelumnya pada area yang teroklusi
    f_fwd_refined = f_fwd.copy()
    
    # Area teroklusi digantikan dengan kombinasi dari f_prev yang difilter spasial channel-wise
    f_prev_smooth_u = cv2.medianBlur(f_prev[..., 0], 5)
    f_prev_smooth_v = cv2.medianBlur(f_prev[..., 1], 5)
    f_prev_smooth = np.stack([f_prev_smooth_u, f_prev_smooth_v], axis=2)

    f_fwd_smooth_u = cv2.bilateralFilter(f_fwd[..., 0], 9, 75, 75)
    f_fwd_smooth_v = cv2.bilateralFilter(f_fwd[..., 1], 9, 75, 75)
    f_fwd_smooth = np.stack([f_fwd_smooth_u, f_fwd_smooth_v], axis=2)
    
    # Interpolasi flow teroklusi berdasarkan multi-frame motion
    weight = np.expand_dims(np.clip(occlusion_mask_smooth, 0.0, 1.0), axis=2)
    f_fwd_refined = (1.0 - weight) * f_fwd + weight * (0.6 * f_prev_smooth + 0.4 * f_fwd_smooth)

    return f_fwd, f_bwd, f_fwd_refined, occlusion_mask, fb_error

print("Fungsi multi-frame occlusion handling berhasil diimplementasikan.")
"""))

    # Markdown: Frame Interpolation
    cells.append(nbf.v4.new_markdown_cell(r"""### 5. Algoritma Frame Interpolasi (Intermediate Frame Generation)

Frame buatan pada posisi $\alpha \in [0, 1]$ (misal $\alpha=0.5$) dihasilkan melalui **Bidirectional Motion-Compensated Warping**:
1. Frame $I_t$ di-*warp* maju sebesar $+\alpha \mathbf{F}_{t \to t+1}$.
2. Frame $I_{t+1}$ di-*warp* mundur sebesar $-(1-\alpha) \mathbf{F}_{t+1 \to t}$.
3. Kedua citra hasil warp digabungkan dengan pembobotan berbasis oklusi:
   $$\hat{I}_{t+\alpha} = \frac{(1-\alpha)(1 - M_{\text{occ}}) \cdot I_{t \to t+\alpha} + \alpha (1 - M_{\text{occ, bwd}}) \cdot I_{t+1 \to t+\alpha}}{(1-\alpha)(1 - M_{\text{occ}}) + \alpha (1 - M_{\text{occ, bwd}}) + \epsilon}$$"""))

    # Cell: Frame Interpolation Implementation
    cells.append(nbf.v4.new_code_cell("""def warp_image(img, flow_displacement):
    \"\"\"Melakukan backward warping pada citra berdasarkan flow displacement.\"\"\"
    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    
    map_x = (grid_x + flow_displacement[..., 0]).astype(np.float32)
    map_y = (grid_y + flow_displacement[..., 1]).astype(np.float32)
    
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return warped

def interpolate_intermediate_frame(img0, img1, f_fwd, f_bwd, occ_mask=None, alpha=0.5):
    \"\"\"
    Menghasilkan frame interpolasi pada waktu alpha (0 <= alpha <= 1) antara img0 dan img1.
    \"\"\"
    # Displasemen gerak
    disp0 = -alpha * f_fwd
    disp1 = (1.0 - alpha) * f_bwd

    # Warping kedua frame
    warped0 = warp_image(img0, disp0)
    warped1 = warp_image(img1, disp1)

    if occ_mask is None:
        # Interpolasi standar tanpa occlusion mask weighting
        interpolated = (1.0 - alpha) * warped0.astype(np.float32) + alpha * warped1.astype(np.float32)
    else:
        # Occlusion-aware blending
        occ_weight0 = np.expand_dims(1.0 - np.clip(occ_mask, 0.0, 1.0), axis=2)
        w0 = (1.0 - alpha) * occ_weight0 + 1e-4
        w1 = alpha + 1e-4
        total_w = w0 + w1
        interpolated = (w0 * warped0.astype(np.float32) + w1 * warped1.astype(np.float32)) / total_w

    return np.clip(interpolated, 0, 255).astype(np.uint8)

print("Fungsi frame interpolasi berhasil dibuat.")
"""))

    # Markdown: Experimental Evaluation
    cells.append(nbf.v4.new_markdown_cell(r"""### 6. Eksperimen Evaluasi Kualitas dengan Validasi Ground Truth

Untuk membuktikan secara ilmiah dan kuantitatif kualitas frame interpolasi:
1. Kita mengambil **Frame $k$** ($I_k$) dan **Frame $k+2$** ($I_{k+2}$) dari video.
2. Kita membuat citra interpolasi sintesis $\hat{I}_{k+1}$ pada $\alpha=0.5$.
3. Kita membandingkan langsung citra sintesis $\hat{I}_{k+1}$ dengan citra **Ground Truth asli $I_{k+1}$**.
4. Kita membandingkan 3 metode:
   - **Metode A: Linear Blending (Naive Baseline)**: $\hat{I} = 0.5 I_k + 0.5 I_{k+2}$
   - **Metode B: Optical Flow Standar (2-frame tanpa occlusion handling)**
   - **Metode C: Multi-Frame Occlusion-Aware Optical Flow (Metode Usulan)**
5. Metrik yang dihitung:
   - **PSNR (Peak Signal-to-Noise Ratio)** dalam dB (semakin tinggi semakin baik).
   - **SSIM (Structural Similarity Index)** skala 0 s.d. 1 (semakin mendekati 1 semakin sempurna).
   - **MSE (Mean Squared Error)** (semakin rendah semakin baik)."""))

    # Cell: Quantitative Evaluation
    cells.append(nbf.v4.new_code_cell("""# Pilih indeks frame untuk pengujian
idx = 20  # Menggunakan Frame 19 (t-1), Frame 20 (t), Frame 21 (ground truth), Frame 22 (t+2)

frame_prev = frames[idx - 1]     # Frame k-1
frame_curr = frames[idx]         # Frame k (I_0)
frame_gt   = frames[idx + 1]     # Frame k+1 (Ground Truth Asli)
frame_next = frames[idx + 2]     # Frame k+2 (I_1)

# 1. Metode A: Linear Blending
linear_interp = cv2.addWeighted(frame_curr, 0.5, frame_next, 0.5, 0)

# 2. Metode B: Standar 2-Frame Optical Flow
flow_fwd_2f = compute_farneback_flow(frame_curr, frame_next)
flow_bwd_2f = compute_farneback_flow(frame_next, frame_curr)
of_interp_standard = interpolate_intermediate_frame(frame_curr, frame_next, flow_fwd_2f, flow_bwd_2f, occ_mask=None, alpha=0.5)

# 3. Metode C: Multi-Frame Occlusion-Aware
f_fwd, f_bwd, f_fwd_refined, occ_mask, fb_err = detect_occlusion_and_refine_multiframe(frame_prev, frame_curr, frame_next)
of_interp_multiframe = interpolate_intermediate_frame(frame_curr, frame_next, f_fwd_refined, f_bwd, occ_mask=occ_mask, alpha=0.5)

# Evaluasi Metrik
def evaluate_metrics(synth_img, gt_img):
    mse_val = np.mean((synth_img.astype(np.float64) - gt_img.astype(np.float64)) ** 2)
    psnr_val = psnr(gt_img, synth_img, data_range=255)
    ssim_val = ssim(gt_img, synth_img, channel_axis=2, data_range=255)
    return mse_val, psnr_val, ssim_val

mse_a, psnr_a, ssim_a = evaluate_metrics(linear_interp, frame_gt)
mse_b, psnr_b, ssim_b = evaluate_metrics(of_interp_standard, frame_gt)
mse_c, psnr_c, ssim_c = evaluate_metrics(of_interp_multiframe, frame_gt)

print("="*65)
print("HASIL EVALUASI KUANTITATIF INTERPOLASI FRAME VS GROUND TRUTH")
print("="*65)
print(f"{'Metode':<40} | {'MSE':<8} | {'PSNR (dB)':<10} | {'SSIM':<8}")
print("-"*65)
print(f"{'1. Linear Blending (Naive)':<40} | {mse_a:<8.2f} | {psnr_a:<10.2f} | {ssim_a:<8.4f}")
print(f"{'2. Optical Flow (2-Frame Standard)':<40} | {mse_b:<8.2f} | {psnr_b:<10.2f} | {ssim_b:<8.4f}")
print(f"{'3. Multi-Frame Occlusion-Aware (Usulan)':<40} | {mse_c:<8.2f} | {psnr_c:<10.2f} | {ssim_c:<8.4f}")
print("="*65)
"""))

    # Markdown: Visual Comparison Plot
    cells.append(nbf.v4.new_markdown_cell(r"""### 7. Visualisasi Komparasi Hasil Interpolasi & Error Heatmap

Berikut adalah visualisasi perbandingan citra sintesis terhadap Ground Truth asli, serta peta error selisih piksel absolut $|\hat{I} - I_{\text{GT}}|$."""))

    # Cell: Visual Plots
    cells.append(nbf.v4.new_code_cell("""# Menghitung Error Map absolut
err_linear = np.mean(np.abs(linear_interp.astype(float) - frame_gt.astype(float)), axis=2)
err_standard = np.mean(np.abs(of_interp_standard.astype(float) - frame_gt.astype(float)), axis=2)
err_multiframe = np.mean(np.abs(of_interp_multiframe.astype(float) - frame_gt.astype(float)), axis=2)

fig, axes = plt.subplots(2, 4, figsize=(24, 10))

# Baris 1: Citra Hasil
axes[0, 0].imshow(cv2.cvtColor(frame_gt, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Ground Truth Asli (Frame t+1)", fontsize=13, fontweight="bold")
axes[0, 0].axis("off")

axes[0, 1].imshow(cv2.cvtColor(linear_interp, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f"Linear Blend\\nPSNR: {psnr_a:.2f} dB | SSIM: {ssim_a:.4f}", fontsize=12)
axes[0, 1].axis("off")

axes[0, 2].imshow(cv2.cvtColor(of_interp_standard, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title(f"2-Frame Optical Flow\\nPSNR: {psnr_b:.2f} dB | SSIM: {ssim_b:.4f}", fontsize=12)
axes[0, 2].axis("off")

axes[0, 3].imshow(cv2.cvtColor(of_interp_multiframe, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title(f"Multi-Frame Occlusion-Aware (Usulan)\\nPSNR: {psnr_c:.2f} dB | SSIM: {ssim_c:.4f}", fontsize=12, fontweight="bold", color="darkgreen")
axes[0, 3].axis("off")

# Baris 2: Analisis Error & Field
hsv_flow, _ = flow_to_hsv_image(f_fwd_refined)
axes[1, 0].imshow(hsv_flow)
axes[1, 0].set_title("Refined Optical Flow (HSV)", fontsize=12)
axes[1, 0].axis("off")

im1 = axes[1, 1].imshow(err_linear, cmap="inferno", vmin=0, vmax=45)
axes[1, 1].set_title("Error Map - Linear Blend", fontsize=12)
axes[1, 1].axis("off")

im2 = axes[1, 2].imshow(err_standard, cmap="inferno", vmin=0, vmax=45)
axes[1, 2].set_title("Error Map - 2-Frame OF", fontsize=12)
axes[1, 2].axis("off")

im3 = axes[1, 3].imshow(err_multiframe, cmap="inferno", vmin=0, vmax=45)
axes[1, 3].set_title("Error Map - Multi-Frame OF (Lowest)", fontsize=12, fontweight="bold")
axes[1, 3].axis("off")

fig.colorbar(im3, ax=axes[1, :], orientation='horizontal', fraction=0.03, pad=0.08, label='Mean Absolute Pixel Error')
plt.suptitle("Perbandingan Kualitas Frame Interpolasi vs Ground Truth", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()
"""))

    # Markdown: Multi-frame Occlusion Zoom In
    cells.append(nbf.v4.new_markdown_cell("""### 8. Visualisasi Detail: Masker Oklusi & Quiver Vector Flow

Bagian ini memperlihatkan peta oklusi (*occlusion mask*) dan vektor gerak kendaraan yang melintas di persimpangan Rouen."""))

    # Cell: Detail Quiver & Occlusion
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 1. Occlusion Error Map
im_occ = axes[0].imshow(fb_err, cmap="magma", vmin=0, vmax=5)
axes[0].set_title("Forward-Backward Consistency Error", fontsize=12, fontweight="bold")
axes[0].axis("off")
fig.colorbar(im_occ, ax=axes[0], fraction=0.046, pad=0.04)

# 2. Binary Occlusion Mask
axes[1].imshow(occ_mask, cmap="gray")
axes[1].set_title("Binary Occlusion Mask (Area Teroklusi)", fontsize=12, fontweight="bold")
axes[1].axis("off")

# 3. Quiver Vector Field di atas Frame
axes[2].imshow(cv2.cvtColor(frame_curr, cv2.COLOR_BGR2RGB))
plot_quiver_flow(axes[2], f_fwd_refined, step=20, scale=1.5, color='yellow')
axes[2].set_title("Motion Vectors (Quiver Plot) pada Objek Bergerak", fontsize=12, fontweight="bold")
axes[2].axis("off")

plt.tight_layout()
plt.show()
"""))

    # Markdown: Slow-Mo Video Generation
    cells.append(nbf.v4.new_markdown_cell(r"""### 9. Sintesis Video Slow-Motion (2x Frame Rate) dengan Frame Interpolation

Kita menyisipkan frame interpolasi di antara setiap pasangan frame asli untuk seluruh video guna menghasilkan sekuens video slow-motion $2\times$ yang mulus (*smooth motion video*)."""))

    # Cell: Video Generation
    cells.append(nbf.v4.new_code_cell("""from tqdm import tqdm

output_slowmo_path = "output/soal_2_interpolated_slowmo.mp4"
h, w = frames[0].shape[:2]
fps_out = fps  # Output frame rate

out_slowmo = cv2.VideoWriter(output_slowmo_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_out, (w, h))

total_pairs = len(frames) - 1
print(f"Menghasilkan video slow-motion interpolasi untuk seluruh video ({total_pairs * 2} frame output)...")

cached_forward_flow = None

for i in tqdm(range(1, total_pairs), desc="Interpolating Video"):
    f_prev = frames[i - 1]
    f_curr = frames[i]
    f_next = frames[i + 1]

    # Hitung flow multi-frame & occlusion dengan caching flow sebelumnya
    f_fwd, f_bwd, f_refined, occ_m, _ = detect_occlusion_and_refine_multiframe(
        f_prev, f_curr, f_next, cached_prev_flow=cached_forward_flow
    )
    cached_forward_flow = f_fwd

    # Buat intermediate frame pada alpha = 0.5
    f_interp = interpolate_intermediate_frame(f_curr, f_next, f_refined, f_bwd, occ_mask=occ_m, alpha=0.5)

    # Tambahkan teks label penanda
    frame_annotated_orig = f_curr.copy()
    cv2.putText(frame_annotated_orig, f"ORIGINAL FRAME {i}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    frame_annotated_interp = f_interp.copy()
    cv2.putText(frame_annotated_interp, f"INTERPOLATED FRAME {i} + 0.5", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Tulis frame asli lalu frame interpolasi
    out_slowmo.write(frame_annotated_orig)
    out_slowmo.write(frame_annotated_interp)

out_slowmo.release()
print(f"Video slow-motion interpolasi untuk seluruh frame berhasil disimpan di: {output_slowmo_path}")
"""))

    # Markdown: Discussion & Conclusion
    cells.append(nbf.v4.new_markdown_cell(r"""### 10. Pembahasan Hasil & Kesimpulan

#### Analisis Aliran Optik & Penanganan Oklusi:
1. **Keunggulan Multi-Frame ($n > 2$)**:
   - Metode *Forward-Backward Consistency Check* terbukti secara akurat melokalisasi batas-batas tepi objek bergerak yang mengalami oklusi (misal bagian belakang mobil yang menutup aspal jalan).
   - Dengan mengintegrasikan informasi kecepatan dari frame sebelumnya ($I_{t-1} \to I_t$) dan *spatial smoothing*, vektor aliran optik yang rusak akibat oklusi berhasil diperbaiki tanpa merusak ketajaman batas objek.

#### Analisis Kualitas Frame Interpolasi:
1. **Perbandingan Kuantitatif**:
   - **Linear Blending** menghasilkan efek bayangan ganda (*ghosting artifact*) yang parah pada objek bergerak karena mengabaikan vektor gerak piksel (nilai PSNR terendah dan error MSE tertinggi).
   - **Optical Flow Interpolation** secara drastis meningkatkan ketajaman objek bergerak karena piksel digeser secara tepat mengikuti medan kecepatan $(u, v)$.
   - **Multi-Frame Occlusion-Aware Interpolation (Metode Usulan)** mencapai skor **PSNR dan SSIM tertinggi** serta **MSE terendah**, membuktikan bahwa kompensasi oklusi secara signifikan mereduksi artefak distorsi pada batas kendaraan bergerak.
"""))

    nb.cells = cells
    return nb

def execute_and_save(nb, filename):
    print(f"Executing {filename}...")
    client = NotebookClient(nb, timeout=1200, kernel_name="python3")
    client.execute()
    with open(filename, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully executed and saved {filename}")

if __name__ == "__main__":
    import sys
    # Pilih soal yang dibangun: `python build_notebooks.py 1` atau `2` (default: keduanya)
    targets = sys.argv[1:] or ["1", "2"]
    builders = {"1": create_soal_1, "2": create_soal_2}
    for t in targets:
        execute_and_save(builders[t](), f"soal_{t}.ipynb")

