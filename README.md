<b><h1>Implementation Combination Architecture of ResUnet++ with ViT for Segmentation Semantics Retina Arteries-Veins</h1>
<h2> Online supplementary material for paper "Artery-Vein Segmentation in Fundus Images using a Fully Convolutional Network" by Hemelings R., Elen B., Stalmans I., Van Keer K., De Boever P., Blaschko M.B. Another paper "Semantic segmentation of artery-venous retinal vessel using simple convolutional neural network" by W. Setiawan, M. I. Utoyo, R. Rulaningtyas, A. Wicaksono </h2>
</b>
<h2> Dataset</h2>
https://github.com/rubenhx/av-segmentation/tree/master/DRIVE_AV_evalmasks/Predicted_AV
<h2>The Program</h2>
<h3> Input the Library yang dibutuhkan</h3>
import os<br>
from enum import Enum<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
from PIL import Image<br>
from tensorflow.keras.utils import to_categorical<br>
from sklearn.model_selection import train_test_split<br>
from tqdm import tqdm<br>
<h3> Fungsi untuk menampilkan gambar Citra, Ground Truth dan Prediction</h3>
def display_images(instances, rows, titles=None, figsize=(20, 20)):<br>
    fig, axes = plt.subplots(rows, len(instances) // rows, figsize=figsize)<br>
    for j, image in enumerate(instances):<br>
        plt.subplot(rows, len(instances) // rows, j + 1)<br>
        plt.imshow(image)<br>
        plt.title('' if titles is None else titles[j])<br>
        plt.axis("off")<br>
    plt.show()<br>
<h3>Fungsi One Hot Encode Masks berfungsi untuk mengubah masker (masks) yang berisi nilai-nilai RGB menjadi label-label yang diencode secara integer, dan kemudian mengonversinya menjadi representasi one-hot encoding </h3>
def one_hot_encode_masks(masks, num_classes):<br>
    integer_encoded_labels = []<br>
<br>
    for mask in tqdm(masks):<br>
        _img_height, _img_width, _img_channels = mask.shape<br>
        encoded_image = np.zeros((_img_height, _img_width, 1)).astype(int)<br>
<br>
        for j, cls in enumerate(MaskColorMap):<br>
            encoded_image[np.all(mask == cls.value, axis=-1)] = j<br>
<br>
        integer_encoded_labels.append(encoded_image)<br>
<br>
    return to_categorical(y=integer_encoded_labels, num_classes=num_classes)<br>
<h3> Class untuk Kelas warna yang digunakan dalam melakukan segmentasi, Warna berupa nilai RGB</h3>
class MaskColorMap(Enum): <br>
    red = (255, 0, 0) <br>
    green = (0, 255, 0) <br>
    blue = (0, 0, 255) <br>
    white = (255, 255, 255) <br>
    black = (0, 0, 0) <br>
<h3>Untuk Mendefinisikan n_classes sebagai 5 kelas</h3>
n_classes = 5
<h3>Fungsi RGB encode mask  untuk mengubah masker(mask) yang telah di-encode secara integer (biasanya hasil dari proses segmentasi) kembali menjadi gambar dengan nilai RGB</h3>
# Function to encode the mask to RGB
def rgb_encode_mask(mask):<br>
    # Initialize rgb image with 3 channels (for RGB)<br>
    rgb_encode_image = np.zeros((mask.shape[0], mask.shape[1], 3))<br>
<br>
    # Iterate over MaskColorMap<br>
    for j, cls in enumerate(MaskColorMap):<br>
        # Convert single integer channel to RGB channels<br>
        rgb_encode_image[mask == j] = np.array(cls.value) / 255.0<br>
    <br>
    return rgb_encode_image<br>
    
<h3> Fungsi untuk mengubah nama files, fungsi ini digunakan untuk mengubah nama file-file gambar di dalam sebuah folder sumber (source_folder) dan menyalinnya ke folder tujuan (destination_folder) dengan nama baru yang diurutkan. Sehingga ketika augment selesai dilakukan, seluruh hasil gambar tersebut akan disimpan di folder images-all dengan terurut</h3>
#ganti nama file-images
import os<br>
import shutil<br>
<br>
def rename_files(source_folder, destination_folder):<br>
    # Membuat folder tujuan jika belum ada<br>
    os.makedirs(destination_folder, exist_ok=True)<br>
<br>
    # Mendapatkan daftar file dalam folder sumber<br>
    file_list = os.listdir(source_folder)<br>
<br>
    # Mengurutkan file berdasarkan nama<br>
    sorted_files = sorted(file_list)<br>
<br>
    # Loop melalui setiap file<br>
    for index, filename in enumerate(sorted_files):<br>
        # Membangun path file lama dan baru<br>
        old_path = os.path.join(source_folder, filename)<br>
        new_filename = "training_" + str(index + 1) + ".png"<br>
        new_path = os.path.join(destination_folder, new_filename)<br>
<br>
        # Mengubah nama file dan memindahkannya ke folder tujuan<br>
        shutil.copy2(old_path, new_path)<br>
<br>
    print("Pengubahan nama dan pemindahan file selesai.")<br>
<br>
# Contoh penggunaan<br>
source_folder = "D:/Gio&Maul/gio/AV_groundTruth/training/images"  # Ganti dengan path folder sumber<br>
destination_folder = "D:/Gio&Maul/gio/AV_groundTruth/training/images-all"  # Ganti dengan path folder tujuan<br>
rename_files(source_folder, destination_folder)<br>

<h3>Fungsi Augmentasi Flip Vertikal</h3>
#vertikal-images<br>
import os<br>
from PIL import Image<br>
import shutil<br>
<br>
def flip_images(input_folder, output_folder):<br>
    # Membuat folder tujuan jika belum ada<br>
    os.makedirs(output_folder, exist_ok=True)<br>
<br>
    # Mengambil daftar file gambar dalam folder input<br>
    image_files = [file for file in os.listdir(input_folder) if file.endswith(('.jpg', '.bmp','.BMP', '.png','tif'))]<br>
<br>
    # Menentukan nomor awal untuk penomoran<br>
    start_number = 0<br>
<br>
    for i, file_name in enumerate(image_files):<br>
        # Membaca gambar menggunakan PIL<br>
        image_path = os.path.join(input_folder, file_name)<br>
        image = Image.open(image_path)<br>
<br>
        # Melakukan flip horizontal pada gambar<br>
        flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)<br>
<br>
        # Menentukan nomor untuk file hasil flip<br>
        flipped_number = start_number + i + 1<br>
<br>
        # Menyimpan gambar hasil flip dalam folder output dengan nomor urut<br>
        flipped_output_path = os.path.join(output_folder, f"vert_{flipped_number}.png")<br>
        flipped_image.save(flipped_output_path)<br>
<br>
        print(f"Gambar {file_name} telah di-flip vertikal ")<br>
<br>
    print("Proses flip dan penyimpanan selesai.")<br>
<br>
# Menentukan folder input<br>
input_folder = "D:/Gio&Maul/gio/AV_groundTruth/training/images"<br>
<br>
# Menentukan folder output untuk hasil flip vertikal<br>
output_folder = "D:/Gio&Maul/gio/AV_groundTruth/training/images-all"<br>
<br>
# Memanggil fungsi flip_images<br>
flip_images(input_folder, output_folder)<br>

<h3>Fungsi Flip Vertikal GroundTruth</h3>
#vertikal-av<br>
import os<br>
from PIL import Image<br>
import shutil<br>
<br>
def flip_images(input_folder, output_folder):<br>
    # Membuat folder tujuan jika belum ada<br>
    os.makedirs(output_folder, exist_ok=True)<br>
<br>
    # Mengambil daftar file gambar dalam folder input<br>
    image_files = [file for file in os.listdir(input_folder) if file.endswith(('.jpg', '.bmp','.BMP', '.png','tif'))]<br>
<br>
    # Menentukan nomor awal untuk penomoran<br>
    start_number = 0<br>
<br>
    for i, file_name in enumerate(image_files):<br>
        # Membaca gambar menggunakan PIL<br>
        image_path = os.path.join(input_folder, file_name)<br>
        image = Image.open(image_path)<br>
<br>
        # Melakukan flip horizontal pada gambar<br>
        flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)<br>
<br>
        # Menentukan nomor untuk file hasil flip<br>
        flipped_number = start_number + i + 1<br>
<br>
        # Menyimpan gambar hasil flip dalam folder output dengan nomor urut<br>
        flipped_output_path = os.path.join(output_folder, f"vert_{flipped_number}.png")<br>
        flipped_image.save(flipped_output_path)<br>
<br>
        print(f"Gambar {file_name} telah di-flip vertikal ")<br>
<br>
    print("Proses flip dan penyimpanan selesai.")<br>
<br>
# Menentukan folder input<br>
input_folder = "D:/Gio&Maul/gio/AV_groundTruth/training/av"<br>
<br>
# Menentukan folder output untuk hasil flip vertikal<br>
output_folder = "D:/Gio&Maul/gio/AV_groundTruth/training/av-all"<br>
<br>
# Memanggil fungsi flip_images<br>
flip_images(input_folder, output_folder)<br>

<h3>Fungsi Flip Horizontal Images</h3>
#horizontal flip-images <br>
import os<br>
from PIL import Image<br>
import shutil<br>
<br>
def flip_images(input_folder, output_folder):<br>
    # Membuat folder tujuan jika belum ada<br>
    os.makedirs(output_folder, exist_ok=True)<br>
<br>
    # Mengambil daftar file gambar dalam folder input<br>
    image_files = [file for file in os.listdir(input_folder) if file.endswith(('.jpg', '.BMP','bmp', '.png', 'tif'))]<br>
<br>
    # Menentukan nomor awal untuk penomoran<br>
    start_number = 0<br>
<br>
    for i, file_name in enumerate(image_files):<br>
        # Membaca gambar menggunakan PIL<br>
        image_path = os.path.join(input_folder, file_name)<br>
        image = Image.open(image_path)<br>
<br>
        # Melakukan flip horizontal pada gambar<br>
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)<br>
<br>
        # Menentukan nomor untuk file hasil flip<br>
        flipped_number = start_number + i + 1<br>
<br>
        # Menyimpan gambar hasil flip dalam folder output dengan nomor urut<br>
        flipped_output_path = os.path.join(output_folder, f"horizontal_{flipped_number}.png")<br>
        flipped_image.save(flipped_output_path)<br>
<br>
        print(f"Gambar {file_name} telah di-flip horizontal dan disimpan dengan nama file citra_{flipped_number}")<br>
<br>
    print("Proses flip dan penyimpanan selesai.")
<br>
# Menentukan folder input<br>
input_folder = "D:/Gio&Maul/gio/AV_groundTruth/training/images"<br>
<br>
# Menentukan folder output untuk hasil flip vertikal<br>
output_folder = "D:/Gio&Maul/gio/AV_groundTruth/training/images-all"<br>
<br>
# Memanggil fungsi flip_images<br>
flip_images(input_folder, output_folder)<br>

<h3>Fungsi Flip Horizontal GroundTruth</h3>
#horizontal flip-av <br>
import os<br>
from PIL import Image<br>
import shutil<br>
<br>
def flip_images(input_folder, output_folder):<br>
    # Membuat folder tujuan jika belum ada<br>
    os.makedirs(output_folder, exist_ok=True)<br>
<br>
    # Mengambil daftar file gambar dalam folder input<br>
    image_files = [file for file in os.listdir(input_folder) if file.endswith(('.jpg', '.BMP','bmp', '.png', 'tif'))]<br>
<br>
    # Menentukan nomor awal untuk penomoran<br>
    start_number = 0<br>
<br>
    for i, file_name in enumerate(image_files):<br>
        # Membaca gambar menggunakan PIL<br>
        image_path = os.path.join(input_folder, file_name)<br>
        image = Image.open(image_path)<br>
<br>
        # Melakukan flip horizontal pada gambar<br>
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)<br>
<br>
        # Menentukan nomor untuk file hasil flip<br>
        flipped_number = start_number + i + 1<br>
<br>
        # Menyimpan gambar hasil flip dalam folder output dengan nomor urut<br>
        flipped_output_path = os.path.join(output_folder, f"horizontal_{flipped_number}.png")<br>
        flipped_image.save(flipped_output_path)<br>
<br>
        print(f"Gambar {file_name} telah di-flip horizontal dan disimpan dengan nama file citra_{flipped_number}")<br>
<br>
    print("Proses flip dan penyimpanan selesai.")
<br>
# Menentukan folder input<br>
input_folder = "D:/Gio&Maul/gio/AV_groundTruth/training/av"<br>
<br>
# Menentukan folder output untuk hasil flip vertikal<br>
output_folder = "D:/Gio&Maul/gio/AV_groundTruth/training/av-all"<br>
<br>
# Memanggil fungsi flip_images<br>
flip_images(input_folder, output_folder)<br>

<h3> Preprocessing Data menggunakan Gussian-blur, Gaussian-blur digunakan untuk menghilangkan noise kecil pada image</h3>
#gaussian_blur <br>
import os <br>
import cv2 <br>
<br>
def gaussian_blur_folder(input_folder, output_folder, kernel_size=(5, 5)):<br>
    # Membuat folder output jika belum ada<br>
    if not os.path.exists(output_folder):<br>
        os.makedirs(output_folder)<br>
<br>
    # Mendapatkan daftar nama file dalam folder input<br>
    file_list = os.listdir(input_folder)<br>
<br>
    for filename in file_list:<br>
        # Mengabaikan file yang bukan gambar<br>
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):<br>
            continue<br>
<br>
        # Baca gambar dari file<br>
        img_path = os.path.join(input_folder, filename)<br>
        img = cv2.imread(img_path)<br>
<br>
        # Lakukan Gaussian blur<br>
        blurred_img = cv2.GaussianBlur(img, kernel_size, 0)<br>
<br>
        # Simpan gambar hasil ke folder output dengan nama yang sama<br>
        output_path = os.path.join(output_folder, filename)<br>
        cv2.imwrite(output_path, blurred_img)<br>
<br>
        print(f"Processed: {filename}")<br>
<br>
# Memasukkan folder input dan output yang diinginkan<br>
input_folder = "D:/Gio&Maul/gio/AV_groundTruth/training/images-all"<br>
output_folder = "D:/Gio&Maul/gio/AV_groundTruth/training/images-all-gaussian-blur"<br>
<br>
# Panggil fungsi gaussian_blur_folder dengan folder input dan output yang ditentukan<br>
gaussian_blur_folder(input_folder, output_folder)<br>

<h3>Preprocessing image menggunakan clahe</h3>
import os<br>
import cv2<br>
#clahe based on stackoferflow<br>
def apply_clahe_to_folder(input_folder, output_folder, gridsize):<br>
    # Membuat folder output jika belum ada<br>
    if not os.path.exists(output_folder):<br>
        os.makedirs(output_folder)<br>
 <br>
    # Mendapatkan daftar file dalam folder input<br>
    image_files = os.listdir(input_folder)<br>
<br>
    # Mengiterasi melalui setiap file citra dalam folder input<br>
    for image_file in image_files:<br>
        # Membuat jalur lengkap untuk citra input<br>
        input_image_path = os.path.join(input_folder, image_file)<br>
<br>
        # Membaca citra RGB<br>
        bgr = cv2.imread(input_image_path)<br>
<br>
        # Mengonversi citra ke ruang warna LAB<br>
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)<br>
<br>
        # Membagi komponen warna LAB<br>
        lab_planes = cv2.split(lab)<br>
<br>
        # Membuat objek CLAHE<br>
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(gridsize, gridsize))<br>
<br>
        # Mengaplikasikan CLAHE pada komponen L (Brightness) LAB<br>
        lab_planes_list = list(lab_planes)  # Mengonversi tuple ke list<br>
        lab_planes_list[0] = clahe.apply(lab_planes_list[0])<br>
        lab_planes = tuple(lab_planes_list)  # Mengonversi kembali list ke tuple<br>
<br>
        # Menggabungkan kembali komponen warna LAB<br>
        lab = cv2.merge(lab_planes)<br>
<br>
        # Mengonversi citra kembali ke ruang warna BGR<br>
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)<br>
<br>
        # Membuat jalur lengkap untuk citra output<br>
        output_image_path = os.path.join(output_folder, image_file)<br>
<br>
        # Menyimpan citra hasil ke folder output<br>
        cv2.imwrite(output_image_path, bgr)<br>
 <br>
    print("CLAHE telah diterapkan pada semua citra dalam folder.")<br>
<br>
# Contoh penggunaan<br>
input_folder = "D:/Gio&Maul/gio/AV_groundTruth/training/images-all-gaussian-blur"<br>
output_folder = "D:/Gio&Maul/gio/AV_groundTruth/training/images-all-clahe"<br>
gridsize = 8  # Atur sesuai kebutuhan Anda<br>
apply_clahe_to_folder(input_folder, output_folder, gridsize)<br>

<h3>Untuk mendapatkan daftar file gambar dari dua direktori image dan groundtrut kemudian mengurutkannya, dan menyimpannya dalam dua variabel, X(image) dan Y(Groundtruth)</h3>
import glob <br>
import cv2 as cv<br>
<br>
# Mendapatkan daftar file citra dalam direktori dan mengurutkannya<br>
X = sorted(glob.glob('D:/Gio&Maul/gio/AV_groundTruth/training/images-all-clahe/*.png'))<br>
Y = sorted(glob.glob('D:/Gio&Maul/gio/AV_groundTruth/training/av-all/*.png'))<br>

<h3>Program ini membantu dalam mempersiapkan dataset citra dengan mengurutkan, membaca, mengonversi format warna, mengubah ukuran (256x256), dan menyimpan citra dalam array NumPy yang siap digunakan untuk pelatihan model pembelajaran mesin atau deep learn</h3>
x_train= [] <br>
<br>
for i in range(len(X)):<br>
    x = cv.imread(X[i])<br>
    x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)<br>
    x = cv2.resize(x,(256,256))<br>
    x_train.append(x)<br>
<br>
X= np.array(x_train)<br>
len(X)<br>

<h3>Sama seperti sebelumnya namun ini untuk data Y(Groundtruth)</h3>
x_train= []<br>
<br>
for i in range(len(Y)):<br>
    y = cv.imread(Y[i])<br>
    y = cv2.cvtColor(y,cv2.COLOR_BGR2RGB)<br>
    y = cv2.resize(y,(256,256))<br>
    y_train.append(y)<br>
<br>
Y= np.array(y_train)<br>
len(Y)<br>

<h3>Program ini bertujuan untuk menampilkan beberapa gambar dari dataset X (gambar input) dan Y (gambar label) secara berdampingan menggunakan Matplotlib.</h3>
import matplotlib.pyplot as plt <br>
<br>
# Set the number of images to display<br>
display_count = 6<br>
<br>
# Generate random indices<br>
random_index = [np.random.randint(0, m) for _ in range(display_count)]<br>
<br>
# Get the sample images<br>
sample_images = [x for z in zip(list(X[random_index]), list(Y[random_index])) for x in z]<br>
<br>
# Set the number of rows and columns for display<br>
rows = 2<br>
columns = display_count if display_count % 2 == 0 else display_count + 1<br>
<br>
# Create a subplot and display images in a larger size<br>
plt.figure(figsize=(15, 7))  # Adjust the figsize to your preferred size<br>
for i, image in enumerate(sample_images):<br>
    plt.subplot(rows, columns, i + 1)<br>
    plt.imshow(image)<br>
    plt.axis('off')<br>
<br>
plt.show()<br>

<h3>Program ini berfungsi untuk mempersiapkan data untuk pelatihan model pembelajaran mesin atau deep learning dalam konteks tugas segmentasi gambar. Menggunakan fungsi one_hot_encode_masks untuk mengubah gambar-gambar label (Y) dari format RGB menjadi integer encoded labels dan menyesuaikan jumlah kelas yang baru</h3>
# convert RGB values to integer encoded labels for categorical_crossentropy<br>
Y = one_hot_encode_masks(Y, num_classes=n_classes)<br>
<br>
# split dataset into training and test groups<br>
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)<br>

<h3>Program ini digunakan untuk memvisualisasikan bagaimana masker(masks) segmentasi dari setiap kelas (misalnya, merah untuk arteri, biru untuk vena, dan seterusnya) ditampilkan dalam gambar RGB. Hal ini memungkinkan untuk memeriksa dan memverifikasi apakah label yang telah dikodekan dengan benar direpresentasikan dalam gambar visual.</h3>
# Warna untuk setiap kelas (merah, hijau, biru, putih, hitam) <br>
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0)]<br>
<br>
# Buat gambar RGB kosong <br>
mask_rgb = np.zeros((256, 256, 3), dtype=np.uint8)<br>
<br>
# Iterasi setiap kelas dan tambahkan warna ke gambar RGB<br>
for i, color in enumerate(colors):<br>
    mask_rgb[Y[0][:, :, i] == 1] = color<br>
<br>
# Tampilkan mask dalam format RGB<br>
plt.imshow(mask_rgb)<br>
plt.show()<br>
![6b711ab8-4e62-4511-9975-13fbb4927ce3]<br>(https://github.com/Giovillando/Segmentation-AV-using-Combination-ResUnet-with-ViT/assets/121701082/2efe4e14-6b58-43f2-8e51-4425ae9c7e74)

<h3> Program di atas adalah implementasi dari sebuah arsitektur jaringan yang menggabungkan beberapa komponen penting untuk tugas segmentasi gambar, dengan menggunakan elemen dari Vision Transformer (ViT) dan ResUNet++.  </h3>
import tensorflow as tf <br>
import tensorflow.keras.layers as L <br>
from tensorflow.keras.models import Model <br>
#Vision Transformer Components <br>
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, Embedding, Concatenate <br>
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint <br>
<br>
#ClassToken <br>
class ClassToken(Layer): <br>
    def __init__(self): <br>
        super().__init__() <br>
<br>
    def build(self, input_shape): <br>
        w_init = tf.random_normal_initializer() <br>
        self.w = tf.Variable( <br>
            initial_value = w_init(shape=(1, 1, input_shape[-1]),  dtype=tf.float32),<br>
            trainable = True<br>
        )<br>
<br>
    def call(self, inputs):<br>
        batch_size = tf.shape(inputs)[0]<br>
        hidden_dim = self.w.shape[-1]<br>
<br>
        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])<br>
        cls = tf.cast(cls, dtype=inputs.dtype)<br>
        return cls<br>
<br>
def mlp(x, cf): <br>
    x = Dense(cf["mlp_dim"], activation="gelu")(x)<br>
    x = Dropout(cf["dropout_rate"])(x)<br>
    x = Dense(cf["hidden_dim"])(x)<br>
    x = Dropout(cf["dropout_rate"])(x<br>
    return x<br>
<br>
def transformer_encoder(x, cf):<br>
    skip_1 = x<br>
    x = LayerNormalization()(x)<br>
    x = MultiHeadAttention(<br>
        num_heads=cf["num_heads"], key_dim=cf["hidden_dim"]<br>
    )(x, x)<br>
    x = Add()([x, skip_1])<br>
<br>
    skip_2 = x<br>
    x = LayerNormalization()(x)<br>
    x = mlp(x, cf)<br>
    x = Add()([x, skip_2])<br>
<br>
    return x<br>
<br>
#ViT Block<br>
def Vit_block(inputs, cf):<br>
    input_shape = (cf["num_patches"], cf["patch_size"]*cf["patch_size"]*cf["num_channels"])<br>
    x = Dense(cf["hidden_dim"])(inputs)<br>
    positions = tf.range(start=0, limit=cf["num_patches"], delta=1)<br>
    pos_embed = Embedding(input_dim=cf["num_patches"], output_dim=cf["hidden_dim"])(positions)<br>
    x = x + pos_embed<br>
    token = ClassToken()(x)<br>
    x = Concatenate(axis=1)([token, x])<br>
<br>
    for _ in range(cf["num_layers"]):<br>
        x = transformer_encoder(x, cf)<br>
        <br>
    x = LayerNormalization()(x)<br>
    x = x[:, 1:, :]  # Skip the class token<br>
    return x<br>
<br>
#Resunet<br>
def SE(inputs, ratio=8):<br>
    channel_axis = -1<br>
    num_filters = inputs.shape[channel_axis]<br>
    se_shape = (1, 1, num_filters)<br>
 <br>
    x = L.GlobalAveragePooling2D()(inputs)<br>
    x = L.Reshape(se_shape)(x)<br>
    x = L.Dense(num_filters // ratio, activation='relu', use_bias=False)(x)<br>
    x = L.Dense(num_filters, activation='sigmoid', use_bias=False)(x)<br>
 <br>
    x = L.Multiply()([inputs, x])<br>
    return x<br>
<br>
def stem_block(inputs, num_filters, strides=1):<br>
    ## Conv 1<br>
    x = L.Conv2D(num_filters, 3, padding="same", strides=strides)(inputs)<br>
    x = L.BatchNormalization()(x)<br>
    x = L.Activation("relu")(x)<br>
    x = L.Conv2D(num_filters, 3, padding="same")(x)<br>
 <br>
    ## Shortcut<br>
    s = L.Conv2D(num_filters, 1, padding="same", strides=strides)(inputs)<br>
    s = L.BatchNormalization()(s)<br>
 <br>
    ## Add<br>
    x = L.Add()([x, s])<br>
    x = SE(x)<br>
    return x<br>
<br>
def resnet_block(inputs, num_filter, strides=1):<br>
 <br>
    ## Conv 1<br>
    x = L.BatchNormalization()(inputs)<br>
    x = L.Activation("relu")(x)<br>
    x = L.Conv2D(num_filter, 3, padding="same", strides=strides)(x)<br>
  <br>
    ## Conv 2<br>
    x = L.BatchNormalization()(x)<br>
    x = L.Activation("relu")(x)<br>
    x = L.Conv2D(num_filter, 3, padding="same", strides=1)(x)<br>
 <br>
    ## Shortcut<br>
    s = L.Conv2D(num_filter, 1, padding="same", strides=strides)(inputs)<br>
    s = L.BatchNormalization()(s)<br>
 <br>
    ## Add<br>
    x = L.Add()([x, s])<br>
    x = SE(x)<br>
    return x<br>
<br>
def aspp_block(inputs, num_filters):<br>
    x1 = L.Conv2D(num_filters, 3, dilation_rate=6, padding="same")(inputs)<br>
    x1 = L.BatchNormalization()(x1)<br>
 <br>
    x2 = L.Conv2D(num_filters, 3, dilation_rate=12, padding="same")(inputs)<br>
    x2 = L.BatchNormalization()(x2)<br>
 <br>
    x3 = L.Conv2D(num_filters, 3, dilation_rate=18, padding="same")(inputs)<br>
    x3 = L.BatchNormalization()(x3)<br>
 <br>
    x4 = L.Conv2D(num_filters, (3, 3), padding="same")(inputs)<br>
    x4 = L.BatchNormalization()(x4)<br>
 <br>
    y = L.Add()([x1, x2, x3, x4])<br>
    y = L.Conv2D(num_filters, 1, padding="same")(y)<br>
    return y<br>
<br>
def attetion_block(x1, x2):<br>
    num_filters = x2.shape[-1]<br>
 <br>
    x1_conv = L.BatchNormalization()(x1)<br>
    x1_conv = L.Activation("relu")(x1_conv)<br>
    x1_conv = L.Conv2D(num_filters, 3, padding="same")(x1_conv)<br>
    x1_pool = L.MaxPooling2D((2, 2))(x1_conv)<br>
 <br>
    x2_conv = L.BatchNormalization()(x2)<br>
    x2_conv = L.Activation("relu")(x2_conv)<br>
    x2_conv = L.Conv2D(num_filters, 3, padding="same")(x2_conv)<br>
 <br>
    x = L.Add()([x1_pool, x2_conv])<br>
 <br>
    x = L.BatchNormalization()(x)<br>
    x = L.Activation("relu")(x)<br>
    x = L.Conv2D(num_filters, 3, padding="same")(x)<br>
 <br>
    x = L.Multiply()([x, x2])<br>
    return x<br>
<br>
def resunet_pp_with_vit(input_shape, num_classes=5):<br>
    """ Inputs """<br>
    inputs = L.Input(input_shape)<br>
<br>
    """ ViT Block """<br>
    patch_size = vit_config["patch_size"]<br>
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)<br>
    vit_inputs = L.Reshape((num_patches, patch_size * patch_size * input_shape[2]))(inputs)<br>
    vit_features = Vit_block(vit_inputs, vit_config)<br>
    vit_features = L.Reshape((input_shape[0] // patch_size, input_shape[1] // patch_size, vit_config["hidden_dim"]))(vit_features)<br>
    <br>
    """ Encoder """<br>
    c1 = stem_block(inputs, 8, strides=1)<br>
    c2 = resnet_block(c1, 16, strides=2)<br>
    c3 = resnet_block(c2, 32, strides=2)<br>
    c4 = resnet_block(c3, 64, strides=2)<br>
 <br>
    """ Bridge """<br>
    b1 = aspp_block(c4, 128)<br>
    <br>
    """ Decoder """<br>
    d1 = attetion_block(c3, b1)<br>
    d1 = L.UpSampling2D((2, 2))(d1)<br>
    d1 = L.Concatenate()([d1, c3])<br>
    d1 = resnet_block(d1, 64)<br>
 <br>
    d2 = attetion_block(c2, d1)<br>
    d2 = L.UpSampling2D((2, 2))(d2)<br>
    d2 = L.Concatenate()([d2, c2])<br>
    d2 = resnet_block(d2, 32)<br>
 <br>
    d3 = attetion_block(c1, d2)<br>
    d3 = L.UpSampling2D((2, 2))(d3)<br>
    d3 = L.Concatenate()([d3, c1])<br>
    d3 = resnet_block(d3, 16)<br>
 <br>
    """ Output"""<br>
    outputs = aspp_block(d3, 8)<br>
    outputs = L.Conv2D(5, 1, padding="same")(outputs)<br>
    outputs = L.Activation("sigmoid")(outputs)<br>
 <br>
    """ Model """<br>
    model = Model(inputs, outputs)<br>
    return model<br>

<h3>Program ini berfungsi untuk membangun, mengonfigurasi, dan melatih model jaringan saraf yang menggabungkan kekuatan dari Vision Transformer (ViT) untuk pengolahan global informasi gambar, dan ResUNet++ untuk mendapatkan detail lokal yang mendalam, khususnya untuk tugas segmentasi gambar dengan input berukuran 256x256 piksel dan 3 saluran warna (RGB).</h3>
vit_config = { <br>
    "num_patches": 256, <br>
    "patch_size": 16, <br>
    "num_channels": 5, <br>
    "hidden_dim": 256, <br>
    "num_heads": 8, <br>
    "mlp_dim": 1024, <br>
    "num_layers": 8, <br>
    "dropout_rate": 0.1 <br>
} <br>
<br>
# Bentuk input <br>
img_height = 256 <br>
img_width = 256 <br>
img_channels = 3 <br>
input_shape = (img_height, img_width, img_channels) <br>
<br>
# Inisialisasi model <br>
model = resunet_pp_with_vit(input_shape, vit_config) <br>
<br>
# Compile model <br>
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])<br>
<br>
# Inisialisasi callback<br>
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')<br>
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')<br>
<br>
# Ringkasan model <br>
model.summary() <br>

<h3>Program ini berfungsi untuk mengatur jalur penyimpanan model (model_save_path) dan menyediakan sebuah placeholder untuk logger CSV yang nantinya akan mencatat log pelatihan</h3>
model_save_path = "D:/Gio&Maul/gio/AV_groundTruth/mresuneta-4.hdf5"

<h3>Program ini adalah bagian dari proses pelatihan model jaringan saraf untuk tugas segmentasi gambar. Ini melibatkan proses pelatihan model yatu (model.fit()), penyimpanan model setelah pelatihan selesai (model.save()), dan pencetakan pesan "print("model saved:", model_save_path)" untuk memberi tahu bahwa model telah disimpan. Model yang telah disimpan dapat menggunakannya kembali untuk prediksi atau melanjutkan pelatihan di waktu yang akan datang tanpa perlu memulai dari awal.</h3>
history = model.fit(X_train, Y_train, epochs=50, batch_size=2, validation_data=(X_test, Y_test), verbose=1)<br>
model.save(model_save_path) <br>
print("model saved:", model_save_path)<br>

<h3>Program di atas berfungsi untuk memplot grafik historis dari pelatihan model. Di dalamnya terdapat pemanggilan fungsi plot_segm_history() yang berasal dari modul keras_unet.utils. Tujuannya adalah untuk memvisualisasikan metrik pelatihan seperti akurasi (accuracy) dan loss (loss) dari model Anda selama proses pelatihan.</h3>

from keras_unet.utils import plot_segm_history <br>
import matplotlib.pyplot as plt<br>
<br>
# plot_segm_history code<br>
<br>
# Plot the history<br>
plot_segm_history(<br>
    history,<br>
    metrics=['accuracy', 'val_accuracy'],<br>
    losses=['loss', 'val_loss']<br>
)<br>

<h3>Program ini berfungsi untuk melakukan visualisasi prediksi model pada dataset pengujian (test set). Pada program ini terdapat rgb_ground_truth sebagai groundtruth dan Prediction sebagai gambar prediksi. Pada program ini bertujuan untuk Memvisualisasikan bagaimana model memprediksi label-mask dari citra input dan kemudian
Membandingkan hasil prediksi(prediction) dengan ground truth(rgb_ground_truth) untuk mengevaluasi kualitas prediksi model. </h3>

import numpy as np <br>
from enum import Enum <br>
import matplotlib.pyplot as plt <br>
<br>
# Function to encode the mask to RGB<br>
def rgb_encode_mask(mask):<br>
    # Initialize rgb image with 3 channels (for RGB)<br>
    rgb_encode_image = np.zeros((mask.shape[0], mask.shape[1], 3))<br>
<br>
    # Iterate over MaskColorMap<br>
    for j, cls in enumerate(MaskColorMap):<br>
        # Convert single integer channel to RGB channels<br>
        rgb_encode_image[mask == j] = np.array(cls.value) / 255.0<br>
    <br>
    return rgb_encode_image<br>
<br>
# Function to display images<br>
def display_images(instances, rows, titles=None, figsize=(20, 20)):<br>
    fig, axes = plt.subplots(rows, len(instances) // rows, figsize=figsize)<br>
    for j, image in enumerate(instances):<br>
        plt.subplot(rows, len(instances) // rows, j + 1)<br>
        plt.imshow(image)<br>
        plt.title('' if titles is None else titles[j])<br>
        plt.axis("off")<br>
    plt.show()<br>
<br>
# Example loop to visualize predictions<br>
for _ in range(50):<br>
    # Choose random number from 0 to test set size<br>
    test_img_number = np.random.randint(0, len(X_test))<br>
<br>
    # Extract test input image<br>
    test_img = X_test[test_img_number]<br>
<br>
    # Ground truth test label converted from one-hot to integer encoding<br>
    ground_truth = np.argmax(Y_test[test_img_number], axis=-1)<br>
<br>
    # Expand first dimension as U-Net requires (m, h, w, nc) input shape<br>
    test_img_input = np.expand_dims(test_img, 0)<br>
<br>
    # Make prediction with model and remove extra dimension<br>
    prediction = np.squeeze(model.predict(test_img_input))<br>
<br>
    # Convert softmax probabilities to integer values<br>
    predicted_img = np.argmax(prediction, axis=-1)<br>
<br>
    # Convert integer encoding to rgb values<br>
    rgb_image = rgb_encode_mask(predicted_img)<br>
    rgb_ground_truth = rgb_encode_mask(ground_truth)<br>
<br>
    # Visualize model predictions<br>
    display_images(<br>
        [test_img[..., :3], rgb_ground_truth, rgb_image],<br>
        rows=1, titles=['Citra', 'Ground Truth', 'Prediction']<br>
    )<br>
![45221bb0-f015-4b72-9761-98f5a612f25e](https://github.com/Giovillando/Segmentation-AV-using-Combination-ResUnet-with-ViT/assets/121701082/052d6325-1f78-4c84-8d65-be49f82f38e3)

<h3>rogram ini berfungsi untuk menyimpan model ke dalam file HDF5 di lokasi yang ditentukan ('D:/Gio&Maul/gio/AV_groundTruth/').</h3>
model_dir = 'D:/Gio&Maul/gio/AV_groundTruth/'<br>
model_name = 'mresuneta-4.hdf5'<br>
model.save(os.path.join(model_dir, model_name))<br>

<h3>Program ini bertujuan untuk memuat kembali model yang telah disimpan sebelumnya ke dalam memori agar dapat digunakan untuk inferensi atau evaluasi tambahan. Jika model yang disimpan menggunakan fungsi kustom atau objek lain selain yang disediakan oleh TensorFlow standar, perlu menyertakan argumen custom_objects untuk memuatnya dengan benar.
</h3>
from tensorflow.keras.models import load_model <br>
model = load_model(model_dir + model_name, custom_objects={'iou_coef':iou_coef})<br>

<h3>Program ini bertujuan untuk memuat kembali berat (weights) dari model yang telah disimpan sebelumnya ke dalam model yang sudah diinisialisasi. Setelah memuat berat model, program ini melakukan prediksi pada data uji (X_test). Hasil prediksi kemudian diolah dengan argmax untuk menghasilkan nilai kelas yang diprediksi untuk setiap piksel, yang umumnya digunakan dalam masalah segmentasi gambar.</h3>
model.load_weights(model_dir+model_name)<br>
<br>
pred=model.predict(X_test)<br>
pred=np.argmax(pred,axis=3)<br>
pred.shape<br>
<br>
<h3>Program ini digunakan untuk mengevaluasi performa model segmentasi gambar dengan menggunakan metrik evaluasi seperti confusion matrix dan classification report. Matriks kebingungan (confusion_matrix) digunakan untuk menunjukkan jumlah prediksi yang benar dan salah untuk setiap kelas. Classification_report memberikan ringkasan berbagai metrik evaluasi seperti presisi, recall, f1-score, dan support untuk setiap kelas.</h3>
from sklearn.metrics import confusion_matrix, classification_report <br>
Y_test1=np.argmax(Y_test,axis=3) <br>
pred=pred.flatten() <br>
Y_test1=Y_test1.flatten() <br>
print(confusion_matrix(Y_test1,pred)) <br>

<h3>Program ini digunakan untuk mengevaluasi seberapa baik model segmentasi gambar dalam melakukan prediksi terhadap kelas-kelas yang berbeda. Dengan melihat laporan klasifikasi ini, Anda dapat mengetahui di mana model berhasil dan di mana perlu dilakukan peningkatan. Misalnya, jika ada kelas dengan precision atau recall rendah, Anda dapat mencoba untuk meningkatkan performa model untuk kelas tersebut dengan memperbaiki data latih atau menyesuaikan arsitektur model.</h3>
print(classification_report(Y_test1,pred))
