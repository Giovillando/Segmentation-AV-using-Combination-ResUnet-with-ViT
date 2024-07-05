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
![Screenshot 2024-07-05 222643](https://github.com/Giovillando/Segmentation-AV-using-Combination-ResUnet-with-ViT/assets/121701082/db0aea86-4cca-4ff3-8ee3-b376b9ea65d8)
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
