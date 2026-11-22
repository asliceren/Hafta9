import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def sentetik_ten_goruntusu_olustur(dosya_yolu: str, boyut: int = 256) -> None:
    img = np.zeros((boyut, boyut, 3), dtype=np.uint8)

    y_ind, x_ind = np.meshgrid(np.arange(boyut), np.arange(boyut), indexing="ij")
    r = np.clip(185 + 25 * np.sin(2.0 * np.pi * x_ind / boyut), 0, 255).astype(np.uint8)
    g = np.clip(135 + 20 * np.cos(2.0 * np.pi * y_ind / boyut), 0, 255).astype(np.uint8)
    b = np.clip(105 + 15 * np.sin(2.0 * np.pi * (x_ind + y_ind) / boyut), 0, 255).astype(np.uint8)

    ten_blok = np.stack([b, g, r], axis=-1)
    img[:, :] = ten_blok

    img[0:70, 0:90] = [220, 40, 30]
    img[170:256, 150:256] = [40, 180, 50]
    img[20:90, 170:245] = [210, 210, 210]
    img[180:245, 15:85] = [25, 25, 25]

    cv2.imwrite(dosya_yolu, img)


def ten_maskesi_rgb(img_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.int32)
    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]

    kosul = (
        (R > 95)
        & (G > 40)
        & (B > 20)
        & ((np.maximum(np.maximum(R, G), B) - np.minimum(np.minimum(R, G), B)) > 15)
        & (np.abs(R - G) > 15)
        & (R > G)
        & (R > B)
    )
    return kosul.astype(np.uint8) * 255


def ten_maskesi_ycbcr(img_bgr: np.ndarray) -> np.ndarray:
    ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Cr = ycbcr[:, :, 1]
    Cb = ycbcr[:, :, 2]

    kosul = (Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127)
    return kosul.astype(np.uint8) * 255


def ten_maskesi_hsv(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    kosul = (H <= 25) & (S >= 30) & (S <= 180) & (V >= 60)
    return kosul.astype(np.uint8) * 255


def ten_disini_kirmizi_yap(img_bgr: np.ndarray, maske: np.ndarray) -> np.ndarray:
    cikti = img_bgr.copy()
    ten_degil = maske == 0
    cikti[ten_degil] = [0, 0, 255]
    return cikti


girdi_dosya = "ten_gorsel.bmp"
alternatif_dosya = "ten_gorsel.jpg"

if os.path.exists(girdi_dosya):
    img_bgr = cv2.imread(girdi_dosya, cv2.IMREAD_COLOR)
elif os.path.exists(alternatif_dosya):
    img_bgr = cv2.imread(alternatif_dosya, cv2.IMREAD_COLOR)
else:
    sentetik_ten_goruntusu_olustur(girdi_dosya, boyut=256)
    img_bgr = cv2.imread(girdi_dosya, cv2.IMREAD_COLOR)

if img_bgr is None:
    raise FileNotFoundError("Girdi goruntusu acilamadi.")

maske_rgb = ten_maskesi_rgb(img_bgr)
maske_ycbcr = ten_maskesi_ycbcr(img_bgr)
maske_hsv = ten_maskesi_hsv(img_bgr)

sonuc_rgb = ten_disini_kirmizi_yap(img_bgr, maske_rgb)
sonuc_ycbcr = ten_disini_kirmizi_yap(img_bgr, maske_ycbcr)
sonuc_hsv = ten_disini_kirmizi_yap(img_bgr, maske_hsv)

cv2.imwrite("ten_rgb.png", sonuc_rgb)
cv2.imwrite("ten_ycbcr.png", sonuc_ycbcr)
cv2.imwrite("ten_hsv.png", sonuc_hsv)

fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

paneller = [
    ("Orijinal Goruntu", cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)),
    ("RGB Ten Tespiti", cv2.cvtColor(sonuc_rgb, cv2.COLOR_BGR2RGB)),
    ("YCbCr Ten Tespiti", cv2.cvtColor(sonuc_ycbcr, cv2.COLOR_BGR2RGB)),
    ("HSV Ten Tespiti", cv2.cvtColor(sonuc_hsv, cv2.COLOR_BGR2RGB)),
]

for ax, (baslik, gorsel) in zip(axes, paneller):
    ax.imshow(gorsel)
    ax.set_title(baslik)
    ax.axis("off")

plt.tight_layout()
plt.savefig("ten_kiyas.png", dpi=200, bbox_inches="tight")
plt.show()
