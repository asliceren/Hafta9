import os
import cv2
import numpy as np


def sentetik_renkli_bmp_olustur(dosya_yolu: str, genislik: int = 512, yukseklik: int = 512) -> None:
    x = np.arange(genislik, dtype=np.float64)[None, :]
    y = np.arange(yukseklik, dtype=np.float64)[:, None]

    b = np.clip(127.5 + 127.5 * np.sin(2.0 * np.pi * x / 256.0), 0, 255).astype(np.uint8)
    g = np.clip(127.5 + 127.5 * np.cos(2.0 * np.pi * y / 256.0), 0, 255).astype(np.uint8)
    r = np.clip(((x / genislik) + (y / yukseklik)) * 127.5, 0, 255).astype(np.uint8)

    img_bgr = np.stack(
        [
            np.broadcast_to(b, (yukseklik, genislik)),
            np.broadcast_to(g, (yukseklik, genislik)),
            np.broadcast_to(r, (yukseklik, genislik)),
        ],
        axis=-1,
    ).copy()
    cv2.imwrite(dosya_yolu, img_bgr)


def bgr_to_hsi_uint8(img_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]

    intensity = (R + G + B) / 3.0

    min_rgb = np.minimum(np.minimum(R, G), B)
    toplam_rgb = R + G + B
    saturation = np.where(toplam_rgb > 1e-12, 1.0 - (3.0 * min_rgb / np.maximum(toplam_rgb, 1e-12)), 0.0)
    saturation = np.clip(saturation, 0.0, 1.0)

    pay = 0.5 * ((R - G) + (R - B))
    payda = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-12
    aci_rad = np.arccos(np.clip(pay / payda, -1.0, 1.0))
    hue_rad = np.where(B <= G, aci_rad, 2.0 * np.pi - aci_rad)
    hue_norm = (hue_rad / (2.0 * np.pi)) % 1.0

    H_u8 = np.clip(hue_norm * 255.0, 0, 255).astype(np.uint8)
    S_u8 = np.clip(saturation * 255.0, 0, 255).astype(np.uint8)
    I_u8 = np.clip(intensity * 255.0, 0, 255).astype(np.uint8)

    return np.stack([H_u8, S_u8, I_u8], axis=-1)


girdi_dosya = "girdi_rgb.bmp"

if not os.path.exists(girdi_dosya):
    sentetik_renkli_bmp_olustur(girdi_dosya, genislik=512, yukseklik=512)

img_bgr = cv2.imread(girdi_dosya, cv2.IMREAD_COLOR)
if img_bgr is None:
    raise FileNotFoundError(f"{girdi_dosya} acilamadi.")

img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_hsi = bgr_to_hsi_uint8(img_bgr)
img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

cv2.imwrite("cikti_hsv.bmp", img_hsv)
cv2.imwrite("cikti_hsi.bmp", img_hsi)
cv2.imwrite("cikti_yuv.bmp", img_yuv)
cv2.imwrite("cikti_lab.bmp", img_lab)

print("Donusum ve BMP kayitlari tamamlandi:")
print(" - cikti_hsv.bmp")
print(" - cikti_hsi.bmp")
print(" - cikti_yuv.bmp")
print(" - cikti_lab.bmp")
