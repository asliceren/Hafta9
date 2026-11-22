import os
import cv2
import numpy as np


def sentetik_renkli_bmp_olustur(dosya_yolu: str, boyut: int = 1000) -> None:
    yy, xx = np.meshgrid(np.arange(boyut, dtype=np.float64), np.arange(boyut, dtype=np.float64))
    b_kanali = np.clip(127.5 + 127.5 * np.sin(2.0 * np.pi * xx / 250.0), 0, 255).astype(np.uint8)
    g_kanali = np.clip(127.5 + 127.5 * np.cos(2.0 * np.pi * yy / 250.0), 0, 255).astype(np.uint8)
    r_kanali = np.clip((xx + yy) / (2.0 * boyut) * 255.0, 0, 255).astype(np.uint8)
    renkli_img = np.stack([b_kanali, g_kanali, r_kanali], axis=-1)
    cv2.imwrite(dosya_yolu, renkli_img)


renkli_dosya = "renkli_1000x1000.bmp"
gri_dosya = "gri_1000x1000.bmp"

if not os.path.exists(renkli_dosya):
    sentetik_renkli_bmp_olustur(renkli_dosya, boyut=1000)

renkli_img = cv2.imread(renkli_dosya, cv2.IMREAD_COLOR)
if renkli_img is None:
    raise FileNotFoundError(f"{renkli_dosya} acilamadi.")

cv2.imshow("Renkli BMP (1000x1000)", renkli_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

gri_img = cv2.cvtColor(renkli_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(gri_dosya, gri_img)

renkli_bayt = os.path.getsize(renkli_dosya)
gri_bayt = os.path.getsize(gri_dosya)

renkli_piksel_veri = renkli_img.shape[0] * renkli_img.shape[1] * renkli_img.shape[2]
gri_piksel_veri = gri_img.shape[0] * gri_img.shape[1] * 1

print(f"Renkli sekil (Y, G, K): {renkli_img.shape} | Piksel veri: {renkli_piksel_veri} bayt | Disk: {renkli_bayt} bayt ({renkli_bayt / (1024 * 1024):.2f} MB)")
print(f"Gri sekil (Y, G):        {gri_img.shape} (Kanal=1) | Piksel veri: {gri_piksel_veri} bayt | Disk: {gri_bayt} bayt ({gri_bayt / (1024 * 1024):.2f} MB)")
print(f"Diskteki boyut orani (Renkli / Gri): {renkli_bayt / gri_bayt:.2f}x")
