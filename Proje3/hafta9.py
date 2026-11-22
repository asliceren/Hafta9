import os
import time
import cv2
import numpy as np


def sentetik_8k_bmp_olustur(dosya_yolu: str) -> None:
    genislik = 7680
    yukseklik = 4320
    x = np.arange(genislik, dtype=np.float64)[None, :]
    y = np.arange(yukseklik, dtype=np.float64)[:, None]

    b = np.clip(127.5 + 127.5 * np.sin(2.0 * np.pi * x / 1920.0), 0, 255).astype(np.uint8)
    g = np.clip(127.5 + 127.5 * np.cos(2.0 * np.pi * y / 1080.0), 0, 255).astype(np.uint8)
    r = np.clip(((x / genislik) + (y / yukseklik)) * 127.5, 0, 255).astype(np.uint8)

    img_8k = np.stack(
        [
            np.broadcast_to(b, (yukseklik, genislik)),
            np.broadcast_to(g, (yukseklik, genislik)),
            np.broadcast_to(r, (yukseklik, genislik)),
        ],
        axis=-1,
    ).copy()
    cv2.imwrite(dosya_yolu, img_8k)


girdi_dosya = "8k_girdi.bmp"
cikti_dosya = "8k_bolunmus.bmp"

sentetik_8k_bmp_olustur(girdi_dosya)

img = cv2.imread(girdi_dosya, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"{girdi_dosya} acilamadi.")

if img.shape != (4320, 7680, 3):
    img = cv2.resize(img, (7680, 4320), interpolation=cv2.INTER_NEAREST)

baslangic = time.perf_counter()
duzenlenmis = img // 2
bitis = time.perf_counter()

gecen_sure_ms = (bitis - baslangic) * 1000.0
gecen_sure_sn = bitis - baslangic

cv2.imwrite(cikti_dosya, duzenlenmis)

print(f"Goruntu boyutu (Y, G, K): {img.shape}")
print(f"Duzenleme suresi: {gecen_sure_ms:.3f} ms ({gecen_sure_sn:.6f} s)")
print(f"Girdi disk boyutu: {os.path.getsize(girdi_dosya) / (1024 * 1024):.2f} MB")
print(f"Cikti disk boyutu: {os.path.getsize(cikti_dosya) / (1024 * 1024):.2f} MB")
