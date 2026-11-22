from pathlib import Path
import cv2
import numpy as np


def sentetik_trafik_ornekleri_olustur(klasor: Path) -> None:
    klasor.mkdir(parents=True, exist_ok=True)

    img1 = np.full((240, 240, 3), (215, 210, 200), dtype=np.uint8)
    cv2.rectangle(img1, (0, 175), (240, 240), (95, 95, 95), -1)
    cv2.circle(img1, (120, 110), 62, (0, 0, 220), 12)
    cv2.circle(img1, (120, 110), 50, (245, 245, 245), -1)
    cv2.putText(img1, "50", (88, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 4)
    cv2.imwrite(str(klasor / "ornek_hiz_limiti.png"), img1)

    img2 = np.full((240, 240, 3), (190, 210, 225), dtype=np.uint8)
    pts = np.array([[120, 45], [50, 175], [190, 175]], dtype=np.int32)
    cv2.polylines(img2, [pts], isClosed=True, color=(0, 0, 220), thickness=10)
    cv2.fillPoly(img2, [pts], color=(240, 240, 240))
    cv2.polylines(img2, [pts], isClosed=True, color=(0, 0, 220), thickness=10)
    cv2.imwrite(str(klasor / "ornek_uyari_ucgen.png"), img2)

    img3 = np.full((240, 240, 3), (210, 200, 190), dtype=np.uint8)
    cv2.circle(img3, (120, 115), 55, (210, 80, 15), -1)
    cv2.imwrite(str(klasor / "ornek_mavi_daire.png"), img3)


def kirmizi_renk_maskesi_coklu_uzay(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    hsv_kirmizi_1 = (h <= 12) & (s >= 70) & (v >= 50)
    hsv_kirmizi_2 = (h >= 168) & (h <= 179) & (s >= 70) & (v >= 50)
    hsv_maske = hsv_kirmizi_1 | hsv_kirmizi_2

    ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycbcr[:, :, 1].astype(np.int16)
    cb = ycbcr[:, :, 2].astype(np.int16)
    ycbcr_kirmizi = (cr >= 145) & (cr - cb >= 20)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.int16)
    rgb_kirmizi = (rgb[:, :, 0] - rgb[:, :, 1] >= 45) & (rgb[:, :, 0] - rgb[:, :, 2] >= 45)

    birlesik = (hsv_maske & ycbcr_kirmizi) | (hsv_maske & rgb_kirmizi)
    maske = birlesik.astype(np.uint8) * 255
    maske = cv2.morphologyEx(maske, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return maske


def cember_geometri_ve_kirmizi_dogrula(
    maske_u8: np.ndarray,
    cx: int,
    cy: int,
    r: int,
    min_kirmizi_orani: float = 0.55,
    min_dolu_sektor: int = 7,
) -> tuple[bool, float]:
    h, w = maske_u8.shape
    halka_maske = np.zeros((h, w), dtype=np.uint8)
    kalinlik = max(4, int(r * 0.16))
    cv2.circle(halka_maske, (cx, cy), r, 255, kalinlik)

    halka_pikselleri = halka_maske > 0
    toplam = int(np.count_nonzero(halka_pikselleri))
    if toplam == 0:
        return False, 0.0

    kirmizi_halka = maske_u8[halka_pikselleri] > 0
    oran = float(np.count_nonzero(kirmizi_halka)) / float(toplam)

    ys, xs = np.where(halka_pikselleri)
    acilar = (np.degrees(np.arctan2(ys - cy, xs - cx)) + 360.0) % 360.0
    sektor_indeksleri = np.clip((acilar // 45.0).astype(int), 0, 7)

    dolu_sektor_sayisi = 0
    for s_idx in range(8):
        s_maske = sektor_indeksleri == s_idx
        if np.any(s_maske):
            s_oran = float(np.count_nonzero(kirmizi_halka[s_maske])) / float(np.count_nonzero(s_maske))
            if s_oran >= 0.40:
                dolu_sektor_sayisi += 1

    gecerli = (oran >= min_kirmizi_orani) and (dolu_sektor_sayisi >= min_dolu_sektor)
    return gecerli, oran


def kirmizi_cemberli_isaret_tespit_et(
    img_bgr: np.ndarray,
    min_kirmizi_orani: float = 0.55,
) -> tuple[np.ndarray, list[tuple[int, int, int, float]]]:
    kirmizi_maske = kirmizi_renk_maskesi_coklu_uzay(img_bgr)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    cr_kanali = ycbcr[:, :, 1]
    v_kanali = hsv[:, :, 2]

    sinyal = cv2.addWeighted(cr_kanali, 0.65, v_kanali, 0.35, 0)
    sinyal = cv2.bitwise_and(sinyal, sinyal, mask=kirmizi_maske)
    sinyal_blurlu = cv2.GaussianBlur(sinyal, (9, 9), 2)

    h, w = img_bgr.shape[:2]
    min_r = max(14, min(h, w) // 15)
    max_r = max(min_r + 10, min(h, w) // 2)

    cemberler = cv2.HoughCircles(
        sinyal_blurlu,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(24, min(h, w) // 4),
        param1=95,
        param2=26,
        minRadius=min_r,
        maxRadius=max_r,
    )

    cikti = img_bgr.copy()
    dogrulananlar: list[tuple[int, int, int, float]] = []

    if cemberler is not None:
        for cx, cy, r in np.round(cemberler[0, :]).astype(int):
            if cx < 0 or cy < 0 or cx >= w or cy >= h or r <= 0:
                continue
            gecerli, oran = cember_geometri_ve_kirmizi_dogrula(
                kirmizi_maske,
                cx,
                cy,
                r,
                min_kirmizi_orani=min_kirmizi_orani,
                min_dolu_sektor=7,
            )
            if gecerli:
                dogrulananlar.append((int(cx), int(cy), int(r), float(oran)))
                cv2.circle(cikti, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(cikti, (cx, cy), 3, (255, 0, 0), -1)
                cv2.putText(
                    cikti,
                    f"RedCircle {oran:.2f}",
                    (max(5, cx - r), max(20, cy - r - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 180, 0),
                    2,
                )

    return cikti, dogrulananlar


def veri_setini_isle(veri_klasoru: str, cikti_klasoru: str = "kirmizi_cember_sonuclar", limit: int = 30) -> None:
    kok = Path(veri_klasoru)
    if not kok.exists():
        print(f"'{veri_klasoru}' bulunamadi. Sentetik test gorselleri olusturuluyor...")
        sentetik_trafik_ornekleri_olustur(kok)

    uzantilar = {".png", ".jpg", ".jpeg", ".bmp"}
    gorsel_dosyalari = [p for p in kok.rglob("*") if p.suffix.lower() in uzantilar]
    gorsel_dosyalari = gorsel_dosyalari[:limit]

    cikti_kok = Path(cikti_klasoru)
    cikti_kok.mkdir(parents=True, exist_ok=True)

    tespit_edilen_gorsel = 0
    for yol in gorsel_dosyalari:
        img = cv2.imread(str(yol), cv2.IMREAD_COLOR)
        if img is None:
            continue

        if max(img.shape[:2]) < 160:
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)

        isaretli, cemberler = kirmizi_cemberli_isaret_tespit_et(img, min_kirmizi_orani=0.55)
        if len(cemberler) > 0:
            tespit_edilen_gorsel += 1
            hedef_yol = cikti_kok / f"tespit_{yol.stem}.png"
            cv2.imwrite(str(hedef_yol), isaretli)
            print(f"[OK] {yol.name} -> {len(cemberler)} kirmizi cember bulundu -> {hedef_yol}")
        else:
            print(f"[--] {yol.name} -> kirmizi cember bulunamadi")

    print(f"Tamamlandi. Toplam incelenen: {len(gorsel_dosyalari)}, tespit edilen gorsel: {tespit_edilen_gorsel}")


if __name__ == "__main__":
    VERI_SETI_KLASORU = "traffic_signs_dataset"
    veri_setini_isle(VERI_SETI_KLASORU, cikti_klasoru="kirmizi_cember_sonuclar", limit=30)
