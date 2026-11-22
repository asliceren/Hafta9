import cv2
import numpy as np


def hareket_roi_ve_ten_maskesi(
    onceki_gri: np.ndarray,
    mevcut_bgr: np.ndarray,
    piksel_fark_esigi: int = 25,
    min_hareketli_piksel: int = 2500,
    genisletme_payi: int = 25,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    mevcut_gri = cv2.cvtColor(mevcut_bgr, cv2.COLOR_BGR2GRAY)
    mevcut_gri = cv2.GaussianBlur(mevcut_gri, (7, 7), 0)

    fark = cv2.absdiff(onceki_gri, mevcut_gri)
    _, hareket_maske = cv2.threshold(fark, piksel_fark_esigi, 255, cv2.THRESH_BINARY)
    hareket_maske = cv2.dilate(hareket_maske, np.ones((5, 5), np.uint8), iterations=2)

    hareketli_piksel_sayisi = int(np.count_nonzero(hareket_maske))
    if hareketli_piksel_sayisi < min_hareketli_piksel:
        return mevcut_gri, None

    ys, xs = np.where(hareket_maske > 0)
    if ys.size == 0 or xs.size == 0:
        return mevcut_gri, None

    h, w = mevcut_gri.shape
    x1 = max(0, int(np.min(xs)) - genisletme_payi)
    y1 = max(0, int(np.min(ys)) - genisletme_payi)
    x2 = min(w, int(np.max(xs)) + genisletme_payi)
    y2 = min(h, int(np.max(ys)) + genisletme_payi)

    return mevcut_gri, (x1, y1, x2, y2)


def roi_icinde_ten_renklendir(
    goruntu_bgr: np.ndarray,
    roi: tuple[int, int, int, int],
    arka_plan_karartma: float = 0.35,
) -> np.ndarray:
    x1, y1, x2, y2 = roi
    cikti = (goruntu_bgr.astype(np.float32) * arka_plan_karartma).clip(0, 255).astype(np.uint8)
    cikti[y1:y2, x1:x2] = goruntu_bgr[y1:y2, x1:x2]

    roi_bgr = goruntu_bgr[y1:y2, x1:x2]
    if roi_bgr.size == 0:
        return cikti

    roi_ycbcr = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    cr = roi_ycbcr[:, :, 1]
    cb = roi_ycbcr[:, :, 2]
    ycbcr_maske = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)

    hh = roi_hsv[:, :, 0]
    ss = roi_hsv[:, :, 1]
    vv = roi_hsv[:, :, 2]
    hsv_maske = (hh <= 25) & (ss >= 25) & (ss <= 185) & (vv >= 50)

    ten_maske = (ycbcr_maske & hsv_maske).astype(np.uint8) * 255
    ten_maske = cv2.morphologyEx(ten_maske, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ten_maske = cv2.morphologyEx(ten_maske, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    roi_renkli = roi_bgr.copy()
    renkli_katman = np.zeros_like(roi_bgr)
    renkli_katman[:, :] = [0, 220, 120]

    ten_bolge = ten_maske > 0
    roi_renkli[ten_bolge] = cv2.addWeighted(roi_bgr, 0.45, renkli_katman, 0.55, 0)[ten_bolge]

    cikti[y1:y2, x1:x2] = roi_renkli
    cv2.rectangle(cikti, (x1, y1), (x2, y2), (255, 200, 0), 2)
    return cikti


def kamera_hareket_ve_ten_tespiti(kamera_indeksi: int = 0) -> None:
    cap = cv2.VideoCapture(kamera_indeksi)
    if not cap.isOpened():
        raise RuntimeError("Web kamerasi acilamadi. Kamera baglantisini ve izinleri kontrol et.")

    onceki_gri = None
    son_kare = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame = cv2.flip(frame, 1)
        gri_blurlu = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)

        if onceki_gri is None:
            onceki_gri = gri_blurlu
            son_kare = frame.copy()
            cv2.imshow("Hareket ROI ve Ten Tespiti", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            continue

        onceki_gri, roi = hareket_roi_ve_ten_maskesi(
            onceki_gri=onceki_gri,
            mevcut_bgr=frame,
            piksel_fark_esigi=25,
            min_hareketli_piksel=2500,
            genisletme_payi=25,
        )

        if roi is not None:
            goruntu_cikti = roi_icinde_ten_renklendir(frame, roi, arka_plan_karartma=0.35)
            durum_metni = f"Hareket Aktif | ROI: {roi}"
        else:
            goruntu_cikti = (frame.astype(np.float32) * 0.35).clip(0, 255).astype(np.uint8)
            durum_metni = "Hareket Yetersiz - Arka Plan Modu"

        cv2.putText(
            goruntu_cikti,
            durum_metni,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            goruntu_cikti,
            "Kaydet: 's' | Cikis: 'q'",
            (15, goruntu_cikti.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 255, 200),
            2,
        )

        son_kare = goruntu_cikti
        cv2.imshow("Hareket ROI ve Ten Tespiti", goruntu_cikti)

        tus = cv2.waitKey(1) & 0xFF
        if tus == ord("s") and son_kare is not None:
            cv2.imwrite("hareket_ten_cikti.png", son_kare)
            print("Kaydedildi: hareket_ten_cikti.png")
        elif tus in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    kamera_hareket_ve_ten_tespiti(0)
