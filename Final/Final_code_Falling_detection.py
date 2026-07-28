"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   SISTEMA DE DETECCIÓN DE CAÍDAS — OAK-1 + YOLOv8-pose                       ║
║   + Identificación de persona objetivo por COLOR AZUL del torso              ║
╚══════════════════════════════════════════════════════════════════════════════╝



  ▸ IDENTIFICACIÓN POR COLOR:
    La persona "objetivo" se identifica porque usa un buzo/camisa AZUL.
    El sistema NO sigue a otras personas en la sala — solo a la del azul.
    Si entra otra persona con jeans (azul desaturado) NO la confunde
    porque el rango HSV es estricto en tono y saturación.

  ▸ ROI DEL TORSO (no bbox completo):
    Usa keypoints de hombros y caderas para extraer SOLO el torso.
    Excluye cabeza, brazos, piernas → cero falsos positivos por color
    de paredes, muebles, otros objetos.

  ▸ ROBUSTEZ A ILUMINACIÓN:
    Rango HSV con tono estricto, saturación moderada, brillo amplio.
    Cierre morfológico para rellenar pliegues y sombras.
    Funciona con luz natural, fluorescente, LED, e incandescente.

  ▸ HISTÉRESIS ANTI-OSCILACIÓN:
    Una vez identificada la persona objetivo, se mantiene aunque
    momentáneamente otra persona tenga score mayor (anti-flicker).

  ▸ MEMORIA TEMPORAL:
    Si la persona objetivo desaparece por <2.5s (oclusión, gira),
    el sistema mantiene su identidad por proximidad al último centroide.


  ▸ OVERLAY MEJORADO:
    Bbox VERDE → persona objetivo (con tu color)
    Bbox GRIS  → otras personas detectadas (no se trackean)
    Debajo de cada bbox: COLOR:24% (% de píxeles azules en torso)

ROPA RECOMENDADA:
  ✅ Buzo/sudadera AZUL 


CALIBRACIÓN — PARÁMETROS PRINCIPALES:

  CALIDA  - Detección caídas:
    SENSIBILIDAD          = 1.0     0.7=estricto · 1.0=normal · 1.5=sensible
    CAIDA_RATIO_BH        = 0.55    bh_actual/bh_baseline < esto → caído
    TIEMPO_CONFIRMAR_S    = 1.3     segundos de votos mayoría para confirmar

  COLOR   - Identificación de persona:
    COLOR_SCORE_MIN       = 0.20    20% de píxeles azules para ser objetivo
    COLOR_HISTERESIS      = 0.13    13% para mantenerse como objetivo
    COLOR_TIMEOUT_S       = 2.5     segundos sin ver al objetivo → buscar otro
    HSV_H_MIN, HSV_H_MAX  = 100, 125    Rango tono (azul rey)
    HSV_S_MIN             = 110         Saturación mínima
    HSV_V_MIN             = 50          Brillo mínimo

  Para ajustar el COLOR a tu buzo específico:
    1. Ponete el buzo y arrancá el sistema
    2. Mirá el % COLOR que aparece debajo de tu bbox
    3. Si es muy bajo (<25%), bajá HSV_S_MIN o ampliá rango H
    4. Si otras personas/objetos tienen >10%, subí HSV_S_MIN

ARRANQUE:
  sudo pigpiod
  source ~/depthai-env/bin/activate
  python3 seguimientoxy.py
"""

import os
import sys
import threading
import time
import queue
import subprocess
from collections import deque

import numpy as np

os.environ.setdefault("DISPLAY", "")

import cv2
import depthai as dai
import pigpio
from depthai_nodes.node import ParsingNeuralNetwork
from flask import Flask, Response

cv2.setNumThreads(2)

try:
    from ticlib import TicUSB
    _TICLIB_OK = True
except ImportError:
    _TICLIB_OK = False


# ═══════════════════════════════════════════════════════════════════════════════
#  ★ CALIBRACIÓN — CAÍDAS ★
# ═══════════════════════════════════════════════════════════════════════════════

SENSIBILIDAD       = 1.0   # 0.7=estricto · 1.0=normal · 1.5=sensible
CAIDA_RATIO_BH     = 0.55  # bh_actual/bh_baseline < esto → caído
TIEMPO_CONFIRMAR_S = 1.3   # segundos de votos mayoría para confirmar


# ═══════════════════════════════════════════════════════════════════════════════
#  ★ CALIBRACIÓN — COLOR (AZUL REY) ★
# ═══════════════════════════════════════════════════════════════════════════════

# Umbrales de identificación
COLOR_SCORE_MIN   = 0.12   # % mínimo de píxeles azules para SER objetivo
COLOR_HISTERESIS  = 0.08   # % mínimo para MANTENERSE como objetivo (anti-oscilación)
COLOR_TIMEOUT_S   = 2.5    # segundos sin ver al objetivo → liberar y buscar otro

# Rangos HSV del azul (OpenCV: H: 0-179, S: 0-255, V: 0-255)
HSV_H_MIN = 90             # era 100 — cubre azul cielo, cobalto, eléctrico
HSV_H_MAX = 135            # era 125 — cubre azul rey y azul oscuro
HSV_S_MIN = 60             # era 110 — el principal culpable del 0%
HSV_S_MAX = 255
HSV_V_MIN = 40             # era 50
HSV_V_MAX = 255

# Tamaño mínimo del ROI del torso (en píxeles) para considerar válido
TORSO_MIN_PIXELS = 80      # era 200 — más permisivo para personas lejos

# Kernel para cierre morfológico (rellena pliegues de tela)
_COLOR_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GENERAL
# ═══════════════════════════════════════════════════════════════════════════════
TIEMPO_ALERTA   = 10
JPEG_CALIDAD    = 72
FLASK_PORT      = 5000
FLASK_HOST      = "0.0.0.0"
FRAME_MAX_AGE_S = 0.18

CAM_W, CAM_H    = 416, 234
IA_W,  IA_H     = 512, 288
CONF_MIN        = 0.30

# Motor X
ZONA_MUERTA_X   = 0.07
PASO_MAX_X      = 55
PASO_MIN_X      = 6
LIMITE_IZQ      = -2000
LIMITE_DER      = 2000
CMD_INTERVALO_X = 0.05
SENTIDO_X       = -1
MAX_DELTA_X     = 35

# Servo Y
SERVO_PIN       = 18
SERVO_MIN       = 1100
SERVO_MAX       = 1780
SERVO_INICIO    = 1100
ZONA_MUERTA_Y   = 0.055
PASO_MAX_Y_US   = 38
PASO_MIN_Y_US   = 4
CMD_INTERVALO_Y = 0.033
SENTIDO_Y       = -1
MAX_DELTA_Y     = 22

# EMA seguimiento
EMA_ALPHA       = 0.45
PRED_FACTOR     = 0.08
FRAC_TORSO      = 0.40
SKIP_UMBRAL_BBOX = 0.02

DEBUG_OVERLAY   = True

# Cache overlay
METRICS_REFRESH_S = 0.20
HUD_REFRESH_S     = 0.10

# CPU afinidad (4 cores RPi 3B+)
CPU_AFFINITY = {
    "frames":   {1},
    "poses":    {2},
    "encoder":  {1},
    "pipeline": {3},
    "watchdog": {3},
}


def _set_affinity(key):
    try:
        os.sched_setaffinity(0, CPU_AFFINITY[key])
    except (AttributeError, OSError):
        pass


# Índices keypoints COCO
KP_NARIZ       = 0
KP_HOMBRO_IZQ  = 5
KP_HOMBRO_DER  = 6
KP_CADERA_IZQ  = 11
KP_CADERA_DER  = 12
KP_RODILLA_IZQ = 13
KP_RODILLA_DER = 14
KP_TOBILLO_IZQ = 15
KP_TOBILLO_DER = 16
KP_CONF        = 0.15

# Parámetros internos del algoritmo de caída
BASELINE_BUF_SIZE    = 60
BASELINE_TOP_N       = 30
BASELINE_MIN_SAMPLES = 20
GRACIA_INICIAL_S     = 2.5
VOTE_BUFFER_SIZE     = 20
VOTE_THRESHOLD       = 14
ANGULO_AGACHADO      = 50.0
ANGULO_CAIDO         = 35.0
VEL_DESCENSO_RATIO   = 0.25
TIEMPO_INMOVIL_MIN_S = 0.35
UMBRAL_INMOVIL       = 0.04
TRACKING_MAX_DIST    = 0.35
TRACKING_TIMEOUT_S   = 3.0


# ═══════════════════════════════════════════════════════════════════════════════
#  IDENTIFICACIÓN POR COLOR — FUNCIONES CORE
# ═══════════════════════════════════════════════════════════════════════════════
def _extraer_roi_torso(frame, kps, det, h, w):
    """
    Extrae el ROI del torso con 3 estrategias en cascada.
    Devuelve (roi_bgr, area_px) o (None, 0).

    Estrategia 1 — hombros + caderas (preciso, puede fallar)
    Estrategia 2 — solo hombros     (parcial, más robusto)
    Estrategia 3 — fallback bbox    (siempre funciona)
    """
    # ── ESTRATEGIA 1: hombros + caderas ──────────────────────────────
    if kps.shape[0] >= 13:
        xs_h, ys_h, xs_c, ys_c = [], [], [], []
        for idx in (KP_HOMBRO_IZQ, KP_HOMBRO_DER):
            if kps[idx, 2] >= KP_CONF:
                xs_h.append(kps[idx, 0] * w)
                ys_h.append(kps[idx, 1] * h)
        for idx in (KP_CADERA_IZQ, KP_CADERA_DER):
            if kps[idx, 2] >= KP_CONF:
                xs_c.append(kps[idx, 0] * w)
                ys_c.append(kps[idx, 1] * h)

        if xs_h and xs_c and ys_h and ys_c:
            all_x = xs_h + xs_c
            x_min = int(max(0, min(all_x) - 8))
            x_max = int(min(w, max(all_x) + 8))
            y_min = int(max(0, min(ys_h)))
            y_max = int(min(h, max(ys_c)))
            if (x_max - x_min) >= 8 and (y_max - y_min) >= 12:
                area = (x_max - x_min) * (y_max - y_min)
                if area >= TORSO_MIN_PIXELS:
                    return frame[y_min:y_max, x_min:x_max], area

        # ── ESTRATEGIA 2: solo hombros ────────────────────────────────
        if len(xs_h) >= 1 and len(ys_h) >= 1:
            x_min = int(max(0, min(xs_h) - 15))
            x_max = int(min(w, max(xs_h) + 15))
            y_min = int(max(0, min(ys_h)))
            altura_estimada = max(40, int((max(ys_h) - min(ys_h)) * 2 + 40))
            y_max = int(min(h, y_min + altura_estimada))
            if (x_max - x_min) >= 8 and (y_max - y_min) >= 12:
                area = (x_max - x_min) * (y_max - y_min)
                if area >= TORSO_MIN_PIXELS:
                    return frame[y_min:y_max, x_min:x_max], area

    # ── ESTRATEGIA 3: fallback bbox (siempre disponible) ─────────────
    bx1 = int(det.xmin * w)
    bx2 = int(det.xmax * w)
    by1 = int(det.ymin * h)
    by2 = int(det.ymax * h)
    bh  = by2 - by1
    bw  = bx2 - bx1

    # Recorte: 15%–65% vertical del bbox (excluye cabeza y piernas)
    # 10% horizontal a cada lado (excluye bordes)
    t_y1 = by1 + int(bh * 0.15)
    t_y2 = by1 + int(bh * 0.65)
    t_x1 = max(0, bx1 + int(bw * 0.10))
    t_x2 = min(w, bx2 - int(bw * 0.10))

    if t_x2 > t_x1 and t_y2 > t_y1:
        area = (t_x2 - t_x1) * (t_y2 - t_y1)
        if area >= TORSO_MIN_PIXELS:
            return frame[t_y1:t_y2, t_x1:t_x2], area

    return None, 0


def _color_score(roi_bgr):
    """
    Calcula el % de píxeles que matchean el rango HSV del azul.
    Aplica cierre morfológico para robustez ante pliegues/sombras.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([HSV_H_MIN, HSV_S_MIN, HSV_V_MIN], dtype=np.uint8)
    upper = np.array([HSV_H_MAX, HSV_S_MAX, HSV_V_MAX], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Cierre morfológico: rellena pequeños agujeros (pliegues, sombras)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _COLOR_KERNEL)

    total = mask.shape[0] * mask.shape[1]
    if total == 0:
        return 0.0
    azules = int(cv2.countNonZero(mask))
    return azules / total


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASE VIGILADA — PERSONA ÚNICA OBJETIVO
# ═══════════════════════════════════════════════════════════════════════════════
class Vigilada:
    """Estado de detección de caídas para la persona objetivo (la del azul)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset_completo()

    def reset_completo(self):
        ahora = time.monotonic()
        self._bh_buffer    = deque(maxlen=BASELINE_BUF_SIZE)
        self._bh_baseline  = 0.0
        self._n_muestras   = 0
        self._baseline_ok  = False
        self._cx_prev      = 0.5
        self._cy_prev      = 0.5
        self._ts_vista     = ahora
        self._ts_aparecio  = ahora
        self._votos        = deque(maxlen=VOTE_BUFFER_SIZE)
        self._cy_history   = deque(maxlen=15)
        self._ts_inmovil   = ahora
        self._bh_history   = deque(maxlen=20)
        self._ts_history   = deque(maxlen=20)
        self.estado        = "DE PIE"
        self.ts_confirmada = None
        self._overlay_str  = ""
        self._overlay_ts   = 0.0
        self._angulo       = 90.0
        self._n_votos      = 0
        self._evidencias   = ""
        self._ratio_bh     = 1.0
        # Color
        self._color_score  = 0.0
        self._ts_color_ok  = ahora

    def _actualizar_baseline(self, bh):
        self._bh_buffer.append(bh)
        self._n_muestras += 1
        if self._n_muestras >= BASELINE_MIN_SAMPLES:
            arr = sorted(self._bh_buffer, reverse=True)
            top = arr[:BASELINE_TOP_N]
            self._bh_baseline = float(np.median(top))
            self._baseline_ok = True

    @staticmethod
    def _angulo_columna(kps):
        if kps.shape[0] < 13:
            return 90.0, False
        ch = (kps[KP_HOMBRO_IZQ, 2] + kps[KP_HOMBRO_DER, 2]) * 0.5
        cc = (kps[KP_CADERA_IZQ, 2] + kps[KP_CADERA_DER, 2]) * 0.5
        if ch < KP_CONF or cc < KP_CONF:
            return 90.0, False
        hx = (kps[KP_HOMBRO_IZQ, 0] + kps[KP_HOMBRO_DER, 0]) * 0.5
        hy = (kps[KP_HOMBRO_IZQ, 1] + kps[KP_HOMBRO_DER, 1]) * 0.5
        cx = (kps[KP_CADERA_IZQ, 0] + kps[KP_CADERA_DER, 0]) * 0.5
        cy = (kps[KP_CADERA_IZQ, 1] + kps[KP_CADERA_DER, 1]) * 0.5
        dx = cx - hx; dy = cy - hy
        lon = np.sqrt(dx*dx + dy*dy)
        if lon < 1e-4:
            return 90.0, False
        cos_a = np.clip(abs(dy) / lon, 0.0, 1.0)
        angulo = float(np.degrees(np.arccos(cos_a)))
        return angulo, True

    def _descenso_brusco(self, bh, ahora):
        self._bh_history.append(bh)
        self._ts_history.append(ahora)
        if len(self._bh_history) < 4:
            return False
        for i in range(len(self._ts_history) - 2, -1, -1):
            if ahora - self._ts_history[i] >= 0.5:
                bh_pasado = self._bh_history[i]
                if bh_pasado > 0.01:
                    return (bh_pasado - bh) / bh_pasado > VEL_DESCENSO_RATIO
                break
        return False

    def _esta_inmovil(self, cy, ahora):
        self._cy_history.append(cy)
        if len(self._cy_history) < 4:
            return False
        recientes = list(self._cy_history)[-6:]
        rango = max(recientes) - min(recientes)
        inmovil = rango < UMBRAL_INMOVIL
        if not inmovil:
            self._ts_inmovil = ahora
        return inmovil

    def actualizar(self, det, kps, color_score, ahora):
        bh = float(det.ymax - det.ymin)
        bw = float(det.xmax - det.xmin)
        cy = float((det.ymin + det.ymax) * 0.5)

        self._ts_vista = ahora
        self._color_score = color_score

        en_gracia = (ahora - self._ts_aparecio) < GRACIA_INICIAL_S

        if self.estado == "DE PIE" or not self._baseline_ok:
            self._actualizar_baseline(bh)

        ratio_bh = bh / self._bh_baseline if self._bh_baseline > 0.01 else 1.0
        self._ratio_bh = ratio_bh

        angulo, angulo_ok = self._angulo_columna(kps)
        self._angulo = angulo

        inmovil = self._esta_inmovil(cy, ahora)
        tiempo_inmovil = ahora - self._ts_inmovil

        brusco = self._descenso_brusco(bh, ahora)

        voto = 0.0
        if not en_gracia and self._baseline_ok:
            evidencias = []

            if ratio_bh < CAIDA_RATIO_BH * SENSIBILIDAD:
                voto += 0.5
                evidencias.append("BH")
            elif ratio_bh < CAIDA_RATIO_BH * SENSIBILIDAD * 1.15:
                voto += 0.25

            if angulo_ok:
                if angulo < ANGULO_CAIDO:
                    voto += 0.3
                    evidencias.append("ANG")
                elif angulo < ANGULO_AGACHADO:
                    voto = max(0.0, voto - 0.15)

            if bh > 0.01 and (bw / bh) > 1.25:
                voto += 0.15
                evidencias.append("GEO")

            if brusco:
                voto += 0.2
                evidencias.append("VEL")

            if tiempo_inmovil >= TIEMPO_INMOVIL_MIN_S:
                voto = min(1.0, voto * 1.3)
                evidencias.append("INM")

            voto = float(np.clip(voto, 0.0, 1.0))
            self._evidencias = "|".join(evidencias) if evidencias else "OK"
        else:
            self._evidencias = "APRENDIENDO" if not self._baseline_ok else "GRACIA"
            voto = 0.0

        self._votos.append(voto)

        n_votos = sum(1 for v in self._votos if v >= 0.5)
        n_total = len(self._votos)
        self._n_votos = n_votos

        tiempo_en_votos = (n_votos / max(1, n_total)) * TIEMPO_CONFIRMAR_S

        if self.estado == "DE PIE":
            if n_total >= 8 and n_votos >= int(VOTE_THRESHOLD * 0.6):
                self.estado = "SOSPECHA"
        elif self.estado == "SOSPECHA":
            if n_votos >= VOTE_THRESHOLD and tiempo_en_votos >= TIEMPO_CONFIRMAR_S * 0.7:
                self.estado = "CAIDA"
                self.ts_confirmada = ahora
            elif n_votos < int(VOTE_THRESHOLD * 0.3):
                self.estado = "DE PIE"
        elif self.estado == "CAIDA":
            if n_votos < int(VOTE_THRESHOLD * 0.3) and not inmovil:
                self.estado = "DE PIE"
                self.ts_confirmada = None
                self._bh_buffer.clear()
                self._n_muestras = 0
                self._baseline_ok = False

        segs_caido = 0
        hay_alerta = False
        if self.estado == "CAIDA" and self.ts_confirmada:
            segs_caido = int(ahora - self.ts_confirmada)
            hay_alerta = segs_caido >= TIEMPO_ALERTA

        return self.estado, segs_caido, hay_alerta

    def hay_timeout(self, ahora):
        return (ahora - self._ts_vista) > TRACKING_TIMEOUT_S

    def texto_estado(self, segs, alerta):
        if alerta:
            return f"!ALERTA {segs}s EN PISO!", (0, 0, 255)
        if self.estado == "CAIDA":
            return f"CAIDA ({segs}s)", (0, 80, 255)
        if self.estado == "SOSPECHA":
            return f"SOSPECHA V:{self._n_votos}/{VOTE_BUFFER_SIZE}", (0, 165, 255)
        if not self._baseline_ok:
            return "APRENDIENDO", (180, 180, 180)
        return "OBJETIVO", (0, 220, 0)

    def get_overlay_str(self, det, ahora):
        if ahora - self._overlay_ts > METRICS_REFRESH_S:
            bh = det.ymax - det.ymin
            self._overlay_str = (
                f"BH:{bh:.2f}/{self._bh_baseline:.2f}={self._ratio_bh:.2f} "
                f"ANG:{self._angulo:.0f}g "
                f"V:{self._n_votos}/{VOTE_BUFFER_SIZE} "
                f"COL:{self._color_score*100:.0f}% "
                f"[{self._evidencias}]"
            )
            self._overlay_ts = ahora
        return self._overlay_str


_vigilada = Vigilada()
_id_objetivo_actual = None   # índice (en lista de dets) de la persona objetivo


# ═══════════════════════════════════════════════════════════════════════════════
#  LÍNEAS DE REFERENCIA PRE-RENDERIZADAS
# ═══════════════════════════════════════════════════════════════════════════════
_CX_F  = CAM_W // 2
_CY_F  = CAM_H // 2
_ZX_PX = int(CAM_W * ZONA_MUERTA_X)
_ZY_PX = int(CAM_H * ZONA_MUERTA_Y)

def _crear_mascara_lineas():
    m = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    g = (75, 75, 75); d = (38, 38, 38)
    cv2.line(m, (_CX_F, 0),          (_CX_F, CAM_H),        g, 1)
    cv2.line(m, (0, _CY_F),          (CAM_W, _CY_F),         g, 1)
    cv2.line(m, (_CX_F-_ZX_PX, 0),   (_CX_F-_ZX_PX, CAM_H), d, 1)
    cv2.line(m, (_CX_F+_ZX_PX, 0),   (_CX_F+_ZX_PX, CAM_H), d, 1)
    cv2.line(m, (0, _CY_F-_ZY_PX),   (CAM_W, _CY_F-_ZY_PX), d, 1)
    cv2.line(m, (0, _CY_F+_ZY_PX),   (CAM_W, _CY_F+_ZY_PX), d, 1)
    return m

_MASCARA_LINEAS = _crear_mascara_lineas()
_ENCODE_PARAMS  = [cv2.IMWRITE_JPEG_QUALITY, JPEG_CALIDAD]


# ═══════════════════════════════════════════════════════════════════════════════
#  DOBLE BUFFER
# ═══════════════════════════════════════════════════════════════════════════════
_db_bufs = [None, None]
_db_ts   = [0.0,  0.0]
_db_idx  = 0
_db_lock = threading.Lock()


def _db_escribir(frame):
    global _db_idx
    with _db_lock:
        slot = 1 - _db_idx
        _db_bufs[slot] = frame
        _db_ts[slot]   = time.monotonic()
        _db_idx        = slot


def _db_leer():
    with _db_lock:
        idx = _db_idx
        return _db_bufs[idx], _db_ts[idx]


# ═══════════════════════════════════════════════════════════════════════════════
#  ESTADO GLOBAL
# ═══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)

_output_frame   = None
_frame_lock     = threading.Lock()
_frame_event    = threading.Event()

_pos_x          = 0
_servo_pw       = SERVO_INICIO
_ts_cmd_x       = 0.0
_ts_cmd_y       = 0.0
_pi             = None
_pi_lock        = threading.Lock()
_tic_device     = None

_ema_cx = 0.5; _ema_cy = 0.5; _ema_iniciado = False
_vel_cx = 0.0; _vel_cy = 0.0; _ts_vel = 0.0
_prev_cx = 0.5; _prev_cy = 0.5

_prev_cx_obj = 0.5
_prev_cy_obj = 0.5

_fps_inf_cnt = 0; _fps_inf_ts = 0.0; _fps_inf = 0
_fps_str_cnt = 0; _fps_str_ts = 0.0; _fps_str = 0

_encode_queue   = queue.Queue(maxsize=1)

_hud_cache_text = ""
_hud_cache_ts   = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════════
def _clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


def _extraer_kps_np(det):
    if not hasattr(det, "keypoints"):
        return np.empty((0, 3), dtype=np.float32)
    kps_list = det.keypoints
    n = len(kps_list)
    if n == 0:
        return np.empty((0, 3), dtype=np.float32)
    arr = np.empty((n, 3), dtype=np.float32)
    for i, kp in enumerate(kps_list):
        arr[i, 0] = kp.x
        arr[i, 1] = kp.y
        arr[i, 2] = kp.confidence if hasattr(kp, "confidence") else 1.0
    return arr


def _seleccionar_objetivo(dets_validas, kps_list, color_scores, frame_h, frame_w):
    """
    Selecciona la persona objetivo basándose en el color score.
    Lógica:
      1. Si hay una persona con color_score >= COLOR_SCORE_MIN → es objetivo
      2. Si el objetivo previo aún tiene color_score >= COLOR_HISTERESIS → mantenerlo
      3. Si nadie supera el umbral → no hay objetivo (motores quietos)
    Devuelve índice en dets_validas o None.
    """
    global _id_objetivo_actual

    if not dets_validas:
        _id_objetivo_actual = None
        return None

    # Encontrar la persona con mayor color score
    max_idx = -1
    max_score = -1.0
    for i, score in enumerate(color_scores):
        if score > max_score:
            max_score = score
            max_idx = i

    # Caso 1: el objetivo previo sigue siendo el mejor → mantener
    if _id_objetivo_actual is not None and _id_objetivo_actual < len(color_scores):
        score_actual = color_scores[_id_objetivo_actual]
        # Si el objetivo previo aún supera la histéresis Y nadie lo supera por mucho
        if score_actual >= COLOR_HISTERESIS:
            # Comparar con el mejor: si el mejor no supera por margen amplio, mantener actual
            if max_idx == _id_objetivo_actual or (max_score - score_actual) < 0.05:
                return _id_objetivo_actual

    # Caso 2: nuevo objetivo si supera el umbral mínimo
    if max_score >= COLOR_SCORE_MIN:
        _id_objetivo_actual = max_idx
        return max_idx

    # Caso 3: nadie califica → sin objetivo
    _id_objetivo_actual = None
    return None


def _punto_seguimiento(det):
    cx = (det.xmin + det.xmax) * 0.5
    cy = det.ymin + (det.ymax - det.ymin) * FRAC_TORSO
    return cx, cy


# ═══════════════════════════════════════════════════════════════════════════════
#  EMA + PREDICCIÓN
# ═══════════════════════════════════════════════════════════════════════════════
def _actualizar_posicion(cx_raw, cy_raw):
    global _ema_cx, _ema_cy, _ema_iniciado
    global _vel_cx, _vel_cy, _ts_vel, _prev_cx, _prev_cy

    if not _ema_iniciado:
        _ema_cx = cx_raw; _ema_cy = cy_raw
        _prev_cx = cx_raw; _prev_cy = cy_raw
        _ts_vel = time.monotonic(); _ema_iniciado = True
        return cx_raw, cy_raw

    _ema_cx = EMA_ALPHA * cx_raw + (1 - EMA_ALPHA) * _ema_cx
    _ema_cy = EMA_ALPHA * cy_raw + (1 - EMA_ALPHA) * _ema_cy

    ahora = time.monotonic()
    dt    = ahora - _ts_vel
    if dt > 0.01:
        a        = 0.4
        _vel_cx  = a * (cx_raw - _prev_cx) / dt + (1 - a) * _vel_cx
        _vel_cy  = a * (cy_raw - _prev_cy) / dt + (1 - a) * _vel_cy
        _prev_cx = cx_raw; _prev_cy = cy_raw; _ts_vel = ahora

    return (
        _clamp(_ema_cx + _vel_cx * PRED_FACTOR, 0.0, 1.0),
        _clamp(_ema_cy + _vel_cy * PRED_FACTOR, 0.0, 1.0),
    )


def _resetear_ema():
    global _ema_iniciado, _vel_cx, _vel_cy
    _ema_iniciado = False; _vel_cx = 0.0; _vel_cy = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  MOTOR X — Tic
# ═══════════════════════════════════════════════════════════════════════════════
def _tic_cmd(*args):
    try:
        r = subprocess.run(["ticcmd", *args], timeout=0.8, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[TIC WARN] {r.stderr.strip()}")
    except Exception:
        pass


def motor_init():
    global _tic_device
    if _TICLIB_OK:
        try:
            _tic_device = TicUSB()
            _tic_device.energize()
            _tic_device.exit_safe_start()
            _tic_device.halt_and_set_position(0)
            print("[MOTOR X] ✓ ticlib USB (<2ms latencia)")
            return
        except Exception as e:
            print(f"[MOTOR X] ticlib falló ({e}) — usando ticcmd")
            _tic_device = None
    _tic_cmd("--resume"); _tic_cmd("--energize")
    _tic_cmd("--exit-safe-start"); _tic_cmd("--halt-and-set-position", "0")
    print("[MOTOR X] ✓ ticcmd")


def motor_mover_x(destino_abs):
    global _pos_x, _ts_cmd_x
    ahora = time.monotonic()
    if ahora - _ts_cmd_x < CMD_INTERVALO_X:
        return
    delta   = _clamp(destino_abs - _pos_x, -MAX_DELTA_X, MAX_DELTA_X)
    destino = int(_clamp(_pos_x + delta, LIMITE_IZQ, LIMITE_DER))
    if destino == _pos_x:
        return
    if _tic_device:
        try:
            _tic_device.set_target_position(destino)
            _tic_device.reset_command_timeout()
        except Exception as e:
            print(f"[TIC USB] {e}")
    else:
        _tic_cmd("--position", str(destino))
    _pos_x = destino
    _ts_cmd_x = ahora


def _correccion_x(cx):
    err = cx - 0.5
    if abs(err) < ZONA_MUERTA_X:
        return 0
    norm  = (abs(err) - ZONA_MUERTA_X) / (0.5 - ZONA_MUERTA_X)
    pasos = max(PASO_MIN_X, int(norm ** 1.6 * PASO_MAX_X))
    return pasos * SENTIDO_X if err > 0 else pasos * -SENTIDO_X


# ═══════════════════════════════════════════════════════════════════════════════
#  SERVO Y
# ═══════════════════════════════════════════════════════════════════════════════
def servo_init():
    """
    Inicializa el servo con rampa suave para no dañarlo.
    Arranca desde el centro (1440µs) y baja lento hasta SERVO_INICIO.
    Velocidad: 10µs cada 15ms  ≈ 1.5 segundos de movimiento total.
    """
    global _pi, _servo_pw
    with _pi_lock:
        _pi = pigpio.pi()
        if not _pi.connected:
            raise RuntimeError("pigpiod no conectado — ejecuta: sudo pigpiod")
        _pi.set_mode(SERVO_PIN, pigpio.OUTPUT)
        pw_arranque = (SERVO_MIN + SERVO_MAX) // 2   # 1440us — centro del rango
        _pi.set_servo_pulsewidth(SERVO_PIN, pw_arranque)
        _servo_pw = pw_arranque

    print(f"[SERVO Y]  Rampa suave {pw_arranque}us → {SERVO_INICIO}us ...")

    # Baja 10µs cada 15ms hasta llegar a SERVO_INICIO
    pw_actual = _servo_pw
    while pw_actual > SERVO_INICIO:
        pw_actual = max(SERVO_INICIO, pw_actual - 10)
        with _pi_lock:
            if _pi:
                _pi.set_servo_pulsewidth(SERVO_PIN, pw_actual)
        _servo_pw = pw_actual
        time.sleep(0.015)

    print(f"[SERVO Y]  ✓ {_servo_pw}us listo")


def servo_stop():
    global _pi
    with _pi_lock:
        if _pi:
            try:
                _pi.set_servo_pulsewidth(SERVO_PIN, 0)
                _pi.stop()
            except Exception:
                pass
            _pi = None
    print("[SERVO Y]  ✓ Detenido")


def servo_mover_y(destino_pw):
    global _servo_pw, _ts_cmd_y
    ahora = time.monotonic()
    if ahora - _ts_cmd_y < CMD_INTERVALO_Y:
        return
    delta   = _clamp(destino_pw - _servo_pw, -MAX_DELTA_Y, MAX_DELTA_Y)
    destino = int(_clamp(_servo_pw + delta, SERVO_MIN, SERVO_MAX))
    if destino == _servo_pw:
        return
    with _pi_lock:
        if _pi:
            try:
                _pi.set_servo_pulsewidth(SERVO_PIN, destino)
                _servo_pw = destino
            except Exception as e:
                print(f"[SERVO] {e}")
    _ts_cmd_y = ahora


def _correccion_y(cy):
    err = cy - 0.5
    if abs(err) < ZONA_MUERTA_Y:
        return 0
    norm     = (abs(err) - ZONA_MUERTA_Y) / (0.5 - ZONA_MUERTA_Y)
    delta_us = max(PASO_MIN_Y_US, int(norm ** 1.6 * PASO_MAX_Y_US))
    return delta_us * SENTIDO_Y if err > 0 else delta_us * -SENTIDO_Y


def watchdog_pigpio():
    _set_affinity("watchdog")
    while True:
        time.sleep(10)
        with _pi_lock:
            ok = (_pi is not None) and _pi.connected
        if not ok:
            print("[WATCHDOG] pigpiod caído — reconectando...")
            try:
                servo_init()
                print("[WATCHDOG] ✓ pigpiod reconectado")
            except Exception as e:
                print(f"[WATCHDOG] fallo: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  HILO ENCODER JPEG
# ═══════════════════════════════════════════════════════════════════════════════
def hilo_encoder(stop_event):
    global _output_frame
    _set_affinity("encoder")
    enc_params = list(_ENCODE_PARAMS)
    while not stop_event.is_set():
        try:
            canvas = _encode_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if canvas is None:
            break
        ok, buf = cv2.imencode(".jpg", canvas, enc_params)
        if ok:
            with _frame_lock:
                _output_frame = buf.tobytes()
            _frame_event.set()


# ═══════════════════════════════════════════════════════════════════════════════
#  HILO FRAMES
# ═══════════════════════════════════════════════════════════════════════════════
def hilo_frames(video_q, stop_event):
    _set_affinity("frames")
    while not stop_event.is_set():
        msg = video_q.get()
        if msg is None or stop_event.is_set():
            break
        _db_escribir(msg.getCvFrame())


# ═══════════════════════════════════════════════════════════════════════════════
#  HILO POSES — PROCESAMIENTO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
def hilo_poses(pose_q, stop_event):
    global _fps_inf_cnt, _fps_inf_ts, _fps_inf
    global _prev_cx_obj, _prev_cy_obj
    global _hud_cache_text, _hud_cache_ts
    global _id_objetivo_actual

    _set_affinity("poses")

    while not stop_event.is_set():
        pose_msg = pose_q.get()
        if pose_msg is None or stop_event.is_set():
            break

        frame, frame_ts = _db_leer()
        if frame is None:
            continue
        ahora = time.monotonic()
        if ahora - frame_ts > FRAME_MAX_AGE_S:
            continue

        h, w = frame.shape[:2]

        # FPS
        _fps_inf_cnt += 1
        if ahora - _fps_inf_ts >= 1.0:
            _fps_inf = _fps_inf_cnt; _fps_inf_cnt = 0; _fps_inf_ts = ahora

        # Detecciones válidas
        dets_validas = [d for d in pose_msg.detections if d.confidence >= CONF_MIN]

        # Sin personas detectadas
        if not dets_validas:
            if _vigilada.hay_timeout(ahora):
                _vigilada.reset_completo()
            _resetear_ema()
            _id_objetivo_actual = None
            canvas = frame.copy()
            cv2.rectangle(canvas, (0, 0), (w, 36), (16, 16, 16), -1)
            cv2.putText(canvas,
                f"INF:{_fps_inf}fps STR:{_fps_str}fps  Sin persona  X:{_pos_x}  Y:{_servo_pw}us",
                (8, 24), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 200, 0), 2)
            try:
                _encode_queue.put_nowait(canvas)
            except queue.Full:
                try: _encode_queue.get_nowait(); _encode_queue.put_nowait(canvas)
                except queue.Full: pass
            continue

        # Extraer keypoints + color score para cada persona
        kps_list = []
        color_scores = []
        for det in dets_validas:
            kps = _extraer_kps_np(det)
            kps_list.append(kps)
            roi, area = _extraer_roi_torso(frame, kps, det, h, w)
            score = _color_score(roi) if roi is not None else 0.0
            color_scores.append(score)

        # Seleccionar persona objetivo (la del azul)
        idx_objetivo = _seleccionar_objetivo(dets_validas, kps_list, color_scores, h, w)

        # Canvas (necesitamos copia para dibujar)
        canvas = frame.copy()

        # ── Dibujar TODAS las detecciones (objetivo en color, resto en gris) ────
        if DEBUG_OVERLAY:
            for i, det in enumerate(dets_validas):
                if i == idx_objetivo:
                    continue  # el objetivo se dibuja después con todos sus detalles
                x1 = int(det.xmin * w); y1 = int(det.ymin * h)
                x2 = int(det.xmax * w); y2 = int(det.ymax * h)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (90, 90, 90), 1)
                cv2.putText(canvas, f"COL:{color_scores[i]*100:.0f}%",
                            (x1, y2 + 12), cv2.FONT_HERSHEY_PLAIN, 0.6, (110, 110, 110), 1)

        # ── Procesar objetivo (si existe) ───────────────────────────────────────
        if idx_objetivo is not None:
            det = dets_validas[idx_objetivo]
            kps = kps_list[idx_objetivo]
            color_score = color_scores[idx_objetivo]

            # Tracking centroide
            _vigilada._cx_prev = (det.xmin + det.xmax) * 0.5
            _vigilada._cy_prev = (det.ymin + det.ymax) * 0.5

            # Algoritmo principal de caída
            estado, segs, hay_alerta = _vigilada.actualizar(det, kps, color_score, ahora)
            texto, color = _vigilada.texto_estado(segs, hay_alerta)

            if hay_alerta:
                cv2.rectangle(canvas, (0, 0), (w-1, h-1), (0, 0, 255), 8)

            if DEBUG_OVERLAY:
                x1 = int(det.xmin * w); y1 = int(det.ymin * h)
                x2 = int(det.xmax * w); y2 = int(det.ymax * h)
                bh_n = det.ymax - det.ymin

                # Bbox del objetivo (grueso, color de estado)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

                # Marca distintiva de "OBJETIVO"
                cv2.putText(canvas, "*", (x1 - 12, y1 + 14),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 200, 0), 2)

                # Línea columna vertebral
                if kps.shape[0] >= 13:
                    ch = (kps[KP_HOMBRO_IZQ, 2] + kps[KP_HOMBRO_DER, 2]) * 0.5
                    cc = (kps[KP_CADERA_IZQ, 2] + kps[KP_CADERA_DER, 2]) * 0.5
                    if ch >= KP_CONF and cc >= KP_CONF:
                        hx = int((kps[KP_HOMBRO_IZQ, 0] + kps[KP_HOMBRO_DER, 0]) * 0.5 * w)
                        hy = int((kps[KP_HOMBRO_IZQ, 1] + kps[KP_HOMBRO_DER, 1]) * 0.5 * h)
                        cx_ = int((kps[KP_CADERA_IZQ, 0] + kps[KP_CADERA_DER, 0]) * 0.5 * w)
                        cy_ = int((kps[KP_CADERA_IZQ, 1] + kps[KP_CADERA_DER, 1]) * 0.5 * h)
                        col_linea = (0, 255, 0) if estado in ("DE PIE", "OBJETIVO") else (0, 165, 255) if estado == "SOSPECHA" else (0, 0, 255)
                        cv2.line(canvas, (hx, hy), (cx_, cy_), col_linea, 3)
                        cv2.circle(canvas, (hx, hy), 4, (255, 255, 0), -1)
                        cv2.circle(canvas, (cx_, cy_), 4, (255, 255, 0), -1)

                # Keypoints visibles
                mask = kps[:, 2] >= KP_CONF
                for i in np.where(mask)[0]:
                    px = int(kps[i, 0] * w); py = int(kps[i, 1] * h)
                    cv2.circle(canvas, (px, py), 2, (200, 200, 0), -1)

                # Métricas (con COLOR%)
                metricas = _vigilada.get_overlay_str(det, ahora)
                cv2.putText(canvas, metricas, (x1, y2 + 14),
                            cv2.FONT_HERSHEY_PLAIN, 0.72, (140, 140, 140), 1)

                # Estado encima bbox
                cv2.putText(canvas, texto, (x1, max(y1 - 14, 12)),
                            cv2.FONT_HERSHEY_PLAIN, 1.1, color, 2)

                # Punto torso
                tx = int((det.xmin + det.xmax) * 0.5 * w)
                ty = int((det.ymin + bh_n * FRAC_TORSO) * h)
                cv2.circle(canvas, (tx, ty), 5, (0, 255, 0), -1)

                # Líneas de referencia
                cv2.addWeighted(canvas, 1.0, _MASCARA_LINEAS, 0.5, 0, canvas)

            # Control motores (solo si hay objetivo)
            cx_raw, cy_raw = _punto_seguimiento(det)
            if not (abs(cx_raw - _prev_cx_obj) < SKIP_UMBRAL_BBOX and
                    abs(cy_raw - _prev_cy_obj) < SKIP_UMBRAL_BBOX):
                _prev_cx_obj = cx_raw
                _prev_cy_obj = cy_raw
                cx_s, cy_s = _actualizar_posicion(cx_raw, cy_raw)

                if DEBUG_OVERLAY:
                    ax, ay = int(cx_s * w), int(cy_s * h)
                    cv2.circle(canvas, (ax, ay), 8, (255, 200, 0), 2)
                    cv2.drawMarker(canvas, (ax, ay), (0, 200, 255), cv2.MARKER_CROSS, 18, 2)

                corr_x = _correccion_x(cx_s)
                if corr_x:
                    motor_mover_x(_pos_x + corr_x)
                corr_y = _correccion_y(cy_s)
                if corr_y:
                    servo_mover_y(_servo_pw + corr_y)

            estado_str = texto
        else:
            # No hay objetivo: timeout
            if _vigilada.hay_timeout(ahora):
                _vigilada.reset_completo()
            _resetear_ema()
            estado_str = f"Buscando objetivo (azul) - {len(dets_validas)} personas"
            hay_alerta = False
            color = (180, 180, 180)

        # HUD
        cv2.rectangle(canvas, (0, 0), (w, 36),
                      (85, 0, 0) if (idx_objetivo is not None and _vigilada.estado == "CAIDA") else (16, 16, 16), -1)

        if "ALERTA" in estado_str:
            hud_col = (0, 0, 255)
        elif "CAIDA" in estado_str:
            hud_col = (0, 80, 255)
        elif "SOSPEC" in estado_str:
            hud_col = (0, 165, 255)
        elif "Buscando" in estado_str:
            hud_col = (180, 180, 180)
        else:
            hud_col = (0, 200, 0)

        if ahora - _hud_cache_ts > HUD_REFRESH_S:
            _hud_cache_text = (
                f"INF:{_fps_inf}fps STR:{_fps_str}fps  "
                f"{estado_str}  X:{_pos_x}  Y:{_servo_pw}us"
            )
            _hud_cache_ts = ahora

        cv2.putText(canvas, _hud_cache_text,
                    (8, 24), cv2.FONT_HERSHEY_PLAIN, 1.0, hud_col, 2)

        cv2.putText(canvas, "Objetivo=AZUL  *=tracked  COL=%azul torso",
                    (8, h - 6), cv2.FONT_HERSHEY_PLAIN, 0.6, (70, 70, 70), 1)

        # Encolar para JPEG
        try:
            _encode_queue.put_nowait(canvas)
        except queue.Full:
            try:
                _encode_queue.get_nowait()
                _encode_queue.put_nowait(canvas)
            except queue.Full:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE DEPTHAI
# ═══════════════════════════════════════════════════════════════════════════════
def _arrancar_pipeline():
    stop_event = threading.Event()

    with dai.Pipeline() as pipeline:
        cam    = pipeline.create(dai.node.Camera).build()
        nn     = pipeline.create(ParsingNeuralNetwork).build(
            cam, "luxonis/yolov8-nano-pose-estimation:coco-512x288"
        )
        parser = nn.getParser()

        video_q = cam.requestOutput(
            (CAM_W, CAM_H), type=dai.ImgFrame.Type.BGR888p
        ).createOutputQueue(maxSize=1, blocking=False)

        pose_q = parser.out.createOutputQueue(maxSize=1, blocking=False)

        pipeline.start()
        print(f"[PIPELINE] ✓ OAK-1 MyriadX — stream {CAM_W}×{CAM_H} | IA {IA_W}×{IA_H}")

        t_frames  = threading.Thread(target=hilo_frames,  args=(video_q, stop_event), daemon=True, name="t-frames")
        t_poses   = threading.Thread(target=hilo_poses,   args=(pose_q,  stop_event), daemon=True, name="t-poses")
        t_encoder = threading.Thread(target=hilo_encoder, args=(stop_event,),          daemon=True, name="t-encoder")
        t_frames.start(); t_poses.start(); t_encoder.start()

        try:
            while pipeline.isRunning():
                time.sleep(0.5)
        finally:
            stop_event.set()
            try: _encode_queue.put_nowait(None)
            except queue.Full: pass
            t_frames.join(timeout=2.0)
            t_poses.join(timeout=2.0)
            t_encoder.join(timeout=2.0)
            print("[PIPELINE] Detenido")


def camara_loop():
    _set_affinity("pipeline")
    while True:
        try:
            _arrancar_pipeline()
            print("[PIPELINE] Reintentando en 3s...")
        except Exception as e:
            print(f"[PIPELINE ERROR] {e}")
            time.sleep(2)
        time.sleep(3)


# ═══════════════════════════════════════════════════════════════════════════════
#  FLASK — MJPEG
# ═══════════════════════════════════════════════════════════════════════════════
def _generar_mjpeg():
    global _fps_str_cnt, _fps_str_ts, _fps_str
    while True:
        _frame_event.wait(timeout=1.0)
        _frame_event.clear()
        with _frame_lock:
            frame = _output_frame
        if frame is None:
            continue
        _fps_str_cnt += 1
        ahora = time.monotonic()
        if ahora - _fps_str_ts >= 1.0:
            _fps_str = _fps_str_cnt; _fps_str_cnt = 0; _fps_str_ts = ahora
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"


@app.route("/")
def index():
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Detector Caidas v17 - Objetivo Azul</title>
<style>
body{{background:#070b0f;color:#c8d8e8;font-family:monospace;
  display:flex;flex-direction:column;align-items:center;padding:20px;gap:12px;}}
h1{{color:#2196f3;font-size:1.3rem;}}
img{{max-width:680px;width:100%;border:2px solid #1c2a3a;border-radius:10px;}}
.info{{background:#0d1117;padding:10px 16px;border-radius:7px;font-size:.75rem;
  display:flex;gap:16px;flex-wrap:wrap;justify-content:center;}}
.tag{{color:#2196f3;}}
.tag2{{color:#00e676;}}
.cal{{background:#0d1117;border:1px solid #1c3a2a;padding:10px 16px;
  border-radius:7px;font-size:.72rem;color:#888;max-width:680px;}}
</style></head><body>
<h1>DETECTOR DE CAIDAS v17 - Objetivo por COLOR AZUL</h1>
<p style="color:#666;font-size:.8rem;">Solo trackea a la persona con buzo/camisa AZUL · Resto se ignora</p>
<img src="/video_feed">
<div class="info">
  <span><span class="tag">HSV_H</span> {HSV_H_MIN}-{HSV_H_MAX}</span>
  <span><span class="tag">HSV_S_MIN</span> {HSV_S_MIN}</span>
  <span><span class="tag">HSV_V_MIN</span> {HSV_V_MIN}</span>
  <span><span class="tag">SCORE_MIN</span> {COLOR_SCORE_MIN*100:.0f}%</span>
  <span><span class="tag">HISTERESIS</span> {COLOR_HISTERESIS*100:.0f}%</span>
</div>
<div class="info">
  <span><span class="tag2">SENSIBILIDAD</span> {SENSIBILIDAD}</span>
  <span><span class="tag2">CAIDA_RATIO_BH</span> {CAIDA_RATIO_BH}</span>
  <span><span class="tag2">TIEMPO_CONF</span> {TIEMPO_CONFIRMAR_S}s</span>
  <span><span class="tag2">VOTOS</span> {VOTE_THRESHOLD}/{VOTE_BUFFER_SIZE}</span>
</div>
<div class="cal">
  <b>Ropa recomendada:</b> Buzo/sudadera AZUL REY entero (saturado, no marino).<br>
  <b>Calibrar color:</b> Ajustá HSV_S_MIN si el % es bajo · HSV_H_MIN/MAX para tono exacto · COLOR_SCORE_MIN para sensibilidad de identificación.<br>
  <b>Overlay:</b> Bbox VERDE=objetivo · Bbox GRIS=otras personas (ignoradas) · COL:XX%=% de azul detectado en torso
</div>
<script>
const img=document.querySelector('img');
img.addEventListener('error',()=>setTimeout(()=>img.src='/video_feed?'+Date.now(),1200));
</script></body></html>"""


@app.route("/video_feed")
def video_feed():
    return _generar_mjpeg(), 200, {
        "Content-Type": "multipart/x-mixed-replace; boundary=frame",
        "Cache-Control": "no-cache"
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "═" * 70)
    print("  DETECTOR DE CAÍDAS — v17.0 (Objetivo por COLOR AZUL)")
    print("  v16 + identificación por buzo azul + ignora otras personas")
    print("═" * 70)
    print(f"  ★ COLOR              : Azul rey (H:{HSV_H_MIN}-{HSV_H_MAX}, S>={HSV_S_MIN}, V>={HSV_V_MIN})")
    print(f"  ★ COLOR_SCORE_MIN    : {COLOR_SCORE_MIN*100:.0f}%")
    print(f"  ★ COLOR_HISTERESIS   : {COLOR_HISTERESIS*100:.0f}%")
    print(f"  ★ COLOR_TIMEOUT_S    : {COLOR_TIMEOUT_S}s")
    print(f"  ★ SENSIBILIDAD       : {SENSIBILIDAD}")
    print(f"  ★ CAIDA_RATIO_BH     : {CAIDA_RATIO_BH}")
    print(f"  ★ TIEMPO_CONFIRMAR_S : {TIEMPO_CONFIRMAR_S}s")
    print(f"    Motor X            : {'ticlib USB' if _TICLIB_OK else 'ticcmd'}")
    print(f"    Servo Y            : {SERVO_INICIO}µs")
    print(f"    Stream             : {CAM_W}×{CAM_H}")
    print("═" * 70 + "\n")

    try:
        motor_init()
        servo_init()
    except Exception as e:
        print(f"[ERROR HARDWARE] {e}")
        sys.exit(1)

    threading.Thread(target=watchdog_pigpio, daemon=True, name="t-watchdog").start()
    threading.Thread(target=camara_loop,     daemon=True, name="t-pipeline").start()

    print("[BOOT] Esperando 4s para que el pipeline arranque...\n")
    time.sleep(4)
    print(f"  ✓ Stream: http://0.0.0.0:{FLASK_PORT}/\n")

    try:
        app.run(host=FLASK_HOST, port=FLASK_PORT,
                threaded=True, debug=False, use_reloader=False)
    finally:
        servo_stop()
        print("[EXIT] Detenido.")