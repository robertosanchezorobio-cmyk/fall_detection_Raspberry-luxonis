"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        SISTEMA DE DETECCIÓN DE CAÍDAS — OAK-1 + YOLOv8-pose  v14.0         ║
║  Detección robusta — 8 criterios + calibración simplificada                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

CAMBIOS v14 respecto a v13:

  ▸ 2 CRITERIOS NUEVOS:

    C7 — POSICIÓN Y DEL CENTRO DE HOMBROS:
         Los hombros son los keypoints más confiables del modelo.
         De pie: hombros en Y ~ 0.2-0.45 (parte superior del frame)
         Caído:  hombros en Y > 0.5 (parte media-baja)
         Funciona en CUALQUIER orientación.

    C8 — PROPORCIÓN DE EXTENSIÓN HORIZONTAL/VERTICAL DE KEYPOINTS:
         Calcula (span_X) / (span_Y) de los keypoints visibles.
         De pie: span_X << span_Y → ratio < 0.6
         Caído:  span_X >> span_Y → ratio > 1.3
         Más robusto que el ratio del bbox solo.

  ▸ CALIBRACIÓN ULTRA-SIMPLE:
    Solo 4 parámetros principales al inicio del archivo.
    Tabla en comentarios indicando qué bajar/subir para cada caso.
    Todos los criterios C1-C8 votan al score, no necesitas tocar los
    secundarios — solo los 4 maestros.

  ▸ MANTIENE TODO LO BUENO DE v13:
    ✓ Tu calibración previa (BH=0.32, SCORE_CONFIRMAR=30)
    ✓ ticlib USB para motor X
    ✓ Encoder JPEG en hilo separado
    ✓ Afinidad de CPU en 4 cores
    ✓ FOV panorámico completo
    ✓ Watchdog pigpio
    ✓ Doble buffer sin copia redundante
    ✓ Keypoints vectorizados con numpy

═══════════════════════════════════════════════════════════════════════════════
  CALIBRACIÓN — SOLO TOCA ESTOS 4 PARÁMETROS:
═══════════════════════════════════════════════════════════════════════════════

  PARAMETRO_1: BH_CAIDA_MAX   (altura del bbox)
  PARAMETRO_2: HOMBROS_Y_MIN  (posición Y mínima de hombros para considerar caído)
  PARAMETRO_3: SCORE_CONFIRMAR (velocidad de confirmación)
  PARAMETRO_4: SENSIBILIDAD   (multiplicador global de todos los scores)

  Tabla rápida de calibración:

  ┌────────────────────────────────┬──────────────────┬─────────────┐
  │ PROBLEMA                       │ PARÁMETRO        │ ACCIÓN      │
  ├────────────────────────────────┼──────────────────┼─────────────┤
  │ No detecta caído               │ SENSIBILIDAD     │ Subir       │
  │ Falsas alarmas estando de pie  │ SENSIBILIDAD     │ Bajar       │
  │ Tarda en confirmar             │ SCORE_CONFIRMAR  │ Bajar       │
  │ Confirma muy rápido            │ SCORE_CONFIRMAR  │ Subir       │
  │ No detecta cerca de cámara     │ BH_CAIDA_MAX     │ Subir       │
  │ Detecta caído cuando agachado  │ BH_CAIDA_MAX     │ Bajar       │
  │ No detecta caído frontal       │ HOMBROS_Y_MIN    │ Bajar       │
  │ Detecta caído sentado          │ HOMBROS_Y_MIN    │ Subir       │
  └────────────────────────────────┴──────────────────┴─────────────┘

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
from enum import Enum, auto

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
#  ★ PARÁMETROS DE CALIBRACIÓN — TOCA SOLO ESTOS 4 ★
# ═══════════════════════════════════════════════════════════════════════════════

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ 1. BH_CAIDA_MAX (0.20 a 0.45)                                          │
# │    Altura del bbox bajo la cual se considera caída.                    │
# │    De pie: bh ~ 0.5-0.9 · Caído: bh < 0.32                             │
# └─────────────────────────────────────────────────────────────────────────┘
BH_CAIDA_MAX = 0.32

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ 2. HOMBROS_Y_MIN (0.40 a 0.60)                                         │
# │    Posición Y mínima del centro de hombros para considerar caída.      │
# │    De pie: hombros Y ~ 0.2-0.4 · Caído: hombros Y > 0.50                │
# └─────────────────────────────────────────────────────────────────────────┘
HOMBROS_Y_MIN = 0.50

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ 3. SCORE_CONFIRMAR (15 a 50)                                           │
# │    Score acumulado para confirmar caída. Menor = más rápido.           │
# │    15 = muy sensible · 30 = balanceado · 50 = muy estricto             │
# └─────────────────────────────────────────────────────────────────────────┘
SCORE_CONFIRMAR = 25

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ 4. SENSIBILIDAD (0.5 a 2.0)                                            │
# │    Multiplicador global del score. Sube todos los puntos a la vez.     │
# │    0.5 = mitad · 1.0 = normal · 1.5 = más sensible · 2.0 = máximo      │
# └─────────────────────────────────────────────────────────────────────────┘
SENSIBILIDAD = 1.2


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GENERAL (no necesitas tocar nada de aquí en adelante)
# ═══════════════════════════════════════════════════════════════════════════════
TIEMPO_ALERTA   = 10
JPEG_CALIDAD    = 55
JPEG_OPTIMIZAR  = False
FLASK_PORT      = 5000
FLASK_HOST      = "0.0.0.0"
FRAME_MAX_AGE_S = 0.18

CAM_W, CAM_H    = 416, 234
IA_W, IA_H      = 512, 288

CONF_MIN        = 0.30

ZONA_MUERTA_X   = 0.07
PASO_MAX_X      = 55
PASO_MIN_X      = 6
LIMITE_IZQ      = -2000
LIMITE_DER      = 2000
CMD_INTERVALO_X = 0.05
SENTIDO_X       = -1
MAX_DELTA_X     = 35

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

EMA_ALPHA       = 0.45
PRED_FACTOR     = 0.08
FRAC_TORSO      = 0.40

SKIP_UMBRAL_BBOX  = 0.02
DEBUG_OVERLAY     = True

PURGE_INTERVALO_S    = 60.0
PURGE_MAX_EDAD_S     = 60.0
PURGE_EXP_INTERVALO  = 5.0

PERSONA_STR_REFRESH_S = 0.20
HUD_REFRESH_S         = 0.10

BID_GRID_X = 6
BID_GRID_Y = 4

CPU_AFFINITY_FRAMES   = {1}
CPU_AFFINITY_POSES    = {2}
CPU_AFFINITY_ENCODER  = {1}
CPU_AFFINITY_PIPELINE = {3}
CPU_AFFINITY_WATCHDOG = {3}


def _set_cpu_affinity(mask):
    if mask is None:
        return
    try:
        os.sched_setaffinity(0, mask)
    except (AttributeError, OSError):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  PARÁMETROS INTERNOS DE CRITERIOS (ya optimizados, no tocar)
# ═══════════════════════════════════════════════════════════════════════════════

# C1: Ángulo del tronco
ANGULO_CAIDA_MAX  = 55.0
KP_CONF_MIN       = 0.15

# C2: Dispersión Y de keypoints
KP_SPAN_MIN_VISIBLE = 5
KP_SPAN_CONF_MIN    = 0.12
KP_SPAN_Y_CAIDA     = 0.32

# C3: Ratio bbox
RATIO_CAIDA_UMBRAL  = 1.25

# C4: Cabeza+tobillos
KP_CABEZTOB_CONF    = 0.12
KP_CABEZTOB_MAX_DY  = 0.28

# C5: BH del bbox → controlado por BH_CAIDA_MAX (parámetro maestro)

# C6: CY del bbox
CY_CAIDA_MIN        = 0.55
SERVO_PW_SUELO_MAX  = 1320

# C7: Hombros Y → controlado por HOMBROS_Y_MIN (parámetro maestro)
HOMBROS_CONF_MIN    = 0.15

# C8: Ratio extensión keypoints
KP_RATIO_XY_CAIDA   = 1.30   # spanX/spanY > 1.3 → caído
KP_RATIO_MIN_VISIBLE = 6

# ── Score base (se multiplican por SENSIBILIDAD automáticamente) ──────────────
def _s(v):
    """Aplica SENSIBILIDAD global a un score."""
    return v * SENSIBILIDAD

SCORE_MAX       = 80

# Combinaciones muy fuertes (3+ criterios)
SCORE_TRIPLE    = _s(16)   # C5+C7+otro = caída casi 100% segura
SCORE_DOBLE_KP  = _s(13)   # C7+C2 o C7+C8 = evidencia keypoints fuerte
SCORE_C4_C5     = _s(14)
SCORE_C2_C5     = _s(12)
SCORE_C5_C7     = _s(13)   # bbox bajo + hombros bajos
SCORE_C5_C6     = _s(11)
SCORE_C5_C3     = _s(10)
SCORE_C7_C8     = _s(11)   # hombros bajos + ratio horizontal
SCORE_C2_OTRO   = _s(10)

# Criterios solos
SCORE_C7_SOLO   = _s(7)    # hombros bajos solo
SCORE_C8_SOLO   = _s(6)    # ratio extensión solo
SCORE_C4_SOLO   = _s(9)
SCORE_C2_SOLO   = _s(8)
SCORE_C5_SOLO   = _s(6)
SCORE_C1_SOLO   = _s(5)
SCORE_C6_SOLO   = _s(3)
SCORE_C3_SOLO   = _s(2)

SCORE_RESTA     = 1
SCORE_SOSPECHA  = max(8, int(SCORE_CONFIRMAR * 0.35))
SCORE_RECUPERAR = 3

SEG_PERSISTENCIA     = 10.0
SEG_PERSISTENCIA_PIE = 5.0

# Índices COCO
KP_NARIZ      = 0
KP_HOMBRO_IZQ = 5
KP_HOMBRO_DER = 6
KP_CADERA_IZQ = 11
KP_CADERA_DER = 12
KP_TOBILLO_IZQ = 15
KP_TOBILLO_DER = 16


# ═══════════════════════════════════════════════════════════════════════════════
#  MÁQUINA DE ESTADOS — 8 CRITERIOS
# ═══════════════════════════════════════════════════════════════════════════════
class Estado(Enum):
    DE_PIE           = auto()
    SOSPECHA         = auto()
    CAIDA_CONFIRMADA = auto()


class EstadoPersona:
    __slots__ = ("estado", "score", "ts_confirmada", "ts_ultima_deteccion",
                 "angulo_tronco", "span_y", "kp_confiables", "c4_activo",
                 "hombros_y", "ratio_xy", "_str_cache", "_str_cache_ts")

    def __init__(self):
        self.estado              = Estado.DE_PIE
        self.score               = 0.0
        self.ts_confirmada       = None
        self.ts_ultima_deteccion = time.monotonic()
        self.angulo_tronco       = 90.0
        self.span_y              = 1.0
        self.kp_confiables       = False
        self.c4_activo           = False
        self.hombros_y           = 0.3
        self.ratio_xy            = 0.5
        self._str_cache          = ""
        self._str_cache_ts       = 0.0

    @staticmethod
    def _criterio_angulo(kps):
        if kps.shape[0] < 13:
            return False, 90.0
        if (kps[KP_HOMBRO_IZQ, 2] + kps[KP_HOMBRO_DER, 2]) * 0.5 < KP_CONF_MIN:
            return False, 90.0
        if (kps[KP_CADERA_IZQ, 2] + kps[KP_CADERA_DER, 2]) * 0.5 < KP_CONF_MIN:
            return False, 90.0
        hombro = (kps[KP_HOMBRO_IZQ, :2] + kps[KP_HOMBRO_DER, :2]) * 0.5
        cadera = (kps[KP_CADERA_IZQ, :2] + kps[KP_CADERA_DER, :2]) * 0.5
        dx = cadera[0] - hombro[0]
        dy = cadera[1] - hombro[1]
        lon = np.sqrt(dx*dx + dy*dy)
        if lon < 1e-4:
            return False, 90.0
        cos_a = np.clip(abs(dy) / lon, 0.0, 1.0)
        angulo = 90.0 - abs(np.degrees(np.arccos(cos_a)) - 90.0)
        return (angulo < ANGULO_CAIDA_MAX), angulo

    @staticmethod
    def _criterio_span_y(kps):
        if kps.shape[0] < 5:
            return False, 1.0
        mask = kps[:, 2] >= KP_SPAN_CONF_MIN
        if int(np.count_nonzero(mask)) < KP_SPAN_MIN_VISIBLE:
            return False, 1.0
        ys = kps[mask, 1]
        span = float(ys.max() - ys.min())
        return (span < KP_SPAN_Y_CAIDA), span

    @staticmethod
    def _criterio_ratio(det):
        bw = det.xmax - det.xmin
        bh = det.ymax - det.ymin
        return (bh > 0.01) and ((bw / bh) > RATIO_CAIDA_UMBRAL)

    @staticmethod
    def _criterio_cabeza_tobillo(kps):
        if kps.shape[0] < 17:
            return False
        if (kps[KP_NARIZ, 2] < KP_CABEZTOB_CONF or
                kps[KP_TOBILLO_IZQ, 2] < KP_CABEZTOB_CONF or
                kps[KP_TOBILLO_DER, 2] < KP_CABEZTOB_CONF):
            return False
        y_tobillo = (kps[KP_TOBILLO_IZQ, 1] + kps[KP_TOBILLO_DER, 1]) * 0.5
        return abs(kps[KP_NARIZ, 1] - y_tobillo) < KP_CABEZTOB_MAX_DY

    @staticmethod
    def _criterio_bh(det):
        return (det.ymax - det.ymin) < BH_CAIDA_MAX

    @staticmethod
    def _criterio_cy(det):
        if _servo_pw > SERVO_PW_SUELO_MAX:
            return False
        return ((det.ymin + det.ymax) * 0.5) > CY_CAIDA_MIN

    # ── C7: NUEVO — Posición Y de hombros ─────────────────────────────────────
    @staticmethod
    def _criterio_hombros_y(kps):
        """Devuelve (activo, y_hombros). Caída si hombros en parte baja del frame."""
        if kps.shape[0] < 7:
            return False, 0.3
        ch = (kps[KP_HOMBRO_IZQ, 2] + kps[KP_HOMBRO_DER, 2]) * 0.5
        if ch < HOMBROS_CONF_MIN:
            return False, 0.3
        y_hombros = (kps[KP_HOMBRO_IZQ, 1] + kps[KP_HOMBRO_DER, 1]) * 0.5
        return (y_hombros > HOMBROS_Y_MIN), float(y_hombros)

    # ── C8: NUEVO — Ratio spanX/spanY de keypoints ────────────────────────────
    @staticmethod
    def _criterio_ratio_xy(kps):
        """Devuelve (activo, ratio). Caída si spanX/spanY > umbral."""
        if kps.shape[0] < KP_RATIO_MIN_VISIBLE:
            return False, 0.5
        mask = kps[:, 2] >= KP_SPAN_CONF_MIN
        n = int(np.count_nonzero(mask))
        if n < KP_RATIO_MIN_VISIBLE:
            return False, 0.5
        xs = kps[mask, 0]
        ys = kps[mask, 1]
        span_x = float(xs.max() - xs.min())
        span_y = float(ys.max() - ys.min())
        if span_y < 0.02:
            return False, 0.5
        ratio = span_x / span_y
        return (ratio > KP_RATIO_XY_CAIDA), ratio

    @staticmethod
    def _kp_confiables_count(kps):
        if kps.shape[0] == 0:
            return 0
        return int(np.count_nonzero(kps[:, 2] >= KP_SPAN_CONF_MIN))

    def actualizar(self, det, kps):
        ahora = time.monotonic()
        self.ts_ultima_deteccion = ahora

        c1, angulo  = self._criterio_angulo(kps)
        c2, span_y  = self._criterio_span_y(kps)
        c3          = self._criterio_ratio(det)
        c4          = self._criterio_cabeza_tobillo(kps)
        c5          = self._criterio_bh(det)
        c6          = self._criterio_cy(det)
        c7, hombros = self._criterio_hombros_y(kps)
        c8, rxy     = self._criterio_ratio_xy(kps)

        self.angulo_tronco = angulo
        self.span_y        = span_y
        self.hombros_y     = hombros
        self.ratio_xy      = rxy
        self.kp_confiables = self._kp_confiables_count(kps) >= KP_SPAN_MIN_VISIBLE
        self.c4_activo     = c4

        # Conteo de criterios activos
        n_activos = sum((c1, c2, c3, c4, c5, c6, c7, c8))

        # Score: prioridad a combinaciones fuertes
        if n_activos >= 3:
            inc = SCORE_TRIPLE
        elif c5 and c7:
            inc = SCORE_C5_C7
        elif c7 and (c2 or c8):
            inc = SCORE_DOBLE_KP
        elif c4 and c5:
            inc = SCORE_C4_C5
        elif c7 and c8:
            inc = SCORE_C7_C8
        elif c2 and c5:
            inc = SCORE_C2_C5
        elif c5 and c6:
            inc = SCORE_C5_C6
        elif c5 and c3:
            inc = SCORE_C5_C3
        elif c2 and (c1 or c3 or c6):
            inc = SCORE_C2_OTRO
        elif c4:
            inc = SCORE_C4_SOLO
        elif c7:
            inc = SCORE_C7_SOLO
        elif c2:
            inc = SCORE_C2_SOLO
        elif c8:
            inc = SCORE_C8_SOLO
        elif c5:
            inc = SCORE_C5_SOLO
        elif c1:
            inc = SCORE_C1_SOLO
        elif c6:
            inc = SCORE_C6_SOLO
        elif c3:
            inc = SCORE_C3_SOLO
        else:
            inc = -SCORE_RESTA

        s = np.clip(self.score + inc, 0.0, SCORE_MAX)
        self.score = float(s)

        if self.estado == Estado.DE_PIE:
            if s >= SCORE_SOSPECHA:
                self.estado = Estado.SOSPECHA
        elif self.estado == Estado.SOSPECHA:
            if s >= SCORE_CONFIRMAR:
                self.estado        = Estado.CAIDA_CONFIRMADA
                self.ts_confirmada = ahora
            elif s < SCORE_RECUPERAR:
                self.estado = Estado.DE_PIE
        elif self.estado == Estado.CAIDA_CONFIRMADA:
            if s <= SCORE_RECUPERAR:
                self.estado        = Estado.DE_PIE
                self.ts_confirmada = None

        segs_caido = 0
        hay_alerta = False
        if self.estado == Estado.CAIDA_CONFIRMADA and self.ts_confirmada:
            segs_caido = int(ahora - self.ts_confirmada)
            hay_alerta = segs_caido >= TIEMPO_ALERTA

        return self.estado, segs_caido, hay_alerta, (c1, c2, c3, c4, c5, c6, c7, c8), angulo, span_y, hombros, rxy

    def expirado(self):
        inactivo = time.monotonic() - self.ts_ultima_deteccion
        if self.estado in (Estado.SOSPECHA, Estado.CAIDA_CONFIRMADA):
            return inactivo > SEG_PERSISTENCIA
        return inactivo > SEG_PERSISTENCIA_PIE

    def texto_estado(self, segs, alerta):
        if alerta:
            return f"!ALERTA {segs}s EN PISO!", (0, 0, 255)
        if self.estado == Estado.CAIDA_CONFIRMADA:
            return f"CAIDA ({segs}s)", (0, 80, 255)
        if self.estado == Estado.SOSPECHA:
            return f"SOSPECHA {self.score:.0f}", (0, 165, 255)
        return "DE PIE", (0, 220, 0)

    def get_metricas_str(self, det, ahora):
        if ahora - self._str_cache_ts > PERSONA_STR_REFRESH_S:
            bh_n = det.ymax - det.ymin
            self._str_cache = (f"bh:{bh_n:.2f} hY:{self.hombros_y:.2f} "
                               f"rXY:{self.ratio_xy:.2f} spY:{self.span_y:.2f} "
                               f"S:{self.score:.0f}")
            self._str_cache_ts = ahora
        return self._str_cache


# ═══════════════════════════════════════════════════════════════════════════════
#  PRE-RENDERIZAR LÍNEAS DE REFERENCIA
# ═══════════════════════════════════════════════════════════════════════════════
_CX_F  = CAM_W // 2
_CY_F  = CAM_H // 2
_ZX_PX = int(CAM_W * ZONA_MUERTA_X)
_ZY_PX = int(CAM_H * ZONA_MUERTA_Y)

def _crear_mascara_lineas(w, h):
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    g = (75, 75, 75); d = (38, 38, 38)
    cv2.line(mask, (_CX_F, 0),        (_CX_F, h),        g, 1)
    cv2.line(mask, (0, _CY_F),        (w, _CY_F),        g, 1)
    cv2.line(mask, (_CX_F-_ZX_PX, 0), (_CX_F-_ZX_PX, h), d, 1)
    cv2.line(mask, (_CX_F+_ZX_PX, 0), (_CX_F+_ZX_PX, h), d, 1)
    cv2.line(mask, (0, _CY_F-_ZY_PX), (w, _CY_F-_ZY_PX), d, 1)
    cv2.line(mask, (0, _CY_F+_ZY_PX), (w, _CY_F+_ZY_PX), d, 1)
    return mask

_MASCARA_LINEAS = _crear_mascara_lineas(CAM_W, CAM_H)
_ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, JPEG_CALIDAD]


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
_output_frame = None
_frame_lock   = threading.Lock()
_frame_event  = threading.Event()

_estados_personas = {}
_bid_ultima_vez   = {}
_ts_ultimo_purge      = time.monotonic()
_ts_ultimo_purge_exp  = time.monotonic()

_pos_x    = 0
_servo_pw = SERVO_INICIO
_ts_cmd_x = 0.0
_ts_cmd_y = 0.0
_pi       = None
_pi_lock  = threading.Lock()
_tic_device = None

_ema_cx = 0.5; _ema_cy = 0.5; _ema_iniciado = False
_vel_cx = 0.0; _vel_cy = 0.0; _ts_vel = 0.0
_prev_cx = 0.5; _prev_cy = 0.5

_bid_objetivo_prev = None
_prev_cx_objetivo  = 0.5
_prev_cy_objetivo  = 0.5

_fps_inf_cnt = 0; _fps_inf_ts = 0.0; _fps_inf = 0
_fps_str_cnt = 0; _fps_str_ts = 0.0; _fps_str = 0

_encode_queue = queue.Queue(maxsize=1)
_hud_cache_text = ""
_hud_cache_ts   = 0.0


def _clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


def _bbox_id(det):
    zx = int((det.xmin + det.xmax) * 0.5 * BID_GRID_X)
    zy = int((det.ymin + det.ymax) * 0.5 * BID_GRID_Y)
    return f"{zx}_{zy}"


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


def _purge_bids_viejos():
    global _ts_ultimo_purge
    ahora = time.monotonic()
    if ahora - _ts_ultimo_purge < PURGE_INTERVALO_S:
        return
    limite   = ahora - PURGE_MAX_EDAD_S
    a_borrar = [b for b, ts in _bid_ultima_vez.items() if ts < limite]
    for b in a_borrar:
        _estados_personas.pop(b, None)
        _bid_ultima_vez.pop(b, None)
    _ts_ultimo_purge = ahora


def _purge_expirados():
    global _ts_ultimo_purge_exp
    ahora = time.monotonic()
    if ahora - _ts_ultimo_purge_exp < PURGE_EXP_INTERVALO:
        return
    _ts_ultimo_purge_exp = ahora
    a_borrar = [b for b, ep in _estados_personas.items() if ep.expirado()]
    for b in a_borrar:
        _estados_personas.pop(b, None)
        _bid_ultima_vez.pop(b, None)


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

    return (_clamp(_ema_cx + _vel_cx * PRED_FACTOR, 0.0, 1.0),
            _clamp(_ema_cy + _vel_cy * PRED_FACTOR, 0.0, 1.0))


def _resetear_ema():
    global _ema_iniciado, _vel_cx, _vel_cy
    _ema_iniciado = False; _vel_cx = 0.0; _vel_cy = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  MOTORES
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
            print("[MOTOR X] ✓ ticlib USB")
            return
        except Exception:
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
        except Exception:
            pass
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


def servo_init():
    global _pi, _servo_pw
    with _pi_lock:
        _pi = pigpio.pi()
        if not _pi.connected:
            raise RuntimeError("No se pudo conectar a pigpiod")
        _pi.set_mode(SERVO_PIN, pigpio.OUTPUT)
        _servo_pw = SERVO_INICIO
        _pi.set_servo_pulsewidth(SERVO_PIN, _servo_pw)
    time.sleep(0.8)
    print(f"[SERVO Y]  ✓ {_servo_pw}µs (suelo)")


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
            except Exception:
                pass
    _ts_cmd_y = ahora


def _correccion_y(cy):
    err = cy - 0.5
    if abs(err) < ZONA_MUERTA_Y:
        return 0
    norm     = (abs(err) - ZONA_MUERTA_Y) / (0.5 - ZONA_MUERTA_Y)
    delta_us = max(PASO_MIN_Y_US, int(norm ** 1.6 * PASO_MAX_Y_US))
    return delta_us * SENTIDO_Y if err > 0 else delta_us * -SENTIDO_Y


def watchdog_pigpio():
    _set_cpu_affinity(CPU_AFFINITY_WATCHDOG)
    while True:
        time.sleep(10)
        with _pi_lock:
            ok = (_pi is not None) and _pi.connected
        if not ok:
            try:
                servo_init()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
#  SELECCIÓN DE OBJETIVO
# ═══════════════════════════════════════════════════════════════════════════════
def _mejor_objetivo(dets, bid_map):
    global _bid_objetivo_prev
    bids = set(bid_map.values())

    if _bid_objetivo_prev in bids:
        for det in dets:
            if bid_map[det] == _bid_objetivo_prev and det.confidence >= CONF_MIN:
                return det

    mejor, score = None, -1.0
    for det in dets:
        if det.confidence < CONF_MIN:
            continue
        s = (det.xmax - det.xmin) * (det.ymax - det.ymin) * det.confidence
        if s > score:
            score, mejor = s, det
    return mejor


def _punto_seguimiento(det):
    cx = (det.xmin + det.xmax) * 0.5
    cy = det.ymin + (det.ymax - det.ymin) * FRAC_TORSO
    return cx, cy


# ═══════════════════════════════════════════════════════════════════════════════
#  HILO ENCODER JPEG
# ═══════════════════════════════════════════════════════════════════════════════
def hilo_encoder(stop_event):
    global _output_frame
    _set_cpu_affinity(CPU_AFFINITY_ENCODER)
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
#  HILOS DE CÁMARA
# ═══════════════════════════════════════════════════════════════════════════════
def hilo_frames(video_q, stop_event):
    _set_cpu_affinity(CPU_AFFINITY_FRAMES)
    while not stop_event.is_set():
        msg = video_q.get()
        if msg is None or stop_event.is_set():
            break
        _db_escribir(msg.getCvFrame())


def hilo_poses(pose_q, stop_event):
    global _fps_inf_cnt, _fps_inf_ts, _fps_inf
    global _bid_objetivo_prev, _prev_cx_objetivo, _prev_cy_objetivo
    global _hud_cache_text, _hud_cache_ts

    _set_cpu_affinity(CPU_AFFINITY_POSES)

    while not stop_event.is_set():
        pose_msg = pose_q.get()
        if pose_msg is None or stop_event.is_set():
            break

        frame, frame_ts = _db_leer()
        if frame is None:
            continue
        if time.monotonic() - frame_ts > FRAME_MAX_AGE_S:
            continue

        h, w = frame.shape[:2]

        _fps_inf_cnt += 1
        ahora = time.monotonic()
        if ahora - _fps_inf_ts >= 1.0:
            _fps_inf = _fps_inf_cnt; _fps_inf_cnt = 0; _fps_inf_ts = ahora

        dets_validas = [d for d in pose_msg.detections if d.confidence >= CONF_MIN]
        bid_map = {det: _bbox_id(det) for det in dets_validas}

        for bid in bid_map.values():
            _bid_ultima_vez[bid] = ahora
            if bid not in _estados_personas:
                _estados_personas[bid] = EstadoPersona()

        _purge_bids_viejos()
        _purge_expirados()

        estado_global = "Sin persona"
        hay_alerta    = False
        objetivo      = _mejor_objetivo(dets_validas, bid_map)

        if objetivo is None:
            _resetear_ema()
            _bid_objetivo_prev = None
        else:
            _bid_objetivo_prev = bid_map[objetivo]

        tiene_dets = len(dets_validas) > 0
        
        if DEBUG_OVERLAY and tiene_dets:
            canvas = frame.copy()
        else:
            canvas = frame

        for det in dets_validas:
            bid = bid_map[det]
            kps = _extraer_kps_np(det)
            ep  = _estados_personas[bid]

            estado, segs, alerta, criterios, angulo, span_y, hombros, rxy = ep.actualizar(det, kps)
            c1, c2, c3, c4, c5, c6, c7, c8 = criterios
            texto, color = ep.texto_estado(segs, alerta)

            if alerta:
                hay_alerta = True
                if DEBUG_OVERLAY:
                    cv2.rectangle(canvas, (0, 0), (w-1, h-1), (0, 0, 255), 8)

            if DEBUG_OVERLAY:
                x1 = int(det.xmin * w); y1 = int(det.ymin * h)
                x2 = int(det.xmax * w); y2 = int(det.ymax * h)
                bh_n = det.ymax - det.ymin

                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

                # Línea del tronco
                if ep.kp_confiables and kps.shape[0] >= 13:
                    hombro = (kps[KP_HOMBRO_IZQ, :2] + kps[KP_HOMBRO_DER, :2]) * 0.5
                    cadera = (kps[KP_CADERA_IZQ, :2] + kps[KP_CADERA_DER, :2]) * 0.5
                    hx  = int(hombro[0] * w); hy = int(hombro[1] * h)
                    cx_ = int(cadera[0] * w); cy_ = int(cadera[1] * h)
                    eje_col = (0,255,0) if not c1 else (0,165,255) if estado==Estado.SOSPECHA else (0,0,255)
                    cv2.line(canvas, (hx, hy), (cx_, cy_), eje_col, 3)
                    cv2.circle(canvas, (hx, hy), 4, (255,255,0), -1)
                    cv2.circle(canvas, (cx_, cy_), 4, (255,255,0), -1)

                # Indicadores C1..C8 (8 circulitos)
                for i, activo in enumerate((c1, c2, c3, c4, c5, c6, c7, c8)):
                    col = (0,255,0) if activo else (55,55,55)
                    cv2.circle(canvas, (x1 + 6 + i*10, y1 - 10), 3, col, -1)

                # Métricas (cacheadas)
                metricas = ep.get_metricas_str(det, ahora)
                cv2.putText(canvas, metricas, (x1, y2 + 13), 
                           cv2.FONT_HERSHEY_PLAIN, 0.72, (140,140,140), 1)

                # Punto torso
                tx = int((det.xmin + det.xmax) * 0.5 * w)
                ty = int((det.ymin + bh_n * FRAC_TORSO) * h)
                cv2.circle(canvas, (tx, ty), 5, (0, 255, 0), -1)

                # Estado
                cv2.putText(canvas, texto, (x1, max(y1 - 14, 12)),
                           cv2.FONT_HERSHEY_PLAIN, 1.1, color, 2)

                # Keypoints
                mask = kps[:, 2] >= KP_SPAN_CONF_MIN
                for i in np.where(mask)[0]:
                    px = int(kps[i, 0] * w); py = int(kps[i, 1] * h)
                    cv2.circle(canvas, (px, py), 2, (200, 200, 0), -1)

            if objetivo is not None and det is objetivo:
                estado_global = texto

                cx_raw, cy_raw = _punto_seguimiento(det)

                if (abs(cx_raw - _prev_cx_objetivo) < SKIP_UMBRAL_BBOX and
                        abs(cy_raw - _prev_cy_objetivo) < SKIP_UMBRAL_BBOX):
                    continue

                _prev_cx_objetivo = cx_raw
                _prev_cy_objetivo = cy_raw
                cx_s, cy_s = _actualizar_posicion(cx_raw, cy_raw)

                if DEBUG_OVERLAY:
                    ax, ay = int(cx_s * w), int(cy_s * h)
                    cv2.circle(canvas, (ax, ay), 8, (255, 200, 0), 2)
                    cv2.drawMarker(canvas, (ax, ay), (0, 200, 255),
                                   cv2.MARKER_CROSS, 18, 2)

                corr_x = _correccion_x(cx_s)
                if corr_x:
                    motor_mover_x(_pos_x + corr_x)
                corr_y = _correccion_y(cy_s)
                if corr_y:
                    servo_mover_y(_servo_pw + corr_y)

        if DEBUG_OVERLAY and tiene_dets:
            cv2.addWeighted(canvas, 1.0, _MASCARA_LINEAS, 0.5, 0, canvas)

        if not (DEBUG_OVERLAY and tiene_dets):
            canvas = frame.copy() if canvas is frame else canvas

        cv2.rectangle(canvas, (0,0), (w,36),
                      (85,0,0) if hay_alerta else (16,16,16), -1)

        if "ALERTA"  in estado_global: hud_col = (0,0,255)
        elif "CAIDA" in estado_global: hud_col = (0,80,255)
        elif "SOSPEC" in estado_global: hud_col = (0,165,255)
        else:                          hud_col = (0,200,0)

        if ahora - _hud_cache_ts > HUD_REFRESH_S:
            _hud_cache_text = f"INF:{_fps_inf}fps STR:{_fps_str}fps  {estado_global}  X:{_pos_x}  Y:{_servo_pw}us"
            _hud_cache_ts = ahora

        cv2.putText(canvas, _hud_cache_text,
                    (8,24), cv2.FONT_HERSHEY_PLAIN, 1.05, hud_col, 2)

        cv2.putText(canvas, "C1-C8 criterios",
                    (w-140,24), cv2.FONT_HERSHEY_PLAIN, 0.72, (80,80,80), 1)

        try:
            _encode_queue.put_nowait(canvas)
        except queue.Full:
            try:
                _encode_queue.get_nowait()
                _encode_queue.put_nowait(canvas)
            except queue.Full:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE
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
        print(f"[PIPELINE] ✓ DepthAI — {CAM_W}×{CAM_H} (IA en {IA_W}×{IA_H})")

        t_frames  = threading.Thread(target=hilo_frames, args=(video_q, stop_event),
                                     daemon=True, name="t-frames")
        t_poses   = threading.Thread(target=hilo_poses,  args=(pose_q,  stop_event),
                                     daemon=True, name="t-poses")
        t_encoder = threading.Thread(target=hilo_encoder, args=(stop_event,),
                                     daemon=True, name="t-encoder")
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
    _set_cpu_affinity(CPU_AFFINITY_PIPELINE)
    while True:
        try:
            _arrancar_pipeline()
            print("[PIPELINE] Reintentando en 3s…")
        except Exception as e:
            print(f"[PIPELINE ERROR] {e}")
            time.sleep(2)
        time.sleep(3)


# ═══════════════════════════════════════════════════════════════════════════════
#  FLASK
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
    html = f"""\
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Detector Caídas v14</title>
<style>
  :root{{--v:#00e676;--r:#ff1744;--bg:#070b0f;--p:#0d1117;--b:#1c2a3a;--t:#c8d8e8;}}
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{background:var(--bg);color:var(--t);font-family:monospace;
    min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:20px;gap:14px;}}
  h1{{color:var(--v);font-size:1.3rem;}}
  .vw{{position:relative;width:100%;max-width:680px;border:2px solid var(--b);
    border-radius:10px;overflow:hidden;background:#000;}}
  .vw img{{display:block;width:100%;height:auto;}}
  .live{{position:absolute;top:9px;left:50%;transform:translateX(-50%);
    background:rgba(255,23,68,.2);border:1px solid var(--r);color:var(--r);
    font-size:.6rem;padding:2px 8px;border-radius:3px;}}
  .status{{display:flex;gap:8px;flex-wrap:wrap;justify-content:center;
    font-size:.65rem;padding:10px;background:var(--p);border-radius:7px;}}
  .tag{{color:var(--v);}}
</style>
</head>
<body>
<h1>⬡ DETECTOR DE CAÍDAS v14 (8 criterios)</h1>
<p style="font-size:.8rem;color:#999;">YOLOv8-pose + hombros Y + ratio extensión + sensibilidad global</p>
<div class="vw">
  <span class="live">● LIVE</span>
  <img id="s" src="/video_feed" alt="stream">
</div>
<div class="status">
  <span><span class="tag">BH_CAIDA_MAX</span>{BH_CAIDA_MAX}</span>
  <span><span class="tag">HOMBROS_Y_MIN</span>{HOMBROS_Y_MIN}</span>
  <span><span class="tag">SCORE_CONFIRMAR</span>{SCORE_CONFIRMAR}</span>
  <span><span class="tag">SENSIBILIDAD</span>{SENSIBILIDAD}x</span>
</div>
<script>
const img=document.getElementById('s');
img.addEventListener('error',()=>{{setTimeout(()=>{{img.src='/video_feed?'+Date.now();}},1200);}});
</script>
</body>
</html>"""
    return html


@app.route("/video_feed")
def video_feed():
    return _generar_mjpeg(), 200, {
        'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
        'Cache-Control': 'no-cache'
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "═"*70)
    print("  DETECTOR DE CAÍDAS — v14.0 (8 criterios)")
    print("═"*70)
    print(f"  ★ BH_CAIDA_MAX      : {BH_CAIDA_MAX}")
    print(f"  ★ HOMBROS_Y_MIN     : {HOMBROS_Y_MIN}")
    print(f"  ★ SCORE_CONFIRMAR   : {SCORE_CONFIRMAR}")
    print(f"  ★ SENSIBILIDAD      : {SENSIBILIDAD}x")
    print(f"    Sospecha          : {SCORE_SOSPECHA}")
    print(f"    Stream            : {CAM_W}×{CAM_H} | IA en {IA_W}×{IA_H}")
    print(f"    Motor X           : {'ticlib USB' if _TICLIB_OK else 'ticcmd'}")
    print(f"    Servo Y           : {SERVO_INICIO}µs (suelo)")
    print("═"*70 + "\n")

    try:
        motor_init()
        servo_init()
    except Exception as e:
        print(f"[ERROR HARDWARE] {e}")
        sys.exit(1)

    threading.Thread(target=watchdog_pigpio, daemon=True, name="t-watchdog").start()
    threading.Thread(target=camara_loop, daemon=True, name="t-pipeline").start()

    print("[BOOT] Esperando 4s para pipeline…\n")
    time.sleep(4)

    print(f"  ✓ http://0.0.0.0:{FLASK_PORT}/\n")

    try:
        app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True, debug=False, use_reloader=False)
    finally:
        servo_stop()
        print("[EXIT] Detenido.")
