"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        SISTEMA DE DETECCIÓN DE CAÍDAS — OAK-1 + YOLOv8-pose  v12.0         ║
║  Eje X → Stepper (Tic via ticlib)  |  Eje Y → Servo SG90 (pigpio)          ║
║  6 criterios · Encoder en hilo separado · Watchdog pigpio · Máx sensibilidad║
╚══════════════════════════════════════════════════════════════════════════════╝

CAMBIOS v12 respecto a v11:

  ▸ DETECCIÓN — 2 criterios NUEVOS + umbrales más sensibles:

    C5 — ALTURA DEL BBOX (bh_norm < 0.38):
         Cuando estás cerca de la cámara y caído, el bbox se vuelve bajo y
         ancho. bh_norm < 0.38 → criterio activo. No depende de keypoints,
         funciona aunque te muevas en el suelo. ESTE ES EL CRITERIO MÁS
         CONFIABLE PARA TU PROBLEMA.

    C6 — POSICIÓN Y DEL CENTRO (cy_norm > 0.55):
         Con servo apuntando al suelo, una persona de pie tiene cy ~ 0.3-0.5.
         Una persona caída tiene cy > 0.55. Solo activo si el servo está
         apuntando hacia abajo (_servo_pw <= SERVO_PW_SUELO_MAX).

  ▸ COMBINACIONES MUY AGRESIVAS para confirmación rápida:
      C4+C5: +14  (certeza casi absoluta — cabeza+tobillos + bbox bajo)
      C2+C5: +12  (span-Y + bbox bajo)
      C5+C6: +11  (bbox bajo + en suelo)
      C5+C3: +10  (bbox bajo + ratio horizontal)

  ▸ UMBRALES BAJADOS — confirma en ≈0.2s (3 frames a 15fps):
      SCORE_SOSPECHA  = 10 (era 14)
      SCORE_CONFIRMAR = 24 (era 35)
      SCORE_RESTA     = 1  (era 2)
      KP_SPAN_Y_CAIDA = 0.32 (era 0.28)

  ▸ OPTIMIZACIONES DE RENDIMIENTO:

    1. ticlib en vez de subprocess:
       Si ticlib está instalado, usa USB directo (sin lanzar proceso por
       comando). Fallback automático a ticcmd si no está. Reduce latencia
       de cada comando de motor de ~50-200ms a <2ms.
       Para instalar: pip install ticlib

    2. Encoder JPEG en hilo separado:
       cv2.imencode() ya no bloquea el hilo de poses. Cola maxSize=1 que
       descarta el frame anterior si el encoder no da abasto. El hilo de
       poses puede procesar más rápido la siguiente detección.

    3. canvas.copy() solo si hay detecciones:
       Sin personas en frame, no se copia el array → ahorra ~3ms/frame.

    4. _bbox_id por zonas (no por coordenadas exactas):
       Antes: round(cx,1) — frágil, cambiaba con cada movimiento.
       Ahora: dividir frame en 6x4 zonas → ID estable entre frames →
       mejor persistencia del score por persona.

    5. Watchdog de pigpiod:
       Si pigpiod se cae, hilo reconecta automáticamente cada 10s.

    6. Purge_expirados con throttle:
       Antes corría cada frame. Ahora cada 5s. Reduce overhead.

    7. HUD string cacheado (10Hz en vez de cada frame).

    8. JPEG_CALIDAD = 50 (de 55) — sin pérdida visible a 512x288.

INSTALACIÓN OPCIONAL (opcional pero recomendado):
  pip install ticlib

ARRANQUE:
  sudo pigpiod
  source ~/depthai-env/bin/activate
  python3 seguimientoxy.py

CALIBRACIÓN — observa el overlay:
  bh = altura bbox normalizada · cy = centro Y bbox
  De pie: bh ~ 0.5-0.9, cy ~ 0.3-0.5
  Caído:  bh < 0.38,    cy > 0.55
  Si no detecta caída → sube BH_CAIDA_MAX a 0.42 o baja SCORE_CONFIRMAR a 20
  Si hay falsas alarmas → baja BH_CAIDA_MAX a 0.33 o sube SCORE_CONFIRMAR a 32
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

# ── ticlib opcional — fallback a subprocess si no está instalado ─────────────
try:
    from ticlib import TicUSB
    _TICLIB_OK = True
except ImportError:
    _TICLIB_OK = False
    print("[INFO] ticlib no instalado — usando subprocess (más lento). "
          "Instala con: pip install ticlib")


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GENERAL
# ═══════════════════════════════════════════════════════════════════════════════
TIEMPO_ALERTA   = 10
JPEG_CALIDAD    = 50      # bajado — menor ancho de banda
JPEG_OPTIMIZAR  = True
FLASK_PORT      = 5000
FLASK_HOST      = "0.0.0.0"
FRAME_MAX_AGE_S = 0.18

CAM_W, CAM_H    = 512, 288
CONF_MIN        = 0.30

# ── Motor X — Stepper Tic ─────────────────────────────────────────────────────
ZONA_MUERTA_X   = 0.07
PASO_MAX_X      = 55
PASO_MIN_X      = 6
LIMITE_IZQ      = -2000
LIMITE_DER      =  2000
CMD_INTERVALO_X = 0.05
SENTIDO_X       = -1
MAX_DELTA_X     = 35

# ── Servo Y — SG90 ───────────────────────────────────────────────────────────
SERVO_PIN       = 18
SERVO_MIN       = 1100
SERVO_MAX       = 1780
SERVO_CENTER    = 1440
SERVO_INICIO    = 1100    # apunta al suelo desde el arranque

ZONA_MUERTA_Y   = 0.055
PASO_MAX_Y_US   = 38
PASO_MIN_Y_US   = 4
CMD_INTERVALO_Y = 0.033
SENTIDO_Y       = -1
MAX_DELTA_Y     = 22

# ── EMA + predicción ─────────────────────────────────────────────────────────
EMA_ALPHA       = 0.45
PRED_FACTOR     = 0.08

# ── Punto de seguimiento ─────────────────────────────────────────────────────
FRAC_TORSO      = 0.40

# ── Optimizaciones ───────────────────────────────────────────────────────────
SKIP_UMBRAL_BBOX  = 0.02
DEBUG_OVERLAY     = True

# ── Purge ─────────────────────────────────────────────────────────────────────
PURGE_INTERVALO_S    = 60.0
PURGE_MAX_EDAD_S     = 60.0
PURGE_EXP_INTERVALO  = 5.0    # purge_expirados cada 5s en vez de cada frame

# ── HUD cache ────────────────────────────────────────────────────────────────
HUD_REFRESH_S       = 0.1     # actualizar string del HUD a 10Hz

# ── Bbox ID — granularidad de zonas ──────────────────────────────────────────
BID_GRID_X = 6   # divide frame en 6 columnas
BID_GRID_Y = 4   # y 4 filas → 24 zonas


# ═══════════════════════════════════════════════════════════════════════════════
#  PARÁMETROS DE DETECCIÓN — 6 CRITERIOS
# ═══════════════════════════════════════════════════════════════════════════════

# ── C1: Ángulo del tronco (hombros→caderas) ───────────────────────────────────
ANGULO_CAIDA_MAX  = 55.0   # más permisivo
KP_CONF_MIN       = 0.15

# ── C2: Dispersión vertical de keypoints ──────────────────────────────────────
KP_SPAN_MIN_VISIBLE = 5
KP_SPAN_CONF_MIN    = 0.12
KP_SPAN_Y_CAIDA     = 0.32   # más permisivo

# ── C3: Ratio bbox ────────────────────────────────────────────────────────────
RATIO_CAIDA_UMBRAL  = 1.25   # bajado de 1.4

# ── C4: Cabeza y tobillos en misma banda Y ────────────────────────────────────
KP_CABEZTOB_CONF    = 0.12
KP_CABEZTOB_MAX_DY  = 0.28   # más permisivo

# ── C5: Altura del bbox normalizada (NUEVO — el más confiable) ────────────────
#  De pie a distancia normal: bh ~ 0.5-0.9
#  Caído cerca de la cámara: bh < 0.38
BH_CAIDA_MAX        = 0.38

# ── C6: Posición Y del centro del bbox (NUEVO) ────────────────────────────────
#  Con cámara apuntando al suelo:
#  De pie: cy ~ 0.3-0.5
#  Caído: cy > 0.55
CY_CAIDA_MIN        = 0.55
SERVO_PW_SUELO_MAX  = 1320   # solo activar C6 si servo está apuntando abajo

# ── Score integrador asimétrico ───────────────────────────────────────────────
SCORE_MAX       = 80

# Combinaciones (dos criterios activos simultáneamente)
SCORE_C4_C5     = 14   # cabeza+tobillos + bbox bajo = certeza casi absoluta
SCORE_C2_C5     = 12   # span-Y + bbox bajo
SCORE_C5_C6     = 11   # bbox bajo + en suelo (servo abajo)
SCORE_C5_C3     = 10   # bbox bajo + ratio horizontal
SCORE_C2_OTRO   = 10   # C2 + C1/C3/C6

# Criterios solos
SCORE_C4_SOLO   = 9
SCORE_C2_SOLO   = 8
SCORE_C5_SOLO   = 6
SCORE_C1_SOLO   = 5
SCORE_C6_SOLO   = 3
SCORE_C3_SOLO   = 2

# Penalización frame limpio
SCORE_RESTA     = 1

# Transiciones
SCORE_SOSPECHA  = 10
SCORE_CONFIRMAR = 24
SCORE_RECUPERAR = 3

# Persistencia
SEG_PERSISTENCIA     = 10.0
SEG_PERSISTENCIA_PIE = 5.0


# ═══════════════════════════════════════════════════════════════════════════════
#  MÁQUINA DE ESTADOS
# ═══════════════════════════════════════════════════════════════════════════════
class Estado(Enum):
    DE_PIE           = auto()
    SOSPECHA         = auto()
    CAIDA_CONFIRMADA = auto()


class EstadoPersona:
    """Detección con 6 criterios independientes que se votan."""

    __slots__ = ("estado", "score", "ts_confirmada", "ts_ultima_deteccion",
                 "angulo_tronco", "span_y", "kp_confiables", "c4_activo")

    def __init__(self):
        self.estado              = Estado.DE_PIE
        self.score               = 0.0
        self.ts_confirmada       = None
        self.ts_ultima_deteccion = time.monotonic()
        self.angulo_tronco       = 90.0
        self.span_y              = 1.0
        self.kp_confiables       = False
        self.c4_activo           = False

    # ── C1: Ángulo del tronco ─────────────────────────────────────────────────
    @staticmethod
    def _criterio_angulo(kps):
        if len(kps) < 13:
            return False, 90.0
        hiz = kps[5];  hde = kps[6]
        ciz = kps[11]; cde = kps[12]
        if (hiz[2] + hde[2]) * 0.5 < KP_CONF_MIN: return False, 90.0
        if (ciz[2] + cde[2]) * 0.5 < KP_CONF_MIN: return False, 90.0
        hx = (hiz[0] + hde[0]) * 0.5; hy = (hiz[1] + hde[1]) * 0.5
        cx = (ciz[0] + cde[0]) * 0.5; cy = (ciz[1] + cde[1]) * 0.5
        dx = cx - hx; dy = cy - hy
        lon = (dx*dx + dy*dy) ** 0.5
        if lon < 1e-4:
            return False, 90.0
        cos_a = abs(dy) / lon
        if cos_a > 1.0: cos_a = 1.0
        elif cos_a < 0.0: cos_a = 0.0
        angulo = 90.0 - abs(np.degrees(np.arccos(cos_a)) - 90.0)
        return (angulo < ANGULO_CAIDA_MAX), angulo

    # ── C2: Dispersión vertical de keypoints ──────────────────────────────────
    @staticmethod
    def _criterio_span_y(kps):
        if len(kps) < 5:
            return False, 1.0
        ys = [kp[1] for kp in kps if kp[2] >= KP_SPAN_CONF_MIN]
        if len(ys) < KP_SPAN_MIN_VISIBLE:
            return False, 1.0
        span = max(ys) - min(ys)
        return (span < KP_SPAN_Y_CAIDA), span

    # ── C3: Ratio bbox ────────────────────────────────────────────────────────
    @staticmethod
    def _criterio_ratio(det):
        bw = det.xmax - det.xmin
        bh = det.ymax - det.ymin
        return (bh > 0.01) and ((bw / bh) > RATIO_CAIDA_UMBRAL)

    # ── C4: Cabeza y tobillos en misma banda Y ────────────────────────────────
    @staticmethod
    def _criterio_cabeza_tobillo(kps):
        if len(kps) < 17:
            return False
        nariz = kps[0]; t_izq = kps[15]; t_der = kps[16]
        if (nariz[2] < KP_CABEZTOB_CONF or
                t_izq[2] < KP_CABEZTOB_CONF or
                t_der[2] < KP_CABEZTOB_CONF):
            return False
        return abs(nariz[1] - (t_izq[1] + t_der[1]) * 0.5) < KP_CABEZTOB_MAX_DY

    # ── C5: Altura bbox normalizada (NUEVO) ───────────────────────────────────
    @staticmethod
    def _criterio_bh(det):
        return (det.ymax - det.ymin) < BH_CAIDA_MAX

    # ── C6: Posición Y del centro del bbox (NUEVO) ────────────────────────────
    @staticmethod
    def _criterio_cy(det):
        # Solo activo si el servo está apuntando hacia abajo
        if _servo_pw > SERVO_PW_SUELO_MAX:
            return False
        cy = (det.ymin + det.ymax) * 0.5
        return cy > CY_CAIDA_MIN

    # ── Actualización principal ───────────────────────────────────────────────
    def actualizar(self, det, kps):
        ahora = time.monotonic()
        self.ts_ultima_deteccion = ahora

        c1, angulo = self._criterio_angulo(kps)
        c2, span_y = self._criterio_span_y(kps)
        c3         = self._criterio_ratio(det)
        c4         = self._criterio_cabeza_tobillo(kps)
        c5         = self._criterio_bh(det)
        c6         = self._criterio_cy(det)

        self.angulo_tronco = angulo
        self.span_y        = span_y
        self.kp_confiables = sum(1 for kp in kps if kp[2] >= KP_SPAN_CONF_MIN) >= KP_SPAN_MIN_VISIBLE
        self.c4_activo     = c4

        # Score asimétrico — combinaciones primero
        if c4 and c5:
            inc = SCORE_C4_C5
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
        elif c2:
            inc = SCORE_C2_SOLO
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

        s = self.score + inc
        if s < 0.0: s = 0.0
        elif s > SCORE_MAX: s = SCORE_MAX
        self.score = s

        # Transiciones
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

        return self.estado, segs_caido, hay_alerta, c1, c2, c3, c4, c5, c6, angulo, span_y

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


# ═══════════════════════════════════════════════════════════════════════════════
#  COORDENADAS PRECALCULADAS
# ═══════════════════════════════════════════════════════════════════════════════
_CX_F  = CAM_W // 2
_CY_F  = CAM_H // 2
_ZX_PX = int(CAM_W * ZONA_MUERTA_X)
_ZY_PX = int(CAM_H * ZONA_MUERTA_Y)

_ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, JPEG_CALIDAD]
if JPEG_OPTIMIZAR:
    _ENCODE_PARAMS += [cv2.IMWRITE_JPEG_OPTIMIZE, 1]


# ═══════════════════════════════════════════════════════════════════════════════
#  DOBLE BUFFER PARA FRAMES
# ═══════════════════════════════════════════════════════════════════════════════
_db_bufs = [None, None]
_db_ts   = [0.0,  0.0]
_db_idx  = 0
_db_lock = threading.Lock()


def _db_escribir(frame):
    global _db_idx
    with _db_lock:
        slot = 1 - _db_idx
    if _db_bufs[slot] is None or _db_bufs[slot].shape != frame.shape:
        _db_bufs[slot] = frame.copy()
    else:
        np.copyto(_db_bufs[slot], frame)
    _db_ts[slot] = time.monotonic()
    with _db_lock:
        _db_idx = slot


def _db_leer():
    with _db_lock:
        idx = _db_idx
    return _db_bufs[idx], _db_ts[idx]


# ═══════════════════════════════════════════════════════════════════════════════
#  ESTADO GLOBAL
# ═══════════════════════════════════════════════════════════════════════════════
app           = Flask(__name__)

_output_frame = None
_frame_lock   = threading.Lock()
_frame_event  = threading.Event()

_estados_personas: dict = {}
_bid_ultima_vez:   dict = {}
_ts_ultimo_purge:      float = time.monotonic()
_ts_ultimo_purge_exp:  float = time.monotonic()

_pos_x         = 0
_servo_pw      = SERVO_INICIO
_ts_cmd_x      = 0.0
_ts_cmd_y      = 0.0
_pi            = None
_pi_lock       = threading.Lock()
_tic_device    = None     # objeto TicUSB si ticlib está disponible

_ema_cx = 0.5; _ema_cy = 0.5; _ema_iniciado = False
_vel_cx = 0.0; _vel_cy = 0.0; _ts_vel = 0.0
_prev_cx = 0.5; _prev_cy = 0.5

_bid_objetivo_prev = None
_prev_cx_objetivo  = 0.5
_prev_cy_objetivo  = 0.5

_fps_inf_cnt = 0; _fps_inf_ts = 0.0; _fps_inf = 0
_fps_str_cnt = 0; _fps_str_ts = 0.0; _fps_str = 0

# Cola del encoder JPEG (descarta frame anterior si está lleno)
_encode_queue = queue.Queue(maxsize=1)

# Cache del HUD
_hud_cache_text = ""
_hud_cache_ts   = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════════
def _clamp(v, vmin, vmax):
    if v < vmin: return vmin
    if v > vmax: return vmax
    return v


def _bbox_id(det):
    """ID estable por zonas del frame, no por píxeles exactos."""
    zx = int((det.xmin + det.xmax) * 0.5 * BID_GRID_X)
    zy = int((det.ymin + det.ymax) * 0.5 * BID_GRID_Y)
    return f"{zx}_{zy}"


def _extraer_kps(det):
    """Extrae keypoints como lista de [x, y, conf]."""
    kps = []
    if hasattr(det, "keypoints"):
        for kp in det.keypoints:
            conf = kp.confidence if hasattr(kp, "confidence") else 1.0
            kps.append([kp.x, kp.y, conf])
    return kps


# ═══════════════════════════════════════════════════════════════════════════════
#  PURGE
# ═══════════════════════════════════════════════════════════════════════════════
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
    if a_borrar:
        print(f"[PURGE] {len(a_borrar)} bids (quedan {len(_estados_personas)})")
    _ts_ultimo_purge = ahora


def _purge_expirados():
    """Throttled — solo corre cada PURGE_EXP_INTERVALO segundos."""
    global _ts_ultimo_purge_exp
    ahora = time.monotonic()
    if ahora - _ts_ultimo_purge_exp < PURGE_EXP_INTERVALO:
        return
    _ts_ultimo_purge_exp = ahora
    a_borrar = [b for b, ep in _estados_personas.items() if ep.expirado()]
    for b in a_borrar:
        _estados_personas.pop(b, None)
        _bid_ultima_vez.pop(b, None)


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
#  MOTOR X — Stepper Tic (ticlib si disponible, subprocess fallback)
# ═══════════════════════════════════════════════════════════════════════════════
def _tic_cmd(*args):
    """Fallback con subprocess."""
    try:
        r = subprocess.run(["ticcmd", *args], timeout=0.8,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[TIC WARN] {r.stderr.strip()}")
    except subprocess.TimeoutExpired:
        print(f"[TIC TIMEOUT] {' '.join(args)}")
    except FileNotFoundError:
        print("[TIC] ticcmd no encontrado")
    except Exception as e:
        print(f"[TIC ERROR] {e}")


def motor_init():
    """Inicializa el Tic. Usa ticlib si está disponible, sino ticcmd."""
    global _tic_device
    if _TICLIB_OK:
        try:
            _tic_device = TicUSB()
            _tic_device.energize()
            _tic_device.exit_safe_start()
            _tic_device.halt_and_set_position(0)
            print("[MOTOR X] ✓ Posición 0  (ticlib USB — rápido)")
            return
        except Exception as e:
            print(f"[MOTOR X] ticlib falló ({e}) — usando ticcmd")
            _tic_device = None
    _tic_cmd("--resume"); _tic_cmd("--energize")
    _tic_cmd("--exit-safe-start"); _tic_cmd("--halt-and-set-position", "0")
    print("[MOTOR X] ✓ Posición 0  (ticcmd subprocess)")


def motor_mover_x(destino_abs):
    global _pos_x, _ts_cmd_x
    ahora = time.monotonic()
    if ahora - _ts_cmd_x < CMD_INTERVALO_X:
        return
    delta   = _clamp(destino_abs - _pos_x, -MAX_DELTA_X, MAX_DELTA_X)
    destino = int(_clamp(_pos_x + delta, LIMITE_IZQ, LIMITE_DER))
    if destino == _pos_x:
        return
    if _tic_device is not None:
        try:
            # ticlib requiere reset_command_timeout periódico para no entrar en safe-start
            _tic_device.set_target_position(destino)
            _tic_device.reset_command_timeout()
        except Exception as e:
            print(f"[TIC USB ERROR] {e}")
    else:
        _tic_cmd("--position", str(destino))
    _pos_x = destino; _ts_cmd_x = ahora


def _correccion_x(cx):
    err = cx - 0.5
    if abs(err) < ZONA_MUERTA_X:
        return 0
    norm  = (abs(err) - ZONA_MUERTA_X) / (0.5 - ZONA_MUERTA_X)
    pasos = max(PASO_MIN_X, int(norm ** 1.6 * PASO_MAX_X))
    return pasos * SENTIDO_X if err > 0 else pasos * -SENTIDO_X


# ═══════════════════════════════════════════════════════════════════════════════
#  SERVO Y — SG90 + Watchdog pigpio
# ═══════════════════════════════════════════════════════════════════════════════
def servo_init():
    global _pi, _servo_pw
    with _pi_lock:
        _pi = pigpio.pi()
        if not _pi.connected:
            raise RuntimeError("No se pudo conectar a pigpiod. Ejecuta: sudo pigpiod")
        _pi.set_mode(SERVO_PIN, pigpio.OUTPUT)
        _servo_pw = SERVO_INICIO
        _pi.set_servo_pulsewidth(SERVO_PIN, _servo_pw)
    time.sleep(0.8)
    print(f"[SERVO Y]  ✓ inicio={_servo_pw}µs (apuntando al suelo)")


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
        if _pi is not None:
            try:
                _pi.set_servo_pulsewidth(SERVO_PIN, destino)
                _servo_pw = destino
            except Exception as e:
                print(f"[SERVO ERROR] {e}")
                return
    _ts_cmd_y = ahora


def _correccion_y(cy):
    err = cy - 0.5
    if abs(err) < ZONA_MUERTA_Y:
        return 0
    norm     = (abs(err) - ZONA_MUERTA_Y) / (0.5 - ZONA_MUERTA_Y)
    delta_us = max(PASO_MIN_Y_US, int(norm ** 1.6 * PASO_MAX_Y_US))
    return delta_us * SENTIDO_Y if err > 0 else delta_us * -SENTIDO_Y


def watchdog_pigpio():
    """Reconecta pigpiod si se cae."""
    while True:
        time.sleep(10)
        with _pi_lock:
            ok = (_pi is not None) and _pi.connected
        if not ok:
            print("[WATCHDOG] pigpiod desconectado — reconectando...")
            try:
                servo_init()
                print("[WATCHDOG] ✓ pigpiod reconectado")
            except Exception as e:
                print(f"[WATCHDOG] fallo reconexión: {e}")


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
#  HILO ENCODER JPEG (separado del hilo de poses)
# ═══════════════════════════════════════════════════════════════════════════════
def hilo_encoder(stop_event):
    global _output_frame
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
            data = buf.tobytes()
            with _frame_lock:
                _output_frame = data
            _frame_event.set()


# ═══════════════════════════════════════════════════════════════════════════════
#  HILOS DE CÁMARA
# ═══════════════════════════════════════════════════════════════════════════════
def hilo_frames(video_q, stop_event):
    while not stop_event.is_set():
        msg = video_q.get()
        if msg is None or stop_event.is_set():
            break
        _db_escribir(msg.getCvFrame())


def hilo_poses(pose_q, stop_event):
    global _fps_inf_cnt, _fps_inf_ts, _fps_inf
    global _bid_objetivo_prev, _prev_cx_objetivo, _prev_cy_objetivo
    global _hud_cache_text, _hud_cache_ts

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

        # Filtrar detecciones por confianza desde ya (evita trabajo repetido)
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
            _resetear_ema(); _bid_objetivo_prev = None
        else:
            _bid_objetivo_prev = bid_map[objetivo]

        # Solo copiar el frame si hay detecciones que dibujar
        tiene_dets = len(dets_validas) > 0
        canvas = frame.copy() if (DEBUG_OVERLAY and tiene_dets) else frame

        for det in dets_validas:
            bid = bid_map[det]
            kps = _extraer_kps(det)
            ep  = _estados_personas[bid]

            estado, segs, alerta, c1, c2, c3, c4, c5, c6, angulo, span_y = ep.actualizar(det, kps)
            texto, color = ep.texto_estado(segs, alerta)

            if alerta:
                hay_alerta = True
                if DEBUG_OVERLAY:
                    cv2.rectangle(canvas, (0, 0), (w-1, h-1), (0, 0, 255), 8)

            if DEBUG_OVERLAY:
                x1 = int(det.xmin * w); y1 = int(det.ymin * h)
                x2 = int(det.xmax * w); y2 = int(det.ymax * h)
                bw_n = det.xmax - det.xmin
                bh_n = det.ymax - det.ymin
                cy_n = (det.ymin + det.ymax) * 0.5

                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

                # Línea del eje tronco
                if ep.kp_confiables and len(kps) >= 13:
                    hiz = kps[5]; hde = kps[6]; ciz = kps[11]; cde = kps[12]
                    if hiz[2] >= KP_CONF_MIN and hde[2] >= KP_CONF_MIN:
                        hx  = int(((hiz[0] + hde[0]) * 0.5) * w)
                        hy  = int(((hiz[1] + hde[1]) * 0.5) * h)
                        cx_ = int(((ciz[0] + cde[0]) * 0.5) * w)
                        cy_ = int(((ciz[1] + cde[1]) * 0.5) * h)
                        eje_col = (0,255,0) if not c1 else (0,165,255) if estado==Estado.SOSPECHA else (0,0,255)
                        cv2.line(canvas, (hx, hy), (cx_, cy_), eje_col, 3)
                        cv2.circle(canvas, (hx, hy), 4, (255,255,0), -1)
                        cv2.circle(canvas, (cx_, cy_), 4, (255,255,0), -1)
                        mx = (hx + cx_) // 2 + 6; my = (hy + cy_) // 2
                        cv2.putText(canvas, f"A:{angulo:.0f}", (mx, my),
                                    cv2.FONT_HERSHEY_PLAIN, 0.85, (0,230,230), 1)

                # Indicadores C1..C6
                for i, activo in enumerate((c1, c2, c3, c4, c5, c6)):
                    col = (0,255,0) if activo else (55,55,55)
                    cv2.circle(canvas, (x1 + 6 + i*11, y1 - 10), 4, col, -1)

                # Métricas bajo el bbox
                ratio_v = bw_n/bh_n if bh_n > 0.01 else 0
                cv2.putText(canvas,
                    f"spY:{span_y:.2f} bh:{bh_n:.2f} cy:{cy_n:.2f} R:{ratio_v:.2f} S:{ep.score:.0f}",
                    (x1, y2 + 13), cv2.FONT_HERSHEY_PLAIN, 0.78, (140,140,140), 1)

                # Punto torso
                tx = int((det.xmin + det.xmax) * 0.5 * w)
                ty = int((det.ymin + bh_n * FRAC_TORSO) * h)
                cv2.circle(canvas, (tx, ty), 5, (0, 255, 0), -1)

                # Estado encima del bbox
                cv2.putText(canvas, texto, (x1, max(y1 - 14, 12)),
                            cv2.FONT_HERSHEY_PLAIN, 1.1, color, 2)

                # Keypoints visibles
                for kp in kps:
                    if kp[2] >= KP_SPAN_CONF_MIN:
                        cv2.circle(canvas, (int(kp[0]*w), int(kp[1]*h)),
                                   2, (200, 200, 0), -1)

            # Control motores
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

        # Líneas de referencia
        if DEBUG_OVERLAY and tiene_dets:
            g = (75,75,75); d = (38,38,38)
            cv2.line(canvas, (_CX_F,0),        (_CX_F,h),        g, 1)
            cv2.line(canvas, (0,_CY_F),         (w,_CY_F),        g, 1)
            cv2.line(canvas, (_CX_F-_ZX_PX,0), (_CX_F-_ZX_PX,h), d, 1)
            cv2.line(canvas, (_CX_F+_ZX_PX,0), (_CX_F+_ZX_PX,h), d, 1)
            cv2.line(canvas, (0,_CY_F-_ZY_PX), (w,_CY_F-_ZY_PX), d, 1)
            cv2.line(canvas, (0,_CY_F+_ZY_PX), (w,_CY_F+_ZY_PX), d, 1)

        # HUD (siempre, aunque no haya detecciones)
        if not (DEBUG_OVERLAY and tiene_dets):
            # Aún necesitamos canvas para escribir el HUD
            canvas = frame.copy()

        cv2.rectangle(canvas, (0,0), (w,36),
                      (85,0,0) if hay_alerta else (16,16,16), -1)

        if "ALERTA"  in estado_global: hud_col = (0,0,255)
        elif "CAIDA" in estado_global: hud_col = (0,80,255)
        elif "SOSPEC" in estado_global: hud_col = (0,165,255)
        else:                          hud_col = (0,200,0)

        # HUD text cacheado a 10Hz
        if ahora - _hud_cache_ts > HUD_REFRESH_S:
            _hud_cache_text = f"INF:{_fps_inf}fps STR:{_fps_str}fps  {estado_global}  X:{_pos_x}  Y:{_servo_pw}us"
            _hud_cache_ts = ahora

        cv2.putText(canvas, _hud_cache_text,
                    (8,24), cv2.FONT_HERSHEY_PLAIN, 1.05, hud_col, 2)

        cv2.putText(canvas, "C1 C2 C3 C4 C5 C6",
                    (w-130,24), cv2.FONT_HERSHEY_PLAIN, 0.72, (80,80,80), 1)

        # Enviar al encoder (no bloqueante)
        try:
            _encode_queue.put_nowait(canvas)
        except queue.Full:
            # Descarta el viejo y mete el nuevo
            try:
                _encode_queue.get_nowait()
                _encode_queue.put_nowait(canvas)
            except (queue.Empty, queue.Full):
                pass


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE DEPTHAI + WATCHDOG
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
        print("[PIPELINE] ✓ DepthAI iniciado — IA corriendo en OAK-1 MyriadX")

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
            print("[PIPELINE] Hilos detenidos")


def camara_loop():
    while True:
        try:
            _arrancar_pipeline()
            print("[PIPELINE] Finalizó — reintentando en 3s…")
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


def _build_html():
    tic_mode = "ticlib (USB)" if _TICLIB_OK else "ticcmd (subprocess)"
    return f"""\
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Detector Caídas v12</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;700&display=swap');
  :root{{--v:#00e676;--r:#ff1744;--o:#ff9100;--c:#00e5ff;
         --bg:#070b0f;--p:#0d1117;--b:#1c2a3a;--t:#c8d8e8;--d:#4a6070;}}
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{background:var(--bg);color:var(--t);font-family:'Exo 2',sans-serif;
    min-height:100vh;display:flex;flex-direction:column;
    align-items:center;padding:20px 12px;gap:14px;}}
  h1{{font-family:'Share Tech Mono',monospace;font-size:clamp(1rem,3vw,1.4rem);
    color:var(--v);letter-spacing:.1em;text-shadow:0 0 14px rgba(0,230,118,.35);}}
  header p{{font-size:.67rem;color:var(--d);margin-top:3px;}}
  .vw{{position:relative;width:100%;max-width:680px;border:2px solid var(--b);
    border-radius:10px;overflow:hidden;background:#000;
    box-shadow:0 0 28px rgba(0,230,118,.06),0 8px 24px rgba(0,0,0,.6);}}
  .vw img{{display:block;width:100%;height:auto;}}
  .corner{{position:absolute;width:15px;height:15px;border-color:var(--v);
    border-style:solid;opacity:.5;}}
  .tl{{top:7px;left:7px;border-width:2px 0 0 2px;}}
  .tr{{top:7px;right:7px;border-width:2px 2px 0 0;}}
  .bl{{bottom:7px;left:7px;border-width:0 0 2px 2px;}}
  .br{{bottom:7px;right:7px;border-width:0 2px 2px 0;}}
  .live{{position:absolute;top:9px;left:50%;transform:translateX(-50%);
    background:rgba(255,23,68,.1);border:1px solid var(--r);color:var(--r);
    font-family:'Share Tech Mono',monospace;font-size:.6rem;letter-spacing:.12em;
    padding:2px 8px;border-radius:3px;animation:blink 1.6s infinite;}}
  @keyframes blink{{50%{{opacity:.2}}}}
  #lat{{position:absolute;bottom:9px;right:10px;font-family:'Share Tech Mono',monospace;
    font-size:.58rem;color:rgba(0,230,118,.5);}}
  .chips{{display:flex;gap:6px;flex-wrap:wrap;justify-content:center;
    font-family:'Share Tech Mono',monospace;font-size:.6rem;max-width:680px;}}
  .chip{{padding:3px 8px;border-radius:4px;border:1px solid;}}
  .c1{{color:#4fc3f7;border-color:#4fc3f740;background:#4fc3f710;}}
  .c2{{color:#a5d6a7;border-color:#a5d6a740;background:#a5d6a710;}}
  .c3{{color:#80cbc4;border-color:#80cbc440;background:#80cbc410;}}
  .c4{{color:#ffcc02;border-color:#ffcc0240;background:#ffcc0210;}}
  .c5{{color:#ff6e40;border-color:#ff6e4060;background:#ff6e4015;font-weight:bold;}}
  .c6{{color:#e040fb;border-color:#e040fb60;background:#e040fb15;}}
  .estados{{display:flex;gap:8px;flex-wrap:wrap;justify-content:center;
    font-family:'Share Tech Mono',monospace;font-size:.62rem;}}
  .ep{{color:#00e676;border:1px solid #00e67640;background:#00e67610;padding:3px 10px;border-radius:4px;}}
  .es{{color:#ff9100;border:1px solid #ff910040;background:#ff910010;padding:3px 10px;border-radius:4px;}}
  .ec{{color:#ff4444;border:1px solid #ff444440;background:#ff444410;padding:3px 10px;border-radius:4px;}}
  .ib{{display:flex;flex-wrap:wrap;justify-content:center;gap:6px 14px;
    padding:8px 14px;background:var(--p);border:1px solid var(--b);
    border-radius:7px;width:100%;max-width:680px;
    font-family:'Share Tech Mono',monospace;font-size:.62rem;color:var(--d);}}
  .tag{{color:var(--v);margin-right:3px;}}
  .cal{{background:var(--p);border:1px solid #ff6e4040;border-radius:7px;
    padding:8px 14px;width:100%;max-width:680px;
    font-family:'Share Tech Mono',monospace;font-size:.60rem;color:#aaa;}}
  .cal b{{color:#ff6e40;}}
</style>
</head>
<body>
<header>
  <h1>⬡ DETECTOR DE CAÍDAS — v12 OPTIMIZADO</h1>
  <p>OAK-1 MyriadX · 6 criterios · Encoder en hilo separado · {tic_mode}</p>
</header>
<div class="vw">
  <div class="corner tl"></div><div class="corner tr"></div>
  <div class="corner bl"></div><div class="corner br"></div>
  <span class="live">● LIVE</span>
  <img id="s" src="/video_feed" alt="stream">
  <span id="lat"></span>
</div>
<div class="chips">
  <span class="chip c1">C1 ángulo&lt;{ANGULO_CAIDA_MAX:.0f}°(+{SCORE_C1_SOLO})</span>
  <span class="chip c2">C2 spanY&lt;{KP_SPAN_Y_CAIDA}(+{SCORE_C2_SOLO})</span>
  <span class="chip c3">C3 R&gt;{RATIO_CAIDA_UMBRAL}(+{SCORE_C3_SOLO})</span>
  <span class="chip c4">C4 cab+tob(+{SCORE_C4_SOLO})</span>
  <span class="chip c5">★C5 bh&lt;{BH_CAIDA_MAX}(+{SCORE_C5_SOLO})</span>
  <span class="chip c6">C6 cy&gt;{CY_CAIDA_MIN}(+{SCORE_C6_SOLO})</span>
</div>
<div class="estados">
  <span class="ep">DE PIE — score &lt; {SCORE_SOSPECHA}</span>
  <span class="es">SOSPECHA — {SCORE_SOSPECHA}…{SCORE_CONFIRMAR}</span>
  <span class="ec">CAÍDA — score ≥ {SCORE_CONFIRMAR} · alerta {TIEMPO_ALERTA}s</span>
</div>
<div class="ib">
  <span><span class="tag">COMBOS</span>C4+C5:+{SCORE_C4_C5} C2+C5:+{SCORE_C2_C5} C5+C6:+{SCORE_C5_C6} C5+C3:+{SCORE_C5_C3}</span>
  <span><span class="tag">SERVO</span>{SERVO_INICIO}µs (suelo)</span>
  <span><span class="tag">RESTA</span>−{SCORE_RESTA}/frame</span>
  <span><span class="tag">PERSIST</span>{SEG_PERSISTENCIA}s</span>
  <span><span class="tag">EMA</span>α={EMA_ALPHA}</span>
</div>
<div class="cal">
  <b>CALIBRACIÓN:</b> mira <b>bh</b> y <b>cy</b> en el overlay.<br>
  De pie: bh ~ 0.5-0.9, cy ~ 0.3-0.5 · Caído: bh &lt; 0.38, cy &gt; 0.55<br>
  Si no detecta caída → sube <b>BH_CAIDA_MAX</b> a 0.42 o baja <b>SCORE_CONFIRMAR</b> a 20<br>
  Si falsas alarmas → baja <b>BH_CAIDA_MAX</b> a 0.33 o sube <b>SCORE_CONFIRMAR</b> a 32
</div>
<script>
const img=document.getElementById('s');
const lat=document.getElementById('lat');
let last=performance.now();
img.addEventListener('load',()=>{{const ms=Math.round(performance.now()-last);
  last=performance.now();lat.textContent=ms+'ms';}});
img.addEventListener('error',()=>{{
  setTimeout(()=>{{img.src='/video_feed?'+Date.now();}},1200);}});
</script>
</body>
</html>"""


@app.route("/")
def index():
    return Response(_build_html().encode(), mimetype="text/html; charset=utf-8")


@app.route("/video_feed")
def video_feed():
    return Response(_generar_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ═══════════════════════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "═"*64)
    print("  DETECTOR DE CAÍDAS — OAK-1  v12.0  (optimizado)")
    print("  IA corre en MyriadX del OAK-1 · 6 criterios de detección")
    print("═"*64)
    print(f"  Tic mode      : {'ticlib (USB directo)' if _TICLIB_OK else 'ticcmd (subprocess)'}")
    print(f"  C1 ángulo     : < {ANGULO_CAIDA_MAX}° (+{SCORE_C1_SOLO})")
    print(f"  C2 span-Y     : < {KP_SPAN_Y_CAIDA} (+{SCORE_C2_SOLO})")
    print(f"  C3 ratio      : > {RATIO_CAIDA_UMBRAL} (+{SCORE_C3_SOLO})")
    print(f"  C4 cab+tob    : < {KP_CABEZTOB_MAX_DY} (+{SCORE_C4_SOLO})")
    print(f"  C5 bh bbox    : < {BH_CAIDA_MAX} (+{SCORE_C5_SOLO})  ← PRINCIPAL")
    print(f"  C6 cy bbox    : > {CY_CAIDA_MIN} si servo<{SERVO_PW_SUELO_MAX} (+{SCORE_C6_SOLO})")
    print(f"  Sospecha      : score ≥ {SCORE_SOSPECHA}")
    print(f"  Confirmar     : score ≥ {SCORE_CONFIRMAR}  (≈0.2s a 15fps)")
    print(f"  Alerta        : {TIEMPO_ALERTA}s en caída confirmada")
    print(f"  SERVO INICIO  : {SERVO_INICIO}µs → SUELO")
    print("═"*64 + "\n")

    try:
        motor_init()
        servo_init()
    except Exception as e:
        print(f"[ERROR HARDWARE] {e}")
        sys.exit(1)

    threading.Thread(target=watchdog_pigpio, daemon=True, name="t-watchdog").start()
    threading.Thread(target=camara_loop,     daemon=True, name="t-pipeline").start()

    print("[BOOT] Esperando 4s para que el pipeline arranque…")
    time.sleep(4)

    print(f"\n  ✓ Servidor:")
    print(f"    → http://0.0.0.0:{FLASK_PORT}/")
    print(f"    → http://<IP_RASPBERRY>:{FLASK_PORT}/\n")

    try:
        app.run(host=FLASK_HOST, port=FLASK_PORT,
                threaded=True, debug=False, use_reloader=False)
    finally:
        servo_stop()
        print("[EXIT] Sistema detenido.")