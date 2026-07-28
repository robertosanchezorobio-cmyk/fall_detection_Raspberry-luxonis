"""
╔══════════════════════════════════════════════════════════════════════╗
║          SISTEMA DE DETECCIÓN DE CAÍDAS — OAK-1 + YOLOv8-pose       ║
║  Eje X → Stepper (Tic)  |  Eje Y → Servo SG90 (pigpio)              ║
║  Streaming Flask en http://<IP>:5000/                                ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import threading
import time
import subprocess

import cv2
import depthai as dai
import pigpio
from depthai_nodes.node import ParsingNeuralNetwork
from flask import Flask, Response

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GENERAL
# ═══════════════════════════════════════════════════════════════════════════════
TIEMPO_ALERTA   = 10        # segundos en el suelo antes de disparar alerta
JPEG_CALIDAD    = 75        # calidad del stream MJPEG (0-100)
FLASK_PORT      = 5000
FLASK_HOST      = "0.0.0.0"

# ── Motor X — stepper controlado por Tic ──────────────────────────────────────
ZONA_MUERTA_X   = 0.08     # fracción del ancho (0-1) ignorada en el centro
PASO_MAX_X      = 60       # pasos máximos por corrección
PASO_MIN_X      = 8        # pasos mínimos por corrección
LIMITE_IZQ      = -2000    # límite físico izquierdo (steps)
LIMITE_DER      =  2000    # límite físico derecho (steps)
CMD_INTERVALO_X = 0.06     # segundos entre comandos al motor X
SENTIDO_X       = -1       # invertir con 1 si el motor va al lado opuesto

# ── Servo Y — SG90 via pigpio ────────────────────────────────────────────────
SERVO_PIN       = 18       # GPIO (BCM) del servo
SERVO_MIN       = 1100     # pulso mínimo en µs  (límite abajo)
SERVO_MAX       = 1780     # pulso máximo en µs  (límite arriba)
SERVO_CENTER    = 1440     # pulso central en µs
ZONA_MUERTA_Y   = 0.06     # fracción del alto ignorada en el centro
PASO_MAX_Y_US   = 12       # µs máximos por corrección
PASO_MIN_Y_US   = 2        # µs mínimos por corrección
CMD_INTERVALO_Y = 0.04     # segundos entre comandos al servo
SENTIDO_Y       = -1       # invertir con 1 si el servo va al lado opuesto

# ── Filtro camisa blanca ──────────────────────────────────────────────────────
BLANCO_S_MAX    = 55       # saturación HSV máxima para "blanco"
BLANCO_V_MIN    = 165      # valor HSV mínimo para "blanco"
BLANCO_COBERT   = 0.20     # fracción mínima de píxeles blancos en el torso


# ═══════════════════════════════════════════════════════════════════════════════
#  ESTADO GLOBAL  (compartido entre hilos)
# ═══════════════════════════════════════════════════════════════════════════════
app               = Flask(__name__)

# Stream Flask
_output_frame     = None
_frame_lock       = threading.Lock()

# Frame crudo de la cámara
_ultimo_frame_raw = None
_frame_raw_lock   = threading.Lock()

# Historial de detección de caídas  {bbox_id: [bool, ...]}
_historial_caida  = {}
_tiempo_caido     = {}     # {bbox_id: timestamp primera caída}

# Estado motores
pos_actual_x      = 0
servo_pw_actual   = SERVO_CENTER
_ultimo_cmd_x     = 0.0
_ultimo_cmd_y     = 0.0
_pi               = None   # instancia pigpio


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════════
def limitar(v, vmin, vmax):
    """Clamp de valor entre vmin y vmax."""
    return max(vmin, min(vmax, v))


# ═══════════════════════════════════════════════════════════════════════════════
#  MOTOR X — Stepper (Tic USB)
# ═══════════════════════════════════════════════════════════════════════════════
def _tic(*args):
    """Envía un comando al controlador Tic via ticcmd."""
    try:
        subprocess.run(
            ["ticcmd", *args],
            check=True, timeout=1, capture_output=True
        )
    except Exception as e:
        print(f"[MOTOR X ERROR] {e}")


def motor_init():
    """Inicializa el stepper y lo lleva a posición 0."""
    _tic("--resume")
    _tic("--energize")
    _tic("--exit-safe-start")
    _tic("--halt-and-set-position", "0")
    print("[MOTOR X] ✓ Inicializado en posición 0")


def motor_mover_x(destino_abs: int):
    """Mueve el stepper a una posición absoluta (con rate-limit y clamp)."""
    global pos_actual_x, _ultimo_cmd_x
    ahora = time.time()
    if ahora - _ultimo_cmd_x < CMD_INTERVALO_X:
        return
    destino = int(limitar(destino_abs, LIMITE_IZQ, LIMITE_DER))
    if destino == pos_actual_x:
        return
    _tic("--position", str(destino))
    pos_actual_x  = destino
    _ultimo_cmd_x = ahora


def calcular_correccion_x(cx_norm: float) -> int:
    """
    Devuelve los pasos a sumar/restar en X según el centro normalizado (0-1).
    Retorna 0 si está dentro de la zona muerta.
    """
    error = cx_norm - 0.5
    if abs(error) < ZONA_MUERTA_X:
        return 0
    norm  = (abs(error) - ZONA_MUERTA_X) / (0.5 - ZONA_MUERTA_X)
    pasos = max(PASO_MIN_X, int(norm * PASO_MAX_X))
    return pasos * SENTIDO_X if error > 0 else pasos * -SENTIDO_X


# ═══════════════════════════════════════════════════════════════════════════════
#  SERVO Y — SG90 via pigpio
# ═══════════════════════════════════════════════════════════════════════════════
def servo_init():
    """Inicializa el servo en la posición central."""
    global _pi, servo_pw_actual
    _pi = pigpio.pi()
    if not _pi.connected:
        raise RuntimeError(
            "No se pudo conectar a pigpiod. Ejecuta: sudo pigpiod"
        )
    _pi.set_mode(SERVO_PIN, pigpio.OUTPUT)
    servo_pw_actual = SERVO_CENTER
    _pi.set_servo_pulsewidth(SERVO_PIN, servo_pw_actual)
    print(f"[SERVO Y]  ✓ Inicializado — centro = {servo_pw_actual} µs")


def servo_stop():
    """Detiene el servo y cierra la conexión pigpio."""
    global _pi
    if _pi:
        _pi.set_servo_pulsewidth(SERVO_PIN, 0)
        _pi.stop()
        _pi = None
    print("[SERVO Y]  ✓ Detenido")


def servo_mover_y(destino_pw: int):
    """Mueve el servo a un ancho de pulso destino (con rate-limit y clamp)."""
    global servo_pw_actual, _ultimo_cmd_y
    ahora = time.time()
    if ahora - _ultimo_cmd_y < CMD_INTERVALO_Y:
        return
    destino = int(limitar(destino_pw, SERVO_MIN, SERVO_MAX))
    if destino == servo_pw_actual:
        return
    _pi.set_servo_pulsewidth(SERVO_PIN, destino)
    servo_pw_actual = destino
    _ultimo_cmd_y   = ahora


def calcular_correccion_y(cy_norm: float) -> int:
    """
    Devuelve los µs a sumar/restar en Y según el centro normalizado (0-1).
    Retorna 0 si está dentro de la zona muerta.
    """
    error = cy_norm - 0.5
    if abs(error) < ZONA_MUERTA_Y:
        return 0
    norm     = (abs(error) - ZONA_MUERTA_Y) / (0.5 - ZONA_MUERTA_Y)
    delta_us = max(PASO_MIN_Y_US, int(norm * PASO_MAX_Y_US))
    return delta_us * SENTIDO_Y if error > 0 else delta_us * -SENTIDO_Y


# ═══════════════════════════════════════════════════════════════════════════════
#  DETECCIÓN DE CAÍDA
# ═══════════════════════════════════════════════════════════════════════════════
def tiene_camisa_blanca(frame: "np.ndarray", det) -> bool:
    """
    Devuelve True si la zona del torso de la detección contiene
    suficientes píxeles de color blanco (según BLANCO_*).
    """
    h, w = frame.shape[:2]
    x1 = max(0, int(det.xmin * w))
    x2 = min(w, int(det.xmax * w))
    y1 = max(0, int(det.ymin * h))
    # Solo analiza el 55 % superior del bounding box (torso)
    y2 = min(h, int((det.ymin + (det.ymax - det.ymin) * 0.55) * h))

    if x2 - x1 < 8 or y2 - y1 < 8:
        return False

    roi  = cv2.resize(frame[y1:y2, x1:x2], (24, 24),
                      interpolation=cv2.INTER_NEAREST)
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] < BLANCO_S_MAX) & (hsv[:, :, 2] > BLANCO_V_MIN)
    return mask.sum() / 576 >= BLANCO_COBERT


def _bbox_id(det) -> str:
    """ID reproducible de un bounding box (centro redondeado)."""
    cx = round((det.xmin + det.xmax) / 2, 2)
    cy = round((det.ymin + det.ymax) / 2, 2)
    return f"{cx}_{cy}"


def es_caida(det, keypoints: list) -> bool:
    """
    Heurística de caída:
      1. Bounding box más ancho que alto (ratio > 1.2).
      2. Vector hombros→caderas más horizontal que vertical (ratio dx/dy > 1.1).
    """
    bw = det.xmax - det.xmin
    bh = det.ymax - det.ymin

    if bh < 0.01:
        return False

    # Criterio 1 — bounding box horizontal
    if (bw / bh) > 1.2:
        return True

    # Criterio 2 — keypoints disponibles (COCO: 5-6 = hombros, 11-12 = caderas)
    if len(keypoints) >= 13:
        conf_hombros = (keypoints[5][2] + keypoints[6][2]) / 2
        conf_caderas = (keypoints[11][2] + keypoints[12][2]) / 2
        if conf_hombros > 0.15 and conf_caderas > 0.15:
            cx_hombros = (keypoints[5][0] + keypoints[6][0]) / 2
            cy_hombros = (keypoints[5][1] + keypoints[6][1]) / 2
            cx_caderas = (keypoints[11][0] + keypoints[12][0]) / 2
            cy_caderas = (keypoints[11][1] + keypoints[12][1]) / 2
            dx = abs(cx_hombros - cx_caderas)
            dy = abs(cy_hombros - cy_caderas)
            if dy > 0.01 and (dx / dy) > 1.1:
                return True

    return False


def suavizar_deteccion(bid: str, caido: bool) -> bool:
    """
    Ventana deslizante de 8 frames: confirma caída si ≥ 5 son positivos.
    Evita falsos positivos por frames ruidosos.
    """
    buf = _historial_caida.setdefault(bid, [])
    buf.append(caido)
    if len(buf) > 8:
        buf.pop(0)
    return sum(buf) >= 5


def estado_final(bid: str, confirmado: bool):
    """
    Devuelve (texto_estado, color_BGR, es_alerta_critica).
    Gestiona el temporizador de tiempo en el suelo.
    """
    ahora = time.time()

    if confirmado:
        _tiempo_caido.setdefault(bid, ahora)
        segundos = int(ahora - _tiempo_caido[bid])

        if segundos >= TIEMPO_ALERTA:
            return f"⚠ ALERTA {segundos}s EN PISO", (0, 0, 255), True
        return f"CAIDA ({segundos}s)", (0, 80, 255), False

    # De pie → limpiar historial
    _tiempo_caido.pop(bid, None)
    _historial_caida.pop(bid, None)
    return "DE PIE", (0, 220, 0), False


def mejor_objetivo(detecciones, frame):
    """
    Devuelve la detección prioritaria: camisa blanca + mayor área × confianza.
    """
    mejor, mejor_score = None, -1.0
    for det in detecciones:
        if det.confidence < 0.35 or not tiene_camisa_blanca(frame, det):
            continue
        score = (det.xmax - det.xmin) * (det.ymax - det.ymin) * det.confidence
        if score > mejor_score:
            mejor_score = score
            mejor = det
    return mejor


# ═══════════════════════════════════════════════════════════════════════════════
#  HILOS DE CÁMARA
# ═══════════════════════════════════════════════════════════════════════════════
def hilo_frames(video_q):
    """
    Hilo 1 — Lee frames de la cámara continuamente y los almacena.
    No se bloquea esperando poses; siempre tiene el frame más reciente.
    """
    global _ultimo_frame_raw
    while True:
        msg = video_q.get()
        if msg is None:
            continue
        with _frame_raw_lock:
            _ultimo_frame_raw = msg.getCvFrame()


def hilo_poses(pose_q):
    """
    Hilo 2 — Consume resultados de pose, anota el frame y publica en Flask.
    También envía correcciones a los motores.
    """
    global _output_frame

    fps_count  = 0
    fps_tiempo = time.time()
    fps        = 0

    while True:
        pose_msg = pose_q.get()
        if pose_msg is None:
            continue

        # Tomar el frame más reciente sin esperar al siguiente ciclo de pose
        with _frame_raw_lock:
            if _ultimo_frame_raw is None:
                continue
            frame = _ultimo_frame_raw.copy()

        h, w = frame.shape[:2]

        # ── Contador FPS ──────────────────────────────────────────────────────
        fps_count += 1
        if time.time() - fps_tiempo >= 1.0:
            fps        = fps_count
            fps_count  = 0
            fps_tiempo = time.time()

        # ── Procesar detecciones ──────────────────────────────────────────────
        estado_global = "Sin persona con camisa blanca"
        hay_alerta    = False
        objetivo      = mejor_objetivo(pose_msg.detections, frame)

        for det in pose_msg.detections:
            if det.confidence < 0.35 or not tiene_camisa_blanca(frame, det):
                continue

            x1 = int(det.xmin * w);  y1 = int(det.ymin * h)
            x2 = int(det.xmax * w);  y2 = int(det.ymax * h)
            bid = _bbox_id(det)

            # Extraer keypoints
            kps = []
            if hasattr(det, "keypoints"):
                for kp in det.keypoints:
                    kps.append([
                        kp.x, kp.y,
                        kp.confidence if hasattr(kp, "confidence") else 1.0
                    ])

            caido      = es_caida(det, kps)
            confirmado = suavizar_deteccion(bid, caido)
            estado, color, alerta = estado_final(bid, confirmado)

            if alerta:
                hay_alerta = True
                # Borde rojo pulsante en el frame completo
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)

            # Dibujar bounding box y etiqueta
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, estado, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            # Punto central
            cx_px = int((det.xmin + det.xmax) / 2 * w)
            cy_px = int((det.ymin + det.ymax) / 2 * h)
            cv2.circle(frame, (cx_px, cy_px), 5, (255, 255, 0), -1)

            # ── Control de motores para el objetivo principal ─────────────────
            if objetivo is not None and det is objetivo:
                estado_global = estado
                cx_n = (det.xmin + det.xmax) / 2
                cy_n = (det.ymin + det.ymax) / 2

                corr_x = calcular_correccion_x(cx_n)
                if corr_x:
                    motor_mover_x(pos_actual_x + corr_x)

                corr_y = calcular_correccion_y(cy_n)
                if corr_y:
                    servo_mover_y(servo_pw_actual + corr_y)

        # ── Líneas de referencia (zona muerta) ───────────────────────────────
        cx_f  = w // 2;          cy_f  = h // 2
        zx_px = int(w * ZONA_MUERTA_X)
        zy_px = int(h * ZONA_MUERTA_Y)
        # Cruz central
        cv2.line(frame, (cx_f, 0),        (cx_f, h),        (100, 100, 100), 1)
        cv2.line(frame, (0,    cy_f),      (w,    cy_f),      (100, 100, 100), 1)
        # Zona muerta X
        cv2.line(frame, (cx_f - zx_px, 0), (cx_f - zx_px, h), (50, 50, 50), 1)
        cv2.line(frame, (cx_f + zx_px, 0), (cx_f + zx_px, h), (50, 50, 50), 1)
        # Zona muerta Y
        cv2.line(frame, (0, cy_f - zy_px), (w, cy_f - zy_px), (50, 50, 50), 1)
        cv2.line(frame, (0, cy_f + zy_px), (w, cy_f + zy_px), (50, 50, 50), 1)

        # ── HUD — barra superior ──────────────────────────────────────────────
        color_hud = (100, 0, 0) if hay_alerta else (20, 20, 20)
        cv2.rectangle(frame, (0, 0), (w, 38), color_hud, -1)

        if "ALERTA" in estado_global:
            txt_col = (0, 0, 255)
        elif "CAIDA" in estado_global:
            txt_col = (0, 100, 255)
        else:
            txt_col = (0, 220, 0)

        hud_text = (
            f"FPS:{fps}  {estado_global}  "
            f"X:{pos_actual_x}  Y:{servo_pw_actual}us"
        )
        cv2.putText(frame, hud_text, (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, txt_col, 2)

        # ── Codificar y publicar frame anotado ────────────────────────────────
        ok, buf = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_CALIDAD]
        )
        if ok:
            with _frame_lock:
                _output_frame = buf.tobytes()


def camara_loop():
    """
    Arranca el pipeline DepthAI con dos colas independientes:
      - video_q → hilo_frames  (máx 4 frames, no bloqueante)
      - pose_q  → hilo_poses   (máx 2 resultados, no bloqueante)
    """
    with dai.Pipeline() as pipeline:
        cam    = pipeline.create(dai.node.Camera).build()
        nn     = pipeline.create(ParsingNeuralNetwork).build(
            cam,
            "luxonis/yolov8-nano-pose-estimation:coco-512x288"
        )
        parser = nn.getParser()

        video_q = cam.requestOutput(
            (512, 288), type=dai.ImgFrame.Type.BGR888p
        ).createOutputQueue(maxSize=4, blocking=False)

        pose_q = parser.out.createOutputQueue(maxSize=2, blocking=False)

        pipeline.start()
        print("[OK] Pipeline DepthAI iniciado")

        threading.Thread(
            target=hilo_frames, args=(video_q,), daemon=True, name="t-frames"
        ).start()
        threading.Thread(
            target=hilo_poses, args=(pose_q,), daemon=True, name="t-poses"
        ).start()

        while pipeline.isRunning():
            time.sleep(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  FLASK — Streaming MJPEG
# ═══════════════════════════════════════════════════════════════════════════════

# HTML embebido (se sirve sin archivos externos)
_HTML = f"""\
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Detector de Caídas — OAK-1</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;700&display=swap');

  :root {{
    --verde:  #00e676;
    --rojo:   #ff1744;
    --fondo:  #070b0f;
    --panel:  #0d1117;
    --borde:  #1c2a3a;
    --texto:  #c8d8e8;
    --dim:    #4a6070;
  }}

  * {{ margin:0; padding:0; box-sizing:border-box; }}

  body {{
    background: var(--fondo);
    color: var(--texto);
    font-family: 'Exo 2', sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 24px 16px;
    gap: 20px;
  }}

  /* ─── Cabecera ─── */
  header {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
  }}
  header h1 {{
    font-family: 'Share Tech Mono', monospace;
    font-size: clamp(1.1rem, 3vw, 1.6rem);
    color: var(--verde);
    letter-spacing: 0.12em;
    text-shadow: 0 0 18px rgba(0,230,118,.45);
  }}
  header p {{
    font-size: 0.72rem;
    color: var(--dim);
    letter-spacing: 0.06em;
  }}

  /* ─── Video ─── */
  .video-wrapper {{
    position: relative;
    width: 100%;
    max-width: 700px;
    border: 2px solid var(--borde);
    border-radius: 12px;
    overflow: hidden;
    background: #000;
    box-shadow: 0 0 40px rgba(0,230,118,.08), 0 8px 32px rgba(0,0,0,.6);
  }}
  .video-wrapper img {{
    display: block;
    width: 100%;
    height: auto;
  }}
  /* Esquinas decorativas */
  .corner {{
    position: absolute;
    width: 18px; height: 18px;
    border-color: var(--verde);
    border-style: solid;
    opacity: .6;
  }}
  .corner.tl {{ top:8px;  left:8px;  border-width:2px 0 0 2px; }}
  .corner.tr {{ top:8px;  right:8px; border-width:2px 2px 0 0; }}
  .corner.bl {{ bottom:8px; left:8px;  border-width:0 0 2px 2px; }}
  .corner.br {{ bottom:8px; right:8px; border-width:0 2px 2px 0; }}

  /* Indicador LIVE */
  .live-badge {{
    position: absolute;
    top: 12px; left: 50%; transform: translateX(-50%);
    background: rgba(255,23,68,.15);
    border: 1px solid var(--rojo);
    color: var(--rojo);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: .12em;
    padding: 2px 10px;
    border-radius: 4px;
    animation: blink 1.6s infinite;
    pointer-events: none;
  }}
  @keyframes blink {{ 50% {{ opacity:.3 }} }}

  /* ─── Info bar ─── */
  .info-bar {{
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 10px 20px;
    padding: 10px 20px;
    background: var(--panel);
    border: 1px solid var(--borde);
    border-radius: 8px;
    width: 100%;
    max-width: 700px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: var(--dim);
  }}
  .info-bar span {{ white-space: nowrap; }}
  .info-bar .tag {{ color: var(--verde); margin-right: 4px; }}
</style>
</head>
<body>
  <header>
    <h1>⬡ SISTEMA DE DETECCIÓN DE CAÍDAS</h1>
    <p>OAK-1 · YOLOv8-pose · Myriad X · Flask MJPEG</p>
  </header>

  <div class="video-wrapper">
    <div class="corner tl"></div>
    <div class="corner tr"></div>
    <div class="corner bl"></div>
    <div class="corner br"></div>
    <span class="live-badge">● LIVE</span>
    <img src="/video_feed" alt="Stream de cámara" onerror="this.src='/video_feed'">
  </div>

  <div class="info-bar">
    <span><span class="tag">MODELO</span>yolov8-nano-pose</span>
    <span><span class="tag">ALERTA</span>tras {TIEMPO_ALERTA}s en piso</span>
    <span><span class="tag">FILTRO</span>camisa blanca</span>
    <span><span class="tag">EJE X</span>stepper Tic</span>
    <span><span class="tag">EJE Y</span>servo SG90</span>
    <span><span class="tag">URL</span>http://&lt;IP&gt;:{FLASK_PORT}/</span>
  </div>
</body>
</html>
"""


def _generar_mjpeg():
    """
    Generador MJPEG para /video_feed.
    Espera activamente hasta que haya un frame disponible.
    Si no hay frame nuevo, reenvía el último para mantener el stream vivo
    y evitar que el navegador lo cierre por timeout.
    """
    ultimo_enviado = None

    while True:
        with _frame_lock:
            frame_actual = _output_frame

        if frame_actual is None:
            # Aún no hay frame — esperar sin acaparar CPU
            time.sleep(0.03)
            continue

        if frame_actual is not ultimo_enviado:
            ultimo_enviado = frame_actual
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_actual +
                b"\r\n"
            )
        else:
            # No hay frame nuevo; dormir un poco antes de reintentar
            time.sleep(0.02)


@app.route("/")
def index():
    return Response(_HTML.encode("utf-8"), mimetype="text/html; charset=utf-8")


@app.route("/video_feed")
def video_feed():
    """Endpoint MJPEG — ábrelo directamente en el navegador si quieres."""
    return Response(
        _generar_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  DETECTOR DE CAÍDAS — OAK-1")
    print("=" * 60)

    try:
        motor_init()
        servo_init()

        # Arrancar el pipeline de cámara en hilo separado
        threading.Thread(
            target=camara_loop, daemon=True, name="t-pipeline"
        ).start()

        # Esperar a que el pipeline esté listo antes de abrir Flask
        print("[BOOT] Esperando 4 s para que el pipeline arranque...")
        time.sleep(4)

        print(f"\n  ✓ Servidor listo:")
        print(f"    → http://{FLASK_HOST}:{FLASK_PORT}/          (página principal)")
        print(f"    → http://<IP_RASPBERRY>:{FLASK_PORT}/         (desde otro equipo)")
        print(f"    → http://<IP_RASPBERRY>:{FLASK_PORT}/video_feed (solo stream)")
        print()

        # use_reloader=False es OBLIGATORIO cuando Flask se lanza desde un hilo
        app.run(
            host=FLASK_HOST,
            port=FLASK_PORT,
            threaded=True,
            debug=False,
            use_reloader=False   # ← CRÍTICO: evita doble-proceso que rompe pigpio/DepthAI
        )

    finally:
        servo_stop()
        print("[EXIT] Sistema detenido.")