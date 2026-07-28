import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
import cv2
import threading
import time
from flask import Flask, Response

app = Flask(__name__)

output_frame  = None
frame_lock    = threading.Lock()
estado_actual = Iniciando...

TIEMPO_ALERTA = 10
historial     = {}
tiempo_caido  = {}

# Solo los keypoints que realmente usamos para deteccion
KEYPOINTS_IMPORTANTES = [5, 6, 11, 12, 15, 16]

# Solo las lineas del esqueleto entre keypoints importantes
SKELETON_REDUCIDO = [
    (5,6),   # hombro a hombro
    (5,11),  # hombro izq a cadera izq
    (6,12),  # hombro der a cadera der
    (11,12), # cadera a cadera
    (11,15), # cadera izq a tobillo izq
    (12,16), # cadera der a tobillo der
]

def bbox_id(det)
    cx = round((det.xmin + det.xmax)  2, 1)
    cy = round((det.ymin + det.ymax)  2, 1)
    return f{cx}_{cy}

def es_caida(det, kps)
    bw = det.xmax - det.xmin
    bh = det.ymax - det.ymin
    if bh  0.01
        return False
    caida_bbox = (bw  bh)  1.2
    caida_kps  = False
    if len(kps) = 13
        conf_h = (kps[5][2] + kps[6][2])  2
        conf_c = (kps[11][2] + kps[12][2])  2
        if conf_h  0.15 and conf_c  0.15
            dx = abs((kps[5][0]+kps[6][0])2 - (kps[11][0]+kps[12][0])2)
            dy = abs((kps[5][1]+kps[6][1])2 - (kps[11][1]+kps[12][1])2)
            if dy  0.01
                caida_kps = (dx  dy)  1.1
    return caida_bbox or caida_kps

def suavizar(bid, caido)
    if bid not in historial
        historial[bid] = []
    historial[bid].append(caido)
    if len(historial[bid])  8
        historial[bid].pop(0)
    return sum(historial[bid]) = 5

def estado_final(bid, confirmado)
    ahora = time.time()
    if confirmado
        if bid not in tiempo_caido
            tiempo_caido[bid] = ahora
        segs = int(ahora - tiempo_caido[bid])
        if segs = TIEMPO_ALERTA
            return fALERTA {segs}s EN PISO, (0,0,255), True
        return fCAIDA ({segs}s), (0,80,255), False
    else
        tiempo_caido.pop(bid, None)
        historial.pop(bid, None)
        return DE PIE, (0,220,0), False

def camara_loop()
    global output_frame, estado_actual
    fps_count = 0
    fps_time  = time.time()
    fps       = 0

    with dai.Pipeline() as pipeline
        cam    = pipeline.create(dai.node.Camera).build()
        nn     = pipeline.create(ParsingNeuralNetwork).build(
                     cam, luxonisyolov8-nano-pose-estimationcoco-512x288)
        parser = nn.getParser()

        video_q = cam.requestOutput(
            (512, 288), type=dai.ImgFrame.Type.BGR888p
        ).createOutputQueue(maxSize=2, blocking=False)
        pose_q = parser.out.createOutputQueue(maxSize=2, blocking=False)

        pipeline.start()
        print(Pipeline iniciado)
        print(Abre http132.138.1.625000)

        while pipeline.isRunning()
            frame_msg = video_q.get()
            pose_msg  = pose_q.get()
            if frame_msg is None or pose_msg is None
                continue

            frame = frame_msg.getCvFrame()
            h, w  = frame.shape[2]

            fps_count += 1
            if time.time() - fps_time = 1.0
                fps = fps_count
                fps_count = 0
                fps_time  = time.time()

            estado_global = Sin personas
            hay_alerta    = False

            for det in pose_msg.detections
                if det.confidence  0.35
                    continue

                x1 = int(det.xmin  w); y1 = int(det.ymin  h)
                x2 = int(det.xmax  w); y2 = int(det.ymax  h)
                bid = bbox_id(det)

                kps = []
                if hasattr(det, 'keypoints')
                    for kp in det.keypoints
                        kps.append([kp.x, kp.y,
                            kp.confidence if hasattr(kp,'confidence') else 1.0])

                caido      = es_caida(det, kps)
                confirmado = suavizar(bid, caido)
                estado, color, alerta = estado_final(bid, confirmado)
                estado_global = estado

                if alerta
                    hay_alerta = True
                    cv2.rectangle(frame, (0,0), (w-1,h-1), (0,0,255), 10)

                # Bounding box
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, estado,
                    (x1, max(y1-8,12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # Solo keypoints importantes
                pts = {}
                for i in KEYPOINTS_IMPORTANTES
                    if i  len(kps) and kps[i][2]  0.2
                        px = int(kps[i][0]  w)
                        py = int(kps[i][1]  h)
                        pts[i] = (px, py)
                        cv2.circle(frame, (px,py), 5, (255,220,0), -1)

                # Esqueleto reducido
                for a, b in SKELETON_REDUCIDO
                    if a in pts and b in pts
                        cv2.line(frame, pts[a], pts[b], (0,220,255), 2)

            estado_actual = estado_global

            # Barra superior
            cv2.rectangle(frame, (0,0), (w,38),
                (100,0,0) if hay_alerta else (20,20,20), -1)
            txt_col = (0,0,255)   if ALERTA in estado_global else 
                      (0,100,255) if CAIDA  in estado_global else 
                      (0,220,0)
            cv2.putText(frame, fFPS{fps}  {estado_global},
                (8,26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, txt_col, 2)

            ret, buf = cv2.imencode('.jpg', frame,
                [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret
                with frame_lock
                    output_frame = buf.tobytes()

HTML = !DOCTYPE html
html
head
meta charset=utf-8
titleDetector de Caidastitle
style
   { margin0; padding0; box-sizingborder-box; }
  body { background#0d0d0d; color#eee;
         font-familyArial,sans-serif;
         displayflex; flex-directioncolumn;
         align-itemscenter; padding20px; }
  h1   { color#00e676; margin-bottom12px; font-size1.4em; }
  #vc  { border3px solid #00e676; border-radius10px;
         overflowhidden; max-width640px; width100%; }
  img  { displayblock; width100%; }
  #inf { margin-top12px; padding8px 20px; background#1a1a1a;
         border-radius8px; font-size0.82em; color#888;
         border1px solid #333; text-aligncenter; }
style
head
body
  h1Sistema de Deteccion de Caidas OAK-1h1
  div id=vcimg src=videodiv
  div id=inf
    IA YOLOv8-pose en Myriad X &nbsp;&nbsp;
    Alerta tras TIEMPO_ALERTA segundos &nbsp;&nbsp;
    Pi 3B+ 132.138.1.62
  div
body
html.replace(TIEMPO_ALERTA, str(TIEMPO_ALERTA))

@app.route('')
def index()
    return Response(HTML.encode(), mimetype='texthtml')

@app.route('video')
def video()
    def gen()
        while True
            with frame_lock
                if output_frame is None
                    time.sleep(0.05)
                    continue
                f = output_frame
            yield (b'--framernContent-Type imagejpegrnrn' + f + b'rn')
    return Response(gen(),
        mimetype='multipartx-mixed-replace; boundary=frame')

if __name__ == '__main__'
    threading.Thread(target=camara_loop, daemon=True).start()
    time.sleep(4)
    print(Servidor http132.138.1.625000)
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
