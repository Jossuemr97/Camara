from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import cv2
import pytesseract
import re
from imutils.video import VideoStream
import imutils
import threading

app = Flask(__name__)
CORS(app)

# Configuración de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

url_video_vlc = "rtsp://servicesmartsa@gmail.com:Placam-2024@192.168.100.53:554/stream1"
vs = VideoStream(url_video_vlc).start()

patron = r'[A-Z]{3}-[0-9]{4}'

plateCode = ""
plateCodeLock = threading.Lock()

def detect_plates_worker():
    global plateCode

    while True:
        try:
            frame = vs.read()
            frame = imutils.resize(frame, width=400)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            img_binario = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]

            contornos, _ = cv2.findContours(img_binario, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contorno in contornos:
                area = cv2.contourArea(contorno)

                if area > 10000:
                    x, y, w, h = cv2.boundingRect(contorno)

                    placa = frame[y:y+h, x:x+w]

                    placa_reconocida = pytesseract.image_to_string(placa, config='-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ-0123456789')

                    if re.match(patron, placa_reconocida):
                        with plateCodeLock:
                            plateCode = placa_reconocida[:8]
                            print(placa_reconocida[:8])

        except Exception as e:
            print(f"Error en hilo de procesamiento: {e}")

def detect_plates():
    while True:
        with plateCodeLock:
            current_plate_code = plateCode

        frame = vs.read()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_plates(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/plate_code', methods=['GET'])
def get_plate_code():
    with plateCodeLock:
        return jsonify({'plate_code': plateCode})

if __name__ == '__main__':
    processing_thread = threading.Thread(target=detect_plates_worker)
    processing_thread.daemon = True
    processing_thread.start()

    # Inicia la aplicación Flask
    app.run(debug=True)