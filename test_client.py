import time
import socketio
import cv2
import base64

sio = socketio.Client()
#logger=True, engineio_logger=True
# start_timer = None


# def send_ping():
#     global start_timer
#     start_timer = time.time()
#     sio.emit('ping_from_client', namespaces = "/camera")


# @sio.event
# def connect():
#     print('connected to server')
#     send_ping()


# @sio.event
# def pong_from_server():
#     global start_timer
#     latency = time.time() - start_timer
#     print('latency is {0:.2f} ms'.format(latency * 1000))
#     sio.sleep(1)
#     if sio.connected:
#         send_ping()


# def on_message(data):
#     return data
# @sio.on('Camera')
# def message_handler(msg):
#     print('Received message: ', msg)
#
# if __name__ == '__main__':
sio.connect('http://10.10.102.225:1000', namespaces = "/camera")
frame = cv2.imread('image/camera_id_14.jpg')
frame = cv2.resize(frame, (9, 6))
retval, buffer = cv2.imencode('.jpg', frame)
jpg_as_text = base64.b64encode(buffer)
while True:
    sio.emit('Image', jpg_as_text, namespace='/camera')
    time.sleep(5)
