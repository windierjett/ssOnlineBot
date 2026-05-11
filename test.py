import websocket, traceback
url = "wss://kg2.ss911.cn:9102/"
try:
    ws = websocket.create_connection(url, timeout=8)
    print("ws_connected")
    ws.close()
except Exception:
    traceback.print_exc()