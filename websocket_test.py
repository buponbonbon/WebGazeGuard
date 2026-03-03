
import asyncio
import json
import base64
import time
import cv2
import websockets

# =============================
# CONFIG
# =============================
WS_URL = "ws://127.0.0.1:8000/api/ws/stream"

# paste access token
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIiwiaWF0IjoxNzcyNTQ5MjE1LCJleHAiOjE3NzMxNTQwMTV9.2ceTiYtumycMPorlTBtKM09kjcBoVsKugq9Xupj2Z74"

# HELPER
def frame_to_b64(frame):
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


async def main():
    async with websockets.connect(WS_URL, max_size=8 * 1024 * 1024) as ws:

        # 1️⃣ AUTH FIRST
        await ws.send(json.dumps({
            "type": "auth",
            "token": TOKEN
        }))

        print("Connected & Auth sent")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam")
            return

        last_print = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            jpeg_b64 = frame_to_b64(frame)
            if jpeg_b64 is None:
                continue

            await ws.send(json.dumps({
                "type": "frame",
                "ts_ms": int(time.time() * 1000),
                "jpeg_b64": jpeg_b64
            }))

            try:
                response = await ws.recv()
            except Exception as e:
                print("WebSocket error:", e)
                break

            data = json.loads(response)

            if data.get("type") == "metrics":
                m = data["payload"]["metrics"]

                now = time.time()
                if now - last_print > 1:
                    last_print = now
                    print(
                        "Risk:", round(m.get("strain_risk", 0), 3),
                        "| Blink:", round(m.get("blink_rate_per_min", 0), 2),
                        "| Yaw:", round(m.get("head_pose_yaw_deg", 0), 2),
                        "| Pitch:", round(m.get("head_pose_pitch_deg", 0), 2),
                        "| Dist:", round(m.get("distance_cm", 0), 2)
                    )

            await asyncio.sleep(0.05)

        cap.release()


if __name__ == "__main__":
    asyncio.run(main())
