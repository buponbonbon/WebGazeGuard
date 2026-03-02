import { wsUrl, getToken } from '../lib/api.js';
import { toast } from '../lib/toast.js';

/**
 * Camera + WebSocket streamer
 * - Uses requestAnimationFrame loop but sends frames at a throttled rate (default 8 FPS) for performance.
 * - Downscales frames to 640px width to reduce CPU/network.
 */
export function CameraStreamer({ onMetrics }) {
  const root = document.createElement('div');
  root.className = 'relative flex flex-col items-center justify-center bg-black aspect-video rounded-2xl overflow-hidden border border-primary/20 shadow-2xl shadow-primary/5';

  const video = document.createElement('video');
  video.className = 'absolute inset-0 w-full h-full object-cover opacity-90';
  video.autoplay = true;
  video.playsInline = true;
  video.muted = true;

  const overlay = document.createElement('div');
  overlay.className = 'absolute inset-0 pointer-events-none';
  overlay.innerHTML = `
    <svg class="w-full h-full text-primary/40" preserveAspectRatio="none" viewBox="0 0 100 100">
      <path d="M10 0 L10 100 M20 0 L20 100 M30 0 L30 100 M40 0 L40 100 M50 0 L50 100 M60 0 L60 100 M70 0 L70 100 M80 0 L80 100 M90 0 L90 100" opacity="0.2" stroke="currentColor" stroke-width="0.05"></path>
      <path d="M0 10 L100 10 M0 20 L100 20 M0 30 L100 30 M0 40 L100 40 M0 50 L100 50 M0 60 L100 60 M0 70 L100 70 M0 80 L100 80 M0 90 L100 90" opacity="0.2" stroke="currentColor" stroke-width="0.05"></path>
      <circle cx="35" cy="40" fill="currentColor" opacity="0.8" r="1.5"></circle>
      <circle cx="65" cy="40" fill="currentColor" opacity="0.8" r="1.5"></circle>
    </svg>
  `;

  const hud = document.createElement('div');
  hud.className = 'absolute top-3 left-3 flex flex-col gap-1';
  hud.innerHTML = `
    <div class="bg-black/60 backdrop-blur-md rounded px-2 py-0.5 border border-white/10">
      <span class="text-[9px] font-bold text-primary tracking-widest" data-mesh>MESH_ACTIVE: --</span>
    </div>
    <div class="bg-black/60 backdrop-blur-md rounded px-2 py-0.5 border border-white/10">
      <span class="text-[9px] font-bold text-white tracking-widest" data-lat>LATENCY: --</span>
    </div>
  `;

  const controls = document.createElement('div');
  controls.className = 'absolute bottom-3 right-3 flex gap-2';
  controls.innerHTML = `
    <button class="bg-primary/90 text-black font-bold text-[10px] px-3 py-1.5 rounded-lg shadow-lg backdrop-blur-sm" data-calib>CALIBRATE</button>
    <button class="bg-white/10 text-white font-bold text-[10px] px-3 py-1.5 rounded-lg border border-white/10" data-stop>STOP</button>
  `;

  root.appendChild(video);
  root.appendChild(overlay);
  root.appendChild(hud);
  root.appendChild(controls);

  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d', { alpha:false, willReadFrequently:false });

  let ws;
  let stream;
  let running = false;
  let lastSent = 0;
  const sendEveryMs = 125; // ~8 FPS

  function setHud({ mesh='TRUE', latency='--' } = {}) {
    hud.querySelector('[data-mesh]').textContent = `MESH_ACTIVE: ${mesh}`;
    hud.querySelector('[data-lat]').textContent = `LATENCY: ${latency}`;
  }

  async function start() {
    if (running) return;
    const token = getToken();
    if (!token) {
      toast('Bạn cần đăng nhập để dùng camera', 'warn');
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode:'user' }, audio:false });
      video.srcObject = stream;
      running = true;

      ws = new WebSocket(wsUrl('/ws/stream'));
      ws.onopen = () => {
        ws.send(JSON.stringify({ type:'auth', token }));
        toast('Kết nối realtime thành công', 'ok');
      };
      ws.onmessage = (ev)=> {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'metrics') onMetrics?.(msg.payload);
        } catch {}
      };
      ws.onerror = ()=> toast('WebSocket lỗi', 'err');
      ws.onclose = ()=> { /* silent */ };

      const loop = () => {
        if (!running) return;
        const t0 = performance.now();

        const vw = video.videoWidth || 640;
        const vh = video.videoHeight || 360;
        if (vw && vh) {
          const targetW = Math.min(640, vw);
          const targetH = Math.round(vh * (targetW / vw));
          canvas.width = targetW;
          canvas.height = targetH;
          ctx.drawImage(video, 0, 0, targetW, targetH);

          const now = performance.now();
          if (now - lastSent >= sendEveryMs && ws && ws.readyState === 1) {
            lastSent = now;
            const jpeg = canvas.toDataURL('image/jpeg', 0.6).split(',')[1]; // base64
            ws.send(JSON.stringify({ type:'frame', ts_ms: Date.now(), jpeg_b64: jpeg }));
          }
        }

        const t1 = performance.now();
        setHud({ mesh:'TRUE', latency: `${Math.round(t1 - t0)}ms` });
        requestAnimationFrame(loop);
      };
      requestAnimationFrame(loop);
    } catch (e) {
      toast('Không mở được camera: hãy cho phép quyền Camera', 'err');
      running = false;
    }
  }

  function stop() {
    running = false;
    if (ws) { try { ws.close(); } catch {} }
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
    toast('Đã dừng camera', 'info');
  }

  controls.querySelector('[data-stop]').addEventListener('click', stop);
  controls.querySelector('[data-calib]').addEventListener('click', () => {
    toast('Hiệu chuẩn: (demo) đặt mặt cách màn hình ~60cm và nhìn thẳng 2s', 'info');
  });

  // Autostart when component mounts
  setTimeout(start, 250);

  root.__start = start;
  root.__stop = stop;
  return root;
}
