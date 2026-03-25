import { toast } from '../lib/toast.js';
import { api, setToken } from '../lib/api.js';
import { CameraStreamer } from '../components/camera.js';
import { MetricsGrid } from '../components/metrics.js';
import { ChatBox } from '../components/chat.js';
import { Modal } from '../components/modal.js';
import { AuthPanel } from '../components/auth.js';

export function Dashboard() {
  const page = document.createElement('div');
  page.className = 'relative flex min-h-screen w-full flex-col bg-background-dark';

  // Top bar
  const top = document.createElement('div');
  top.className = 'flex items-center bg-background-dark/80 backdrop-blur-md p-4 pb-3 justify-between sticky top-0 z-50 border-b border-white/5';
  top.innerHTML = `
    <div class="flex items-center gap-3">
      <div class="text-primary flex size-9 shrink-0 items-center justify-center bg-primary/10 rounded-lg" data-icon="app">
        <span class="material-symbols-outlined font-light">visibility</span>
      </div>
      <div class="flex flex-col">
        <h2 class="text-white text-base font-bold leading-none tracking-tight">Vision Guard</h2>
        <div class="flex items-center gap-1.5 mt-1">
          <span class="flex h-1.5 w-1.5 rounded-full bg-primary animate-pulse"></span>
          <p class="text-[9px] uppercase tracking-[0.05em] font-semibold text-slate-400">Live AI Analysis</p>
        </div>
      </div>
    </div>

    <div class="flex items-center gap-2">
      <button class="flex size-9 items-center justify-center rounded-lg bg-card-dark text-slate-300 border border-white/5 hover:opacity-90" data-noti>
        <span class="material-symbols-outlined text-[20px]">notifications</span>
      </button>
      <button class="size-9 rounded-full bg-primary/20 border border-primary/30 flex items-center justify-center overflow-hidden hover:opacity-90" data-user>
        <span class="material-symbols-outlined">person</span>
      </button>
    </div>
  `;
  page.appendChild(top);

  const content = document.createElement('div');
  content.className = 'p-4 space-y-4 pb-28';
  page.appendChild(content);

  let currentSymptomText = '';

  const camera = CameraStreamer({
    getText: () => currentSymptomText,
    onMetrics: (payload) => {
      metrics.setMetrics(payload.metrics);
      const r = payload.metrics.strain_risk || 0;
      if (r > 0.75) toast('Nguy cơ mỏi mắt cao → nghỉ 20–30s', 'warn');
      if ((payload.metrics.distance_cm || 0) < 45) toast('Bạn đang ngồi quá gần màn hình', 'warn');
    }
  });
  content.appendChild(camera);

  const metrics = MetricsGrid();
  content.appendChild(metrics);

  const actions = document.createElement('div');
  actions.className = 'bg-card-dark border border-white/5 p-4 rounded-2xl flex items-center justify-between gap-3';
  actions.innerHTML = `
    <div class="flex items-center gap-2">
      <span class="material-symbols-outlined text-primary">download</span>
      <div>
        <div class="text-sm font-bold">Xuất dữ liệu phiên</div>
        <div class="text-[10px] text-slate-400">Tải CSV</div>
      </div>
    </div>
    <button class="bg-primary text-black font-bold text-xs px-3 py-2 rounded-xl" data-export>DOWNLOAD CSV</button>
  `;

  actions.querySelector('[data-export]').addEventListener('click', async ()=> {
    try {
      const url = (import.meta.env.VITE_API_BASE || 'http://localhost:8000') + '/api/export.csv';

      const res = await fetch(url);
      if (!res.ok) throw new Error('Không có dữ liệu');

      const blob = await res.blob();
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'session_metrics.csv';
      a.click();
      URL.revokeObjectURL(a.href);

      toast('Đã tải CSV', 'ok');
    } catch(e) {
      toast('Tải CSV lỗi', 'err');
    }
  });

  content.appendChild(actions);

  const chat = ChatBox({
    onUserMessage: (text) => {
      currentSymptomText = text;
    }
  });
  content.appendChild(chat);

  top.querySelector('[data-noti]').addEventListener('click', ()=> {
    toast('Hệ thống đang chạy realtime', 'info');
  });

  top.querySelector('[data-user]').addEventListener('click', ()=> {
    toast('Demo mode: không cần đăng nhập', 'info');
  });

  return page;
}