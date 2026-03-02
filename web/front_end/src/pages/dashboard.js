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
      <button class="flex size-9 items-center justify-center rounded-lg bg-card-dark text-slate-300 border border-white/5 hover:opacity-90" data-noti aria-label="Notifications">
        <span class="material-symbols-outlined text-[20px]">notifications</span>
      </button>
      <button class="size-9 rounded-full bg-primary/20 border border-primary/30 flex items-center justify-center overflow-hidden hover:opacity-90" data-user aria-label="Account">
        <span class="material-symbols-outlined">person</span>
      </button>
    </div>
  `;
  page.appendChild(top);

  // Content
  const content = document.createElement('div');
  content.className = 'p-4 space-y-4 pb-28';
  page.appendChild(content);

  const camera = CameraStreamer({
    onMetrics: (payload) => {
      metrics.setMetrics(payload.metrics);
      // realtime alerts
      const r = payload.metrics.strain_risk || 0;
      if (r > 0.75) toast('Nguy cơ mỏi mắt cao → nghỉ 20–30s và chớp mắt', 'warn');
      if ((payload.metrics.distance_cm || 0) < 45) toast('Bạn đang ngồi quá gần màn hình', 'warn');
    }
  });
  content.appendChild(camera);

  const metrics = MetricsGrid();
  content.appendChild(metrics);

  // Quick actions
  const actions = document.createElement('div');
  actions.className = 'bg-card-dark border border-white/5 p-4 rounded-2xl flex items-center justify-between gap-3';
  actions.innerHTML = `
    <div class="flex items-center gap-2">
      <span class="material-symbols-outlined text-primary">download</span>
      <div>
        <div class="text-sm font-bold">Xuất dữ liệu phiên</div>
        <div class="text-[10px] text-slate-400">Tải CSV để bạn feed vào pipeline AI cá nhân</div>
      </div>
    </div>
    <button class="bg-primary text-black font-bold text-xs px-3 py-2 rounded-xl hover:opacity-90" data-export>DOWNLOAD CSV</button>
  `;
  actions.querySelector('[data-export]').addEventListener('click', async ()=> {
    try {
      const token = localStorage.getItem('vg_token');
      if (!token) return toast('Cần đăng nhập trước', 'warn');
      // direct download with auth token
      const url = (import.meta.env.VITE_API_BASE || 'http://localhost:8000') + '/api/export.csv';
      const res = await fetch(url, { headers: { Authorization: `Bearer ${token}` }});
      if (!res.ok) throw new Error('Không có dữ liệu');
      const blob = await res.blob();
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'session_metrics.csv';
      a.click();
      URL.revokeObjectURL(a.href);
      toast('Đã tải CSV', 'ok');
    } catch(e) {
      toast('Tải CSV lỗi: ' + (e?.message||''), 'err');
    }
  });
  content.appendChild(actions);

  const chat = ChatBox();
  content.appendChild(chat);

  // Bottom nav with "sensor" hover + active state
  const nav = document.createElement('div');
  nav.className = 'fixed bottom-0 left-0 right-0 bg-background-dark/90 backdrop-blur-2xl border-t border-white/5 px-8 py-2 pb-8 flex justify-between items-center z-50';
  nav.innerHTML = `
    ${navItem('Guard','dashboard',true)}
    ${navItem('Stats','analytics')}
    ${navItemCenter('Track','add')}
    ${navItem('Coaching','medical_services')}
    ${navItem('Me','person')}
  `;
  page.appendChild(nav);

  function bindIconSensors(root) {
    root.querySelectorAll('[data-sense]').forEach(btn => {
      btn.addEventListener('mouseenter', ()=> btn.classList.add('ring-2','ring-primary/40'));
      btn.addEventListener('mouseleave', ()=> btn.classList.remove('ring-2','ring-primary/40'));
      btn.addEventListener('mousedown', ()=> btn.classList.add('icon-press'));
      btn.addEventListener('mouseup', ()=> btn.classList.remove('icon-press'));
    });
  }
  bindIconSensors(top);
  bindIconSensors(nav);

  // Notifications click
  top.querySelector('[data-noti]').addEventListener('click', ()=> {
    toast('Thông báo: Hệ thống đang chạy realtime', 'info');
  });

  // Account modal: login/register + logout
  top.querySelector('[data-user]').addEventListener('click', async () => {
    const box = document.createElement('div');
    box.className = 'space-y-3';

    // Tạo overlay trước để dùng được trong onAuthed
    const overlay = Modal({ title:'Tài khoản', contentEl: box, onClose: () => overlay.remove() });

    const me = await safeMe();
    if (me) {
      box.innerHTML = `
        <div class="text-sm">Đang đăng nhập: <span class="font-bold">${me.email}</span></div>
        <button class="w-full bg-white/5 border border-white/10 text-white font-semibold rounded-xl py-2 text-sm" data-logout>Đăng xuất</button>
      `;
      box.querySelector('[data-logout]').addEventListener('click', () => {
        setToken('');
        toast('Đã đăng xuất', 'info');
        overlay.remove();
      });
    } else {
      function mountAuth(mode = 'login') {
        box.innerHTML = '';

        const panel = AuthPanel({
          mode,
          onAuthed: () => overlay.remove()
        });

        panel.addEventListener('auth:switch', (e) => {
          mountAuth(e.detail.mode);
        });

        box.appendChild(panel);
      }
      mountAuth('login');
    }

    document.body.appendChild(overlay);
  });

  // Zoom controls: Ctrl + / Ctrl - / Ctrl 0 (browser-level + CSS scale fallback)
  let uiScale = 1;
  window.addEventListener('keydown', (e)=> {
    if (!e.ctrlKey) return;
    if (e.key === '+' || e.key === '=') { uiScale = Math.min(1.3, uiScale + 0.05); applyScale(); }
    if (e.key === '-') { uiScale = Math.max(0.85, uiScale - 0.05); applyScale(); }
    if (e.key === '0') { uiScale = 1; applyScale(); }
  });
  function applyScale() {
    page.style.transformOrigin = 'top center';
    page.style.transform = `scale(${uiScale})`;
    toast(`UI scale: ${Math.round(uiScale*100)}%`, 'info');
  }

  async function safeMe() {
    try { return await api('/api/me'); } catch { return null; }
  }

  return page;
}

function navItem(label, icon, active=false) {
  return `
    <button class="flex flex-col items-center gap-1 ${active?'text-primary':'text-slate-500'}" data-sense>
      <span class="material-symbols-outlined ${active?'font-variation-fill':''}">${icon}</span>
      <span class="text-[9px] font-bold uppercase tracking-tighter">${label}</span>
    </button>
  `;
}
function navItemCenter(label, icon) {
  return `
    <button class="flex flex-col items-center gap-1 text-slate-500" data-sense>
      <div class="size-10 -mt-6 bg-primary rounded-full flex items-center justify-center text-black shadow-lg shadow-primary/20 border-4 border-background-dark">
        <span class="material-symbols-outlined">${icon}</span>
      </div>
      <span class="text-[9px] font-bold uppercase tracking-tighter">${label}</span>
    </button>
  `;
}
