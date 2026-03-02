import { api, setToken } from '../lib/api.js';
import { toast } from '../lib/toast.js';

export function AuthPanel({ mode='login', onAuthed }) {
  const wrap = document.createElement('div');
  wrap.className = 'space-y-3';

  const title = document.createElement('div');
  title.className = 'text-sm font-bold';
  title.textContent = mode === 'login' ? 'Đăng nhập' : 'Tạo tài khoản';
  wrap.appendChild(title);

  const form = document.createElement('form');
  form.className = 'space-y-2';

  form.innerHTML = `
    <label class="block">
      <div class="text-[10px] text-slate-400 font-semibold uppercase tracking-wider mb-1">Email</div>
      <input required type="email" class="w-full bg-black/30 border border-white/10 rounded-xl px-3 py-2 text-sm" placeholder="you@email.com" />
    </label>
    <label class="block">
      <div class="text-[10px] text-slate-400 font-semibold uppercase tracking-wider mb-1">Mật khẩu</div>
      <input required type="password" minlength="6" class="w-full bg-black/30 border border-white/10 rounded-xl px-3 py-2 text-sm" placeholder="Tối thiểu 6 ký tự" />
    </label>
    <button type="submit" class="w-full bg-primary text-black font-bold rounded-xl py-2 text-sm">Xác nhận</button>
    <button type="button" class="w-full bg-white/5 border border-white/10 text-white font-semibold rounded-xl py-2 text-sm" data-switch>
      ${mode === 'login' ? 'Chưa có tài khoản? Đăng ký' : 'Đã có tài khoản? Đăng nhập'}
    </button>
  `;

  const [emailEl, passEl] = form.querySelectorAll('input');
  form.addEventListener('submit', async (e)=> {
    e.preventDefault();
    try {
      const path = mode === 'login' ? '/api/auth/login' : '/api/auth/register';
      const out = await api(path, { method:'POST', auth:false, body:{ email: emailEl.value, password: passEl.value } });
      setToken(out.access_token);
      toast('Đăng nhập thành công', 'ok');
      onAuthed?.();
    } catch (err) {
      toast('Không đăng nhập được: ' + (err?.message || ''), 'err');
    }
  });

  form.querySelector('[data-switch]').addEventListener('click', ()=> {
    const ev = new CustomEvent('auth:switch', { detail: { mode: mode === 'login' ? 'register' : 'login' } });
    wrap.dispatchEvent(ev);
  });

  wrap.appendChild(form);
  return wrap;
}
