import { api } from '../lib/api.js';
import { toast } from '../lib/toast.js';

export function ChatBox() {
  const shell = document.createElement('div');
  shell.className = 'flex flex-col bg-card-dark/50 rounded-2xl overflow-hidden border border-white/5 h-[340px] shadow-lg';

  shell.innerHTML = `
    <div class="p-3 border-b border-white/5 flex justify-between items-center bg-card-dark/80">
      <div class="flex items-center gap-2">
        <div class="size-8 rounded-full bg-primary/20 border border-primary/40 flex items-center justify-center text-primary">
          <span class="material-symbols-outlined text-[18px]">smart_toy</span>
        </div>
        <div>
          <h3 class="text-xs font-bold leading-none">AI Eye Coach</h3>
          <span class="text-[9px] text-green-400 font-medium" data-status>Online</span>
        </div>
      </div>
      <button class="text-slate-400 hover:text-white" data-more>
        <span class="material-symbols-outlined text-[18px]">more_vert</span>
      </button>
    </div>

    <div class="flex-1 p-4 space-y-4 overflow-y-auto custom-scrollbar" data-feed></div>

    <div class="p-3 bg-card-dark border-t border-white/5">
      <div class="flex items-center gap-2 bg-black/40 rounded-xl px-3 py-1.5 border border-white/10">
        <input class="bg-transparent border-none focus:ring-0 text-xs flex-1 text-white placeholder:text-slate-500" placeholder="Hỏi coach bất kỳ..." type="text" data-input />
        <button class="text-primary p-1 hover:opacity-90" data-send aria-label="Send">
          <span class="material-symbols-outlined">send</span>
        </button>
      </div>
    </div>
  `;

  const feed = shell.querySelector('[data-feed]');
  const input = shell.querySelector('[data-input]');

  function addBubble(text, who='bot') {
    const wrap = document.createElement('div');
    wrap.className = `flex flex-col gap-1 ${who==='me' ? 'items-end' : 'items-start'}`;
    const bubble = document.createElement('div');
    bubble.className = who==='me'
      ? 'bg-primary/20 p-3 rounded-2xl rounded-tr-none max-w-[85%] border border-primary/20'
      : 'bg-white/5 p-3 rounded-2xl rounded-tl-none max-w-[85%] border border-white/5';
    bubble.innerHTML = `<p class="text-[13px] leading-relaxed ${who==='me'?'text-white':'text-slate-200'}"></p>`;
    bubble.querySelector('p').textContent = text;
    const t = document.createElement('span');
    t.className = `text-[8px] text-slate-500 ${who==='me'?'mr-1':'ml-1'}`;
    t.textContent = new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
    wrap.appendChild(bubble);
    wrap.appendChild(t);
    feed.appendChild(wrap);
    feed.scrollTop = feed.scrollHeight;
  }

  addBubble('Phiên làm việc bắt đầu. Mình có thể nhắc 20-20-20, tư thế, khoảng cách và chớp mắt.', 'bot');

  async function send() {
    const msg = input.value.trim();
    if (!msg) return;
    input.value = '';
    addBubble(msg, 'me');

    try {
      const out = await api('/api/chat', { method:'POST', body:{ message: msg } });
      addBubble(out.reply || '(no reply)', 'bot');
      if (out.safety_note) toast(out.safety_note, 'info');
    } catch (e) {
      toast('Chat lỗi: ' + (e?.message||''), 'err');
      addBubble('Mình đang gặp lỗi kết nối. Thử lại sau nhé.', 'bot');
    }
  }

  shell.querySelector('[data-send]').addEventListener('click', send);
  input.addEventListener('keydown', (e)=> {
    if (e.key === 'Enter') send();
  });

  return shell;
}
