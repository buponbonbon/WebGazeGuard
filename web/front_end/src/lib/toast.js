/** Tiny toast system (no dependencies). */
let container;

export function ensureToasts() {
  if (container) return container;
  container = document.createElement('div');
  container.className = 'fixed top-16 right-4 z-[9999] flex flex-col gap-2';
  document.body.appendChild(container);
  return container;
}

export function toast(message, type='info') {
  ensureToasts();
  const el = document.createElement('div');
  const colors = {
    info: 'bg-card-dark border-white/10 text-slate-200',
    ok: 'bg-green-900/40 border-green-500/30 text-green-100',
    warn: 'bg-yellow-900/40 border-yellow-500/30 text-yellow-100',
    err: 'bg-red-900/40 border-red-500/30 text-red-100'
  };
  el.className = `max-w-[320px] border rounded-xl px-3 py-2 shadow-lg backdrop-blur-md ${colors[type] || colors.info}`;
  el.innerHTML = `<div class="text-xs font-semibold">${message}</div>`;
  container.appendChild(el);
  setTimeout(()=> { el.style.opacity='0'; el.style.transform='translateY(-4px)'; el.style.transition='200ms'; }, 2800);
  setTimeout(()=> el.remove(), 3200);
}
