export function Modal({ title, contentEl, onClose }) {
  const overlay = document.createElement('div');
  overlay.className = 'fixed inset-0 z-[9000] bg-black/60 backdrop-blur-sm flex items-center justify-center p-4';
  overlay.addEventListener('click', (e)=> {
    if (e.target === overlay) onClose();
  });

  const card = document.createElement('div');
  card.className = 'w-full max-w-md bg-card-dark border border-white/10 rounded-2xl overflow-hidden shadow-2xl';

  const header = document.createElement('div');
  header.className = 'flex items-center justify-between p-3 border-b border-white/10 bg-card-dark/80';
  header.innerHTML = `
    <div class="text-sm font-bold">${title}</div>
    <button class="text-slate-300 hover:text-white" aria-label="Close">
      <span class="material-symbols-outlined text-[20px]">close</span>
    </button>
  `;
  header.querySelector('button').addEventListener('click', onClose);

  const body = document.createElement('div');
  body.className = 'p-4';
  body.appendChild(contentEl);

  card.appendChild(header);
  card.appendChild(body);
  overlay.appendChild(card);

  return overlay;
}
