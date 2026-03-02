const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export function getToken() {
  return localStorage.getItem('vg_token') || '';
}

export function setToken(t) {
  if (t) localStorage.setItem('vg_token', t);
  else localStorage.removeItem('vg_token');
}

export async function api(path, { method='GET', body, auth=true } = {}) {
  const headers = { 'Content-Type': 'application/json' };
  if (auth) {
    const token = getToken();
    if (token) headers['Authorization'] = `Bearer ${token}`;
  }
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const txt = await res.text().catch(()=> '');
    throw new Error(txt || `HTTP ${res.status}`);
  }
  const ct = res.headers.get('content-type') || '';
  return ct.includes('application/json') ? res.json() : res.text();
}

export function wsUrl(path='/ws/stream') {
  // Convert http://host:8000 -> ws://host:8000
  const base = API_BASE.replace(/^http/, 'ws');
  return `${base}${path}`;
}
