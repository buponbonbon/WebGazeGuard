import './styles/global.css';
import { Dashboard } from './pages/dashboard.js';
import { toast } from './lib/toast.js';

// Mount app
const app = document.getElementById('app');
app.appendChild(Dashboard());

// Basic capability checks
if (!navigator.mediaDevices?.getUserMedia) {
  toast('Trình duyệt không hỗ trợ Camera API. Hãy dùng Chrome/Edge mới.', 'err');
}
