/** Metric cards + risk bar */
export function MetricsGrid() {
  const grid = document.createElement('div');
  grid.className = 'grid grid-cols-2 gap-3';

  const card = (title, icon, unit='') => {
    const el = document.createElement('div');
    el.className = 'bg-card-dark border border-white/5 p-3 rounded-2xl';
    el.innerHTML = `
      <div class="flex justify-between items-start mb-1">
        <p class="text-slate-400 text-[10px] font-semibold uppercase tracking-wider">${title}</p>
        <span class="material-symbols-outlined text-primary text-[18px]">${icon}</span>
      </div>
      <div class="flex items-baseline gap-1">
        <span class="text-xl font-bold" data-value>--</span>
        <span class="text-[10px] text-slate-500">${unit}</span>
      </div>
      <div class="mt-1 text-[9px] text-slate-400" data-status>--</div>
    `;
    return el;
  };

  const cBlink = card('Blink Rate', 'visibility', '/min');
  const cEar = card('EAR Value', 'data_usage', '');
  const cPose = card('Head Pose', 'architecture', 'deg');
  const cDist = card('Distance', 'straighten', 'cm');

  grid.append(cBlink, cEar, cPose, cDist);

  const risk = document.createElement('div');
  risk.className = 'bg-card-dark border border-white/5 p-4 rounded-2xl mt-3';
  risk.innerHTML = `
    <div class="flex justify-between items-center mb-3">
      <div class="flex items-center gap-2">
        <span class="material-symbols-outlined text-yellow-500 text-[20px]">warning</span>
        <h4 class="text-sm font-bold uppercase tracking-wide">Strain Risk</h4>
      </div>
      <span class="text-xs font-bold text-yellow-500" data-risk-text>--</span>
    </div>
    <div class="relative h-2.5 w-full bg-slate-800 rounded-full overflow-hidden shadow-inner">
      <div class="h-full bg-yellow-500 rounded-full" style="width: 0%" data-risk-bar></div>
    </div>
    <div class="flex justify-between mt-2">
      <span class="text-[9px] font-bold text-slate-500">RESTED</span>
      <span class="text-[9px] font-bold text-slate-500">CRITICAL</span>
    </div>
    <div class="mt-2 text-[10px] text-slate-400">Posture flag: <span class="text-slate-200 font-semibold" data-posture>--</span></div>
  `;

  function setMetrics(m) {
    if (!m) return;
    cBlink.querySelector('[data-value]').textContent = Math.round(m.blink_rate_per_min).toString();
    cBlink.querySelector('[data-status]').textContent = m.blink_rate_per_min < 10 ? 'Low blink → hãy chớp mắt chủ động' : 'Normal';

    cEar.querySelector('[data-value]').textContent = m.ear.toFixed(2);
    cEar.querySelector('[data-status]').textContent = m.ear < 0.18 ? 'Blinking/Closed' : 'Open';

    const yaw = m.head_pose_yaw_deg ?? 0;
    const pitch = m.head_pose_pitch_deg ?? 0;
    cPose.querySelector('[data-value]').textContent = `${Math.round(yaw)}° / ${Math.round(pitch)}°`;
    cPose.querySelector('[data-status]').textContent = (Math.abs(yaw)>12 || Math.abs(pitch)>10) ? 'Slight Tilt' : 'OK';

    cDist.querySelector('[data-value]').textContent = Math.round(m.distance_cm).toString();
    cDist.querySelector('[data-status]').textContent = m.distance_cm < 50 ? 'Too close' : 'Optimal';

    const riskVal = Math.round((m.strain_risk ?? 0) * 100);
    risk.querySelector('[data-risk-text]').textContent = `${riskVal}%`;
    risk.querySelector('[data-risk-bar]').style.width = `${riskVal}%`;
    risk.querySelector('[data-posture]').textContent = m.posture_flag || '--';
  }

  grid.setMetrics = setMetrics;
  grid.appendChild(risk);
  return grid;
}
