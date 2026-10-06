let currentMode = 'proposed';
let pollInterval = null;

function selectMode(mode) {
  currentMode = mode;
  document.getElementById('btnModeProposed').classList.toggle('active', mode === 'proposed');
  document.getElementById('btnModeBaseline').classList.toggle('active', mode === 'baseline');
}

async function startGeneration() {
  const prompt = document.getElementById('promptInput').value.trim();
  if (!prompt) {
    alert('Please enter a text prompt.');
    return;
  }

  const keyframes = parseInt(document.getElementById('keyframesSelect').value);
  const resolution = parseInt(document.getElementById('resSelect').value);
  const fps = parseInt(document.getElementById('fpsSelect').value);

  const btnGen = document.getElementById('btnGenerate');
  const statusBox = document.getElementById('statusBox');
  const statusStage = document.getElementById('statusStage');
  const statusPct = document.getElementById('statusPct');
  const progressBar = document.getElementById('progressBar');
  
  btnGen.disabled = true;
  statusBox.style.display = 'block';
  statusStage.innerText = 'INITIALIZING PIPELINE...';
  statusPct.innerText = '5%';
  progressBar.style.width = '5%';

  // Hide comparison view during single generation
  document.getElementById('compViewBox').style.display = 'none';

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, mode: currentMode, keyframes, resolution, fps })
    });

    const data = await res.json();
    if (data.error) {
      showError(data.error);
      return;
    }

    const jobId = data.job_id;
    pollStatus(jobId);

  } catch (err) {
    showError(err.message);
  }
}

function pollStatus(jobId) {
  if (pollInterval) clearInterval(pollInterval);

  pollInterval = setInterval(async () => {
    try {
      const res = await fetch(`/api/status/${jobId}`);
      const data = await res.json();

      if (data.error) {
        clearInterval(pollInterval);
        showError(data.error);
        return;
      }

      document.getElementById('statusStage').innerText = data.status;
      document.getElementById('statusPct').innerText = `${data.progress}%`;
      document.getElementById('progressBar').style.width = `${data.progress}%`;

      if (data.status === 'COMPLETE' && data.result) {
        clearInterval(pollInterval);
        renderResult(data.result);
      } else if (data.status === 'ERROR') {
        clearInterval(pollInterval);
        showError(data.error || 'Pipeline execution failed.');
      }
    } catch (err) {
      clearInterval(pollInterval);
      showError(err.message);
    }
  }, 350);
}

function renderResult(result) {
  document.getElementById('btnGenerate').disabled = false;
  document.getElementById('statusBox').style.display = 'none';

  // Render Video
  const videoPlayer = document.getElementById('videoPlayer');
  const videoPlaceholder = document.getElementById('videoPlaceholder');
  const videoActions = document.getElementById('videoActions');
  const btnDownload = document.getElementById('btnDownload');

  videoPlaceholder.style.display = 'none';
  videoPlayer.style.display = 'block';
  videoPlayer.src = result.video_url;
  videoPlayer.play();

  videoActions.style.display = 'flex';
  btnDownload.href = `${result.video_url}?download=1`;

  // Render Keyframes
  const keyframeStrip = document.getElementById('keyframeStrip');
  const keyframeGrid = document.getElementById('keyframeGrid');
  keyframeGrid.innerHTML = '';
  keyframeStrip.style.display = 'block';

  if (result.keyframes_b64) {
    result.keyframes_b64.forEach((b64, idx) => {
      const card = document.createElement('div');
      card.className = 'keyframe-card';
      card.innerHTML = `
        <img src="${b64}" alt="Keyframe K${idx + 1}" />
        <div class="keyframe-tag">K${idx + 1}</div>
      `;
      keyframeGrid.appendChild(card);
    });
  }

  // Render Metrics
  const metricsBox = document.getElementById('metricsBox');
  const metricsTableBody = document.getElementById('metricsTableBody');
  metricsBox.style.display = 'block';
  metricsTableBody.innerHTML = '';

  const m = result.metrics;
  const metricsData = [
    { label: 'Mode', val: result.mode_label, desc: 'Pipeline execution mode' },
    { label: 'Total Frames', val: `${result.frame_count} frames (${result.duration}s)`, desc: 'Keyframe expansion (T → 4T-3)' },
    { label: 'Total Latency', val: `${m.generation_time_sec} s`, desc: 'End-to-end pipeline execution time' },
    { label: 'Interpolation Latency', val: `${m.interp_latency_sec} s`, desc: 'Latent frame interpolation time' },
    { label: 'Temporal Flicker Score', val: m.flicker.toFixed(4), desc: 'Lower score = smoother, flicker-free video' },
    { label: 'CLIP Alignment Score', val: m.clipsim.toFixed(4), desc: 'Cosine similarity between prompt & video frames' },
    { label: 'Reconstruction PSNR', val: `${m.psnr.toFixed(2)} dB`, desc: 'Peak Signal-to-Noise Ratio' },
    { label: 'Structural SSIM', val: m.ssim.toFixed(4), desc: 'Structural Similarity Index' }
  ];

  metricsData.forEach(item => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${item.label}</td>
      <td class="value-highlight">${item.val}</td>
      <td style="color: var(--text-muted);">${item.desc}</td>
    `;
    metricsTableBody.appendChild(tr);
  });
}

async function runComparison() {
  const prompt = document.getElementById('promptInput').value.trim();
  if (!prompt) {
    alert('Please enter a text prompt.');
    return;
  }

  const keyframes = parseInt(document.getElementById('keyframesSelect').value);
  const resolution = parseInt(document.getElementById('resSelect').value);
  const fps = parseInt(document.getElementById('fpsSelect').value);

  const btnGen = document.getElementById('btnGenerate');
  const btnComp = document.getElementById('btnCompare');
  const statusBox = document.getElementById('statusBox');
  
  btnGen.disabled = true;
  btnComp.disabled = true;
  statusBox.style.display = 'block';
  document.getElementById('statusStage').innerText = 'RUNNING BENCHMARK COMPARISON...';
  document.getElementById('statusPct').innerText = '50%';
  document.getElementById('progressBar').style.width = '50%';

  try {
    const res = await fetch('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, keyframes, resolution, fps })
    });

    const data = await res.json();
    btnGen.disabled = false;
    btnComp.disabled = false;
    statusBox.style.display = 'none';

    // Display Side-by-Side Comparison
    const compViewBox = document.getElementById('compViewBox');
    compViewBox.style.display = 'block';

    const b = data.baseline;
    const p = data.proposed;

    const compVideoBaseline = document.getElementById('compVideoBaseline');
    const compVideoProposed = document.getElementById('compVideoProposed');

    compVideoBaseline.src = b.video_url;
    compVideoProposed.src = p.video_url;

    compVideoBaseline.play();
    compVideoProposed.play();

    const speedup = (b.metrics.interp_latency_sec / Math.max(1e-4, p.metrics.interp_latency_sec)).toFixed(2);
    const flickerDiff = (((b.metrics.flicker - p.metrics.flicker) / Math.max(1e-4, b.metrics.flicker)) * 100).toFixed(1);

    document.getElementById('compMetricsBaseline').innerHTML = `
      Latency: ${b.metrics.generation_time_sec}s | Interp: ${b.metrics.interp_latency_sec}s<br>
      Flicker: ${b.metrics.flicker.toFixed(4)} | PSNR: ${b.metrics.psnr.toFixed(2)}dB
    `;

    document.getElementById('compMetricsProposed').innerHTML = `
      Latency: ${p.metrics.generation_time_sec}s | Interp: ${p.metrics.interp_latency_sec}s (${speedup}x Speedup)<br>
      Flicker: ${p.metrics.flicker.toFixed(4)} (${flickerDiff}% Smoother) | PSNR: ${p.metrics.psnr.toFixed(2)}dB
    `;

  } catch (err) {
    btnGen.disabled = false;
    btnComp.disabled = false;
    showError(err.message);
  }
}

function showError(msg) {
  document.getElementById('btnGenerate').disabled = false;
  document.getElementById('btnCompare').disabled = false;
  document.getElementById('statusStage').innerText = 'ERROR';
  document.getElementById('statusPct').innerText = '100%';
  document.getElementById('progressBar').style.width = '100%';
  document.getElementById('progressBar').style.background = '#ef4444';
  alert(`Error: ${msg}`);
}
