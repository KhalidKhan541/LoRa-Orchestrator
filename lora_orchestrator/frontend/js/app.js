window.app = {
  sessionId: null,
  filePath: null,
  fileName: null,
  isRunning: false,
  status: 'idle',
  lastEval: null,
};

document.addEventListener('DOMContentLoaded', () => {
  window.uploadModule.initUpload();
  window.configModule.populateModelSelect();
  window.chartsModule.initChart();

  // Toggle switches
  document.querySelectorAll('.toggle').forEach((toggle) => {
    toggle.addEventListener('click', () => {
      toggle.classList.toggle('active');
    });
  });

  // Start button
  const startBtn = document.getElementById('startBtn');
  startBtn.addEventListener('click', handleStart);

  // Stop button (same button, toggles)
  async function handleStart() {
    if (window.app.isRunning) {
      handleStop();
      return;
    }

    if (!window.app.sessionId) {
      alert('Please upload a dataset first.');
      return;
    }

    const config = window.configModule.getConfig();
    config.file_path = window.app.filePath;
    config.file_name = window.app.fileName;

    try {
      const res = await fetch(`${window.location.origin}/api/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });

      if (!res.ok) {
        const err = await res.text();
        alert('Failed to start: ' + err);
        return;
      }

      const data = await res.json();
      window.app.isRunning = true;
      window.app.status = 'running';

      startBtn.textContent = '⏹ Stop Run';
      startBtn.classList.add('running', 'active-pulse');
      document.getElementById('statusBadge').textContent = 'Running';
      document.getElementById('statusBadge').className = 'status-badge running';
      window.configModule.setFormEnabled(false);
      window.chartsModule.resetChart();

      window.streamModule.startStream(data.session_id);

    } catch (err) {
      alert('Error: ' + err.message);
    }
  }

  function handleStop() {
    window.streamModule.stopStream();
    window.app.isRunning = false;
    window.app.status = 'idle';

    startBtn.innerHTML = '▶ Start Fine-Tuning';
    startBtn.classList.remove('running', 'active-pulse');
    document.getElementById('statusBadge').textContent = 'Idle';
    document.getElementById('statusBadge').className = 'status-badge';
    window.configModule.setFormEnabled(true);
  }
});
