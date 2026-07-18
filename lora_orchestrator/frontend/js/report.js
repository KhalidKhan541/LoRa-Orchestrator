async function showReport(reportUrl, sessionId) {
  const panel = document.getElementById('reportPanel');
  const header = document.getElementById('reportHeader');
  const icon = document.getElementById('reportIcon');
  const title = document.getElementById('reportTitle');
  const content = document.getElementById('reportContent');

  panel.classList.add('visible');

  const status = window.app.status || 'done';
  if (status === 'done') {
    header.className = 'report-header success';
    icon.textContent = '✓';
    title.textContent = 'Training Complete';
  } else {
    header.className = 'report-header failed';
    icon.textContent = '✗';
    title.textContent = 'Training Failed';
  }

  try {
    const res = await fetch(reportUrl);
    const md = await res.text();
    content.innerHTML = marked.parse(md);
  } catch (err) {
    content.innerHTML = '<p style="color:var(--status-error)">Failed to load report.</p>';
  }

  document.getElementById('downloadReportBtn').onclick = async () => {
    try {
      const res = await fetch(reportUrl);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `report_${sessionId}.md`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      alert('Download failed');
    }
  };

  document.getElementById('downloadAdapterBtn').onclick = () => {
    window.open(`/api/adapter/${sessionId}`, '_blank');
  };
}

window.reportModule = { showReport };
