let lossChart = null;

function initChart() {
  const ctx = document.getElementById('lossChart').getContext('2d');
  lossChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Train Loss',
          data: [],
          borderColor: '#f59e0b',
          backgroundColor: 'rgba(245,158,11,0.1)',
          borderWidth: 2,
          tension: 0.3,
          pointRadius: 4,
          pointBackgroundColor: '#f59e0b',
        },
        {
          label: 'Val Loss',
          data: [],
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59,130,246,0.1)',
          borderWidth: 2,
          borderDash: [5, 5],
          tension: 0.3,
          pointRadius: 4,
          pointBackgroundColor: '#3b82f6',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 500 },
      scales: {
        x: {
          title: { display: true, text: 'Epoch', color: '#94a3b8' },
          ticks: { color: '#94a3b8' },
          grid: { color: '#1e2d3d' },
        },
        y: {
          title: { display: true, text: 'Loss', color: '#94a3b8' },
          ticks: { color: '#94a3b8' },
          grid: { color: '#1e2d3d' },
          beginAtZero: false,
        },
      },
      plugins: {
        legend: {
          position: 'top',
          align: 'end',
          labels: { color: '#94a3b8', boxWidth: 12, padding: 16 },
        },
        tooltip: {
          backgroundColor: '#1a2235',
          titleColor: '#f1f5f9',
          bodyColor: '#94a3b8',
          borderColor: '#2a3f5a',
          borderWidth: 1,
        },
      },
    },
  });
}

function addEpochPoint(epoch, trainLoss, valLoss) {
  if (!lossChart) return;
  lossChart.data.labels.push('Epoch ' + epoch);
  lossChart.data.datasets[0].data.push(trainLoss);
  lossChart.data.datasets[1].data.push(valLoss);
  lossChart.update();
}

function addEarlyStopMarker(epoch, reason) {
  if (!lossChart) return;
  const chart = lossChart;
  const xScale = chart.scales.x;
  const yScale = chart.scales.y;
  const meta = chart.getDatasetMeta(0);
  if (meta.data[epoch - 1]) {
    const x = meta.data[epoch - 1].x;
    const annotationPlugin = {
      id: 'earlyStopLine',
      afterDraw: (chart) => {
        const ctx = chart.ctx;
        const xAxis = chart.scales.x;
        const yAxis = chart.scales.y;
        const xPos = xAxis.getPixelForValue(epoch - 1);
        ctx.save();
        ctx.beginPath();
        ctx.setLineDash([5, 5]);
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2;
        ctx.moveTo(xPos, yAxis.top);
        ctx.lineTo(xPos, yAxis.bottom);
        ctx.stroke();
        ctx.fillStyle = '#ef4444';
        ctx.font = '11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Early Stop: ' + reason, xPos, yAxis.top - 8);
        ctx.restore();
      },
    };
    if (!chart.config.plugins) chart.config.plugins = [];
    chart.config.plugins.push(annotationPlugin);
    chart.update();
  }
}

function resetChart() {
  if (!lossChart) return;
  lossChart.data.labels = [];
  lossChart.data.datasets[0].data = [];
  lossChart.data.datasets[1].data = [];
  lossChart.update();
}

window.chartsModule = { initChart, addEpochPoint, addEarlyStopMarker, resetChart };
