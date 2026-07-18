const API_BASE = window.location.origin;

async function populateModelSelect() {
  try {
    const res = await fetch(`${API_BASE}/api/models`);
    const data = await res.json();
    const select = document.getElementById('baseModelSelect');
    select.innerHTML = '';
    data.models.forEach((m) => {
      const opt = document.createElement('option');
      opt.value = m.key;
      opt.textContent = `${m.name} (${m.params})`;
      select.appendChild(opt);
    });
  } catch (err) {
    console.error('Failed to load models:', err);
  }
}

function getConfig() {
  const toggles = document.querySelectorAll('.toggle');
  const config = {
    session_id: window.app.sessionId,
    file_path: window.app.filePath,
    file_name: window.app.fileName,
    template_format: document.getElementById('templateSelect').value,
    base_model: document.getElementById('baseModelSelect').value,
    target_metric: document.getElementById('targetMetricSelect').value,
    target_value: parseFloat(document.getElementById('targetValueInput').value),
    max_iterations: parseInt(document.getElementById('maxIterInput').value),
    max_samples: parseInt(document.getElementById('maxSamplesInput').value),
    quantization: document.getElementById('quantizationToggle').classList.contains('active'),
    use_wandb: document.getElementById('wandbToggle').classList.contains('active'),
    use_mlflow: false,
    llm_provider: 'openai',
    llm_model: null,
  };
  return config;
}

function setFormEnabled(enabled) {
  const elements = document.querySelectorAll('.config-form select, .config-form input, .toggle');
  elements.forEach((el) => {
    el.style.pointerEvents = enabled ? 'auto' : 'none';
    el.style.opacity = enabled ? '1' : '0.5';
  });
}

window.configModule = { populateModelSelect, getConfig, setFormEnabled };
