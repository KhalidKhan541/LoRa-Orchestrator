let eventSource = null;

const AGENT_ICONS = {
  dataset_agent: '📊',
  hyperparam_agent: '⚙️',
  training_agent: '🏋️',
  eval_agent: '📏',
  decision_agent: '🧠',
  report_agent: '📄',
};

const AGENT_TAGS = {
  dataset_agent: 'dataset',
  hyperparam_agent: 'hyperparam',
  training_agent: 'training',
  eval_agent: 'eval',
  decision_agent: 'decision',
  report_agent: 'report',
};

function startStream(sessionId) {
  const url = `${window.location.origin}/api/stream/${sessionId}`;
  eventSource = new EventSource(url);

  eventSource.onmessage = (e) => {
    try {
      const event = JSON.parse(e.data);
      dispatchEvent(event);
    } catch (err) {
      console.error('SSE parse error:', err);
    }
  };

  eventSource.onerror = () => {
    console.error('SSE connection error');
    addLogEntry('system', 'Connection lost. Retrying...', 'error');
  };
}

function dispatchEvent(event) {
  switch (event.type) {
    case 'agent_start':
      setNodeState(event.agent, 'active');
      addLogEntry(event.agent, event.message || `Running ${event.agent}...`);
      break;

    case 'agent_complete':
      setNodeState(event.agent, 'complete');
      if (event.agent === 'hyperparam_agent' && event.data) {
        updateHpPanel(event.data);
      }
      addLogEntry(event.agent, 'Completed');
      break;

    case 'training_log':
      window.chartsModule.addEpochPoint(event.epoch, event.train_loss, event.val_loss);
      addLogEntry('training', `Epoch ${event.epoch} — loss: ${event.train_loss.toFixed(4)} val_loss: ${event.val_loss.toFixed(4)}`);
      break;

    case 'early_stop':
      window.chartsModule.addEarlyStopMarker(event.epoch, event.reason);
      addLogEntry('training', `⚠ Early stop: ${event.reason} at epoch ${event.epoch}`, 'warning');
      break;

    case 'eval_complete':
      updateEvalScores(event);
      addLogEntry('eval', `BLEU=${event.bleu.toFixed(4)} ROUGE-L=${event.rouge_l.toFixed(4)} PPL=${event.perplexity.toFixed(2)}`);
      break;

    case 'decision':
      setNodeState('decision_agent', 'active');
      setTimeout(() => setNodeState('decision_agent', 'complete'), 1000);
      addLogEntry('decision', `${event.decision.toUpperCase()}: ${event.reasoning}`);
      break;

    case 'complete':
      document.getElementById('statusBadge').textContent = 'Done';
      document.getElementById('statusBadge').className = 'status-badge done';
      addLogEntry('system', 'Pipeline complete!');
      if (event.report_url) {
        window.reportModule.showReport(event.report_url, window.app.sessionId);
      }
      stopStream();
      break;

    case 'error':
      addLogEntry('system', `Error: ${event.message}`, 'error');
      document.getElementById('statusBadge').textContent = 'Failed';
      document.getElementById('statusBadge').className = 'status-badge failed';
      stopStream();
      break;

    case 'ping':
      break;
  }
}

function setNodeState(agent, state) {
  const agentMap = {
    dataset: 'dataset_agent',
    hyperparam: 'hyperparam_agent',
    training: 'training_agent',
    eval: 'eval_agent',
    decision: 'decision_agent',
    report: 'report_agent',
  };
  const key = agentMap[agent] || agent;
  const node = document.querySelector(`.pipeline-node[data-agent="${key.replace('_agent', '')}"]`);
  if (!node) return;
  node.className = `pipeline-node node-${state}`;
}

function addLogEntry(agent, message, type = '') {
  const log = document.getElementById('agentLog');
  const now = new Date();
  const ts = now.toTimeString().substring(0, 8);
  const tag = AGENT_TAGS[agent] || agent;
  const entry = document.createElement('div');
  entry.className = `log-entry ${type}`;
  entry.innerHTML = `<span class="timestamp">[${ts}]</span> <span class="agent-tag ${tag}">[${agent}]</span> ${message}`;
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
}

function updateEvalScores(event) {
  const prev = window.app.lastEval || {};
  setMetricCard('bleu', event.bleu, prev.bleu, false);
  setMetricCard('rouge', event.rouge_l, prev.rouge_l, false);
  setMetricCard('ppl', event.perplexity, prev.perplexity, true);
  setMetricCard('em', event.exact_match, prev.exact_match, false);
  window.app.lastEval = { bleu: event.bleu, rouge_l: event.rouge_l, perplexity: event.perplexity, exact_match: event.exact_match };
}

function setMetricCard(prefix, value, prevValue, lowerIsBetter = false) {
  const valEl = document.getElementById(prefix + 'Value');
  const deltaEl = document.getElementById(prefix + 'Delta');
  const cardEl = document.getElementById('metric' + prefix.charAt(0).toUpperCase() + prefix.slice(1));
  if (!valEl) return;

  if (prefix === 'ppl') {
    valEl.textContent = value.toFixed(2);
  } else if (prefix === 'em') {
    valEl.textContent = (value * 100).toFixed(1) + '%';
  } else {
    valEl.textContent = value.toFixed(4);
  }

  if (prevValue !== undefined && prevValue !== null) {
    const delta = value - prevValue;
    const improved = lowerIsBetter ? delta < 0 : delta > 0;
    deltaEl.textContent = (improved ? '↑' : '↓') + ' ' + Math.abs(delta).toFixed(4);
    deltaEl.className = `metric-delta ${improved ? 'positive' : 'negative'}`;
  }

  const target = parseFloat(document.getElementById('targetValueInput')?.value || '0.6');
  if (prefix === 'rouge' && value >= target && cardEl) {
    cardEl.classList.add('target-met');
  }
}

function updateHpPanel(data) {
  const grid = document.getElementById('hpGrid');
  const reasoning = document.getElementById('hpReasoning');
  if (!grid) return;

  const fields = ['lora_r', 'lora_alpha', 'lora_dropout', 'learning_rate', 'num_epochs', 'batch_size', 'lr_scheduler', 'optimizer'];
  grid.innerHTML = '';
  fields.forEach((key) => {
    if (data[key] !== undefined) {
      const item = document.createElement('div');
      item.className = 'hp-item';
      item.innerHTML = `<span class="hp-key">${key}</span><span class="hp-val">${data[key]}</span>`;
      grid.appendChild(item);
    }
  });

  if (data.reasoning) {
    reasoning.textContent = data.reasoning;
  }
}

function stopStream() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
}

window.streamModule = { startStream, stopStream, addLogEntry, setNodeState };
