# UI_SPEC.md — LoRA Fine-Tuning Orchestrator
> Frontend layout, design system, components, and interaction states

---

## 1. Layout Overview

Two-phase UI: **Setup Phase** (before training starts) and **Run Phase**
(training is live). The layout does not change — panels update in place.

```
┌─────────────────────────────────────────────────────────────────────┐
│  HEADER: "LoRA Orchestrator"  ·  status badge  ·  session ID       │
├────────────────────────┬────────────────────────────────────────────┤
│                        │                                             │
│   LEFT PANEL (380px)   │   RIGHT PANEL (flex: 1, min-width: 580px) │
│                        │                                             │
│  ┌──────────────────┐  │  ┌────────────────────────────────────┐   │
│  │  1. Upload Panel │  │  │   Agent Pipeline Visualizer         │   │
│  │  (drag-drop)     │  │  │   [Dataset]→[HP]→[Train]→[Eval]→   │   │
│  └──────────────────┘  │  │   [Decision] ↩ or → [Report]       │   │
│                        │  └────────────────────────────────────┘   │
│  ┌──────────────────┐  │                                             │
│  │  2. Config Panel │  │  ┌────────────────────────────────────┐   │
│  │  · Base model    │  │  │   Live Loss Chart                   │   │
│  │  · Template      │  │  │   (Chart.js line, dual axis)        │   │
│  │  · Target metric │  │  │   train_loss + val_loss per epoch   │   │
│  │  · Target value  │  │  └────────────────────────────────────┘   │
│  │  · Max retries   │  │                                             │
│  └──────────────────┘  │  ┌──────────────┬─────────────────────┐   │
│                        │  │ Eval Scores  │  Hyperparameter Log  │   │
│  ┌──────────────────┐  │  │  BLEU: —     │  rank: —  lr: —      │   │
│  │  3. Start Button │  │  │  ROUGE: —    │  alpha: —  ep: —     │   │
│  │  [▶ Start Run]   │  │  │  PPL: —      │  dropout: —         │   │
│  └──────────────────┘  │  └──────────────┴─────────────────────┘   │
│                        │                                             │
│  ┌──────────────────┐  │  ┌────────────────────────────────────┐   │
│  │  4. Agent Log    │  │  │   Report Panel                      │   │
│  │  (scrollable)    │  │  │   (hidden until done)               │   │
│  └──────────────────┘  │  └────────────────────────────────────┘   │
│                        │                                             │
└────────────────────────┴────────────────────────────────────────────┘
```

---

## 2. Design System

### Color Tokens

```css
:root {
  /* Backgrounds */
  --bg-base:         #0a0e1a;   /* page background */
  --bg-panel:        #111827;   /* card / panel background */
  --bg-card:         #1a2235;   /* inner card / nested surface */
  --bg-input:        #1e2a3a;   /* input field background */
  --bg-hover:        #1f2d42;   /* hover state */

  /* Borders */
  --border-subtle:   #1e2d3d;   /* panel borders */
  --border-default:  #2a3f5a;   /* input / card borders */
  --border-active:   #3b82f6;   /* focused / active element */

  /* Accent colors (agents) */
  --accent-dataset:  #3b82f6;   /* blue — Dataset Agent */
  --accent-hyparam:  #8b5cf6;   /* purple — Hyperparameter Agent */
  --accent-train:    #f59e0b;   /* amber — Training Agent */
  --accent-eval:     #14b8a6;   /* teal — Evaluation Agent */
  --accent-decision: #ec4899;   /* pink — Decision Agent */
  --accent-report:   #10b981;   /* green — Report Agent */

  /* Status colors */
  --status-idle:     #374151;
  --status-active:   #3b82f6;
  --status-success:  #10b981;
  --status-warning:  #f59e0b;
  --status-error:    #ef4444;

  /* Text */
  --text-primary:    #f1f5f9;
  --text-secondary:  #94a3b8;
  --text-muted:      #475569;
  --text-code:       #fb923c;

  /* Chart colors */
  --chart-train:     #f59e0b;
  --chart-val:       #3b82f6;
  --chart-grid:      #1e2d3d;

  /* Typography */
  --font-sans:  'Inter', system-ui, sans-serif;
  --font-mono:  'JetBrains Mono', 'Fira Code', monospace;

  /* Spacing scale */
  --space-xs:  4px;
  --space-sm:  8px;
  --space-md:  16px;
  --space-lg:  24px;
  --space-xl:  32px;

  /* Radius */
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 14px;

  /* Shadows */
  --shadow-panel: 0 4px 24px rgba(0,0,0,0.4);
  --shadow-glow-blue: 0 0 20px rgba(59,130,246,0.25);
  --shadow-glow-green: 0 0 20px rgba(16,185,129,0.25);
}
```

### Typography

```css
body           { font-family: var(--font-sans); font-size: 14px; }
.panel-title   { font-size: 11px; font-weight: 600; letter-spacing: 0.1em;
                 text-transform: uppercase; color: var(--text-muted); }
.metric-value  { font-size: 28px; font-weight: 700; font-family: var(--font-mono); }
.metric-label  { font-size: 11px; color: var(--text-secondary); }
.log-entry     { font-family: var(--font-mono); font-size: 12px; }
code, pre      { font-family: var(--font-mono); color: var(--text-code); }
```

---

## 3. Component Specifications

### 3.1 Upload Panel

**States:** empty → file_selected → uploading → uploaded → error

```
┌─────────────────────────────────┐
│  ┌─────────────────────────┐    │
│  │                         │    │
│  │   ↑  Drop dataset here  │    │  ← dashed border, dashes animate on drag-over
│  │   or click to browse    │    │
│  │                         │    │
│  │  .csv  .jsonl  .json    │    │
│  └─────────────────────────┘    │
│                                 │
│  After file selected:           │
│  📄 dataset.jsonl  (2.4 MB)    │
│  ████████████████░░  80%        │  ← upload progress bar
└─────────────────────────────────┘
```

- Accepted: `.csv`, `.jsonl`, `.json`
- Max size: 500MB (show error if exceeded)
- On hover/drag: border color → `--accent-dataset`, background tints blue
- Upload progress bar: animated fill, color `--accent-dataset`
- After upload: show filename, size, and green checkmark

### 3.2 Config Panel

Fields (in order):

| Field | Type | Default | Options |
|-------|------|---------|---------|
| Base Model | Select | qwen2.5-7b | All SUPPORTED_MODELS keys |
| Template Format | Select | alpaca | alpaca, sharegpt |
| Target Metric | Select | rouge_l | bleu, rouge_l, perplexity, exact_match |
| Target Value | Number | 0.60 | 0.0 – 1.0 (step 0.01) |
| Max Iterations | Number | 3 | 1 – 5 |
| Quantization | Toggle | ON | 4-bit / none |
| W&B Tracking | Toggle | OFF | — |
| Max Samples | Number | 10000 | 100 – 500000 |

Styling: dark input fields, `--bg-input` background, `--border-default` border,
focus ring `--border-active`. Toggles: pill shape, green when ON.

### 3.3 Start Button

```
[ ▶  Start Fine-Tuning ]
```

- Default: `--accent-dataset` background, white text, full width
- Hover: 10% brighter, subtle upward translate (1px)
- Disabled (no file uploaded): `--status-idle` background, cursor: not-allowed
- Running: replaced by `[ ⏹  Stop Run ]` in `--status-error` color
- Pulsing animation while pipeline is active

### 3.4 Agent Pipeline Visualizer

Five agent boxes connected by arrows. Each box has 4 states:

```
IDLE:     ┌─────────────┐    ACTIVE:   ┌─────────────┐
          │  ○ Dataset  │              │  ◉ Dataset  │  ← spinning ring, glow
          └─────────────┘              └─────────────┘
                                         box-shadow: var(--shadow-glow-blue)
                                         border-color: var(--accent-dataset)

COMPLETE: ┌─────────────┐    ERROR:    ┌─────────────┐
          │  ✓ Dataset  │              │  ✗ Dataset  │
          └─────────────┘              └─────────────┘
          border: --status-success     border: --status-error
```

Arrow between boxes: `──▶` in `--border-default` color.
The Decision Agent box has a return arrow curving back to Hyperparam Agent.
Show iteration counter badge on Decision node: `#2` on second pass.

Agent color mapping:
- Dataset → `--accent-dataset` (blue)
- Hyperparam → `--accent-hyparam` (purple)
- Training → `--accent-train` (amber)
- Eval → `--accent-eval` (teal)
- Decision → `--accent-decision` (pink)
- Report → `--accent-report` (green)

### 3.5 Live Loss Chart

Chart.js line chart. Updates live via SSE `training_log` events.

```
Loss
1.8 │╲
1.4 │ ╲─╲
1.0 │    ╲──╲────╲
0.8 │         ╲───╲──── train_loss (amber)
0.6 │              ╲─── val_loss (blue)
    └──────────────────▶ Epoch
    1   2   3   4   5   6
```

- X axis: epoch number (auto-extends as epochs arrive)
- Y axis: loss value (auto-scale, min 0)
- Two lines: `train_loss` (amber, solid) and `val_loss` (blue, dashed)
- Animation: point appears and line draws to it on each SSE event
- Tooltip on hover: shows epoch + both loss values
- Legend: top-right, inside chart
- Background: `--bg-card`, grid lines: `--chart-grid`
- Early stop event: vertical red dashed line with label "Early Stop"

### 3.6 Eval Scores Panel

Four metric cards in a 2×2 grid:

```
┌─────────────┐  ┌─────────────┐
│  BLEU       │  │  ROUGE-L    │
│   0.423     │  │   0.612     │
│  ↑ +0.05    │  │  ↑ +0.09   │
└─────────────┘  └─────────────┘
┌─────────────┐  ┌─────────────┐
│  Perplexity │  │ Exact Match │
│   12.3      │  │   71.4%     │
│  ↓ -3.1     │  │  ↑ +8.2%   │
└─────────────┘  └─────────────┘
```

- `--` before first evaluation completes
- Delta arrow: green ↑ for improvement, red ↓ for regression
- Target value shown as faint line behind the number: `target: 0.60`
- If metric meets target: card border turns `--status-success`

### 3.7 Hyperparameter Log Panel

Shows current hyperparams in a compact grid. Updates when Decision Agent
triggers an "adjust" and Hyperparam Agent selects new values.

```
┌─────────────────────────────────────────┐
│  HYPERPARAMETERS             Iter #1    │
│                                         │
│  rank:      16      alpha:    32        │
│  dropout:   0.05    lr:       2e-4      │
│  epochs:    5       scheduler: cosine   │
│  optimizer: paged_adamw_8bit            │
│                                         │
│  💬 "Baseline config for 7B model..."  │
└─────────────────────────────────────────┘
```

- Show LLM reasoning string (truncated to 2 lines, expandable on click)
- Changed values on iteration 2+: highlight in `--accent-hyparam`
- Previous iteration values shown strikethrough in muted color

### 3.8 Agent Log Panel

Fixed-height (220px), scrollable, monospace, dark:

```
[12:34:01] [Dataset]    Loaded 8,423 rows from dataset.jsonl
[12:34:02] [Dataset]    Removed 47 duplicates, 12 empty rows
[12:34:03] [Dataset]    Formatted to Alpaca template
[12:34:05] [Hyperparam] Selected rank=16, lr=2e-4 (reasoning: ...)
[12:34:06] [Training]   Epoch 1/5 — loss: 1.821 val_loss: 1.934
[12:34:18] [Training]   Epoch 2/5 — loss: 1.432 val_loss: 1.611
```

- Color per agent matches pipeline node colors
- Auto-scroll to bottom on new entry
- Pulsing `●` indicator at top-right while running
- Monospace font, 12px

### 3.9 Report Panel

Hidden until `status == "done"` or `status == "failed"`. Then slides in.

```
┌───────────────────────────────────────────────────────┐
│  ✓ Training Complete                                   │
│                                                        │
│  Experiment Report                                     │
│  ─────────────────────────────────────────────────    │
│  [rendered Markdown content here]                      │
│                                                        │
│  [ ⬇ Download Adapter ]  [ ⬇ Download Report (.md) ] │
└───────────────────────────────────────────────────────┘
```

- Success: green header with ✓
- Failure: red header with ✗ and escalation message
- Report rendered as HTML from Markdown (use marked.js)

---

## 4. Responsive Behavior

- **Desktop (≥1200px):** Full two-panel layout as above
- **Tablet (768–1199px):** Right panel stacks below left panel, full width
- **Mobile (<768px):** Not supported for v1 — show "Use desktop for best experience"

---

## 5. Interaction States Summary

| Event | UI change |
|-------|-----------|
| File dropped | Upload panel shows progress bar |
| Upload complete | Green checkmark, file name shown, Start button activates |
| Start clicked | Button → "Stop Run", pipeline node 1 activates |
| `agent_start` SSE | Corresponding node → active (spinning) |
| `agent_complete` SSE | Corresponding node → complete (checkmark) |
| `training_log` SSE | Chart point added, log entry added |
| `early_stop` SSE | Chart shows vertical red line, log entry |
| `eval_complete` SSE | Eval score cards update with values |
| `decision: adjust` SSE | Decision node pulses pink, arrow loops back, iteration badge updates |
| `complete` SSE | Report panel slides in, all nodes green |
| `error` SSE | Error node turns red, error message in log, report panel shows failure |

---

## 6. JavaScript Files Responsibilities

| File | Responsibility |
|------|---------------|
| `app.js` | Init, state object, coordinates all modules |
| `upload.js` | Drag-drop, file validation, POST /api/upload, progress |
| `config.js` | Config form read/write, build request payload |
| `stream.js` | EventSource connection, parse SSE events, dispatch to other modules |
| `charts.js` | Chart.js init, `addEpochPoint()`, early stop marker |
| `report.js` | Show report panel, render Markdown via marked.js, download buttons |

---

## 7. External JS Libraries (CDN)

```html
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked@9.1.6/marked.min.js"></script>
```

No other external libraries. No frameworks. No build step.
