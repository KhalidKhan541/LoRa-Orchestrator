const API_BASE = window.location.origin;

function initUpload() {
  const zone = document.getElementById('uploadZone');
  const input = document.getElementById('fileInput');

  zone.addEventListener('click', () => input.click());

  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });

  zone.addEventListener('dragleave', () => {
    zone.classList.remove('drag-over');
  });

  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  input.addEventListener('change', () => {
    if (input.files.length > 0) {
      handleFile(input.files[0]);
    }
  });
}

async function handleFile(file) {
  const allowed = ['.csv', '.jsonl', '.json'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!allowed.includes(ext)) {
    alert('Only .csv, .jsonl, .json files are accepted.');
    return;
  }
  if (file.size > 500 * 1024 * 1024) {
    alert('File too large (max 500MB).');
    return;
  }

  const progress = document.getElementById('uploadProgress');
  const fill = document.getElementById('uploadProgressFill');
  const fileInfo = document.getElementById('uploadFileInfo');
  const fileName = document.getElementById('uploadFileName');
  const fileSize = document.getElementById('uploadFileSize');

  progress.style.display = 'block';
  fill.style.width = '0%';
  fileInfo.style.display = 'none';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE}/api/upload`);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 100);
        fill.style.width = pct + '%';
      }
    };

    const response = await new Promise((resolve, reject) => {
      xhr.onload = () => {
        if (xhr.status === 200) {
          resolve(JSON.parse(xhr.responseText));
        } else {
          reject(new Error(xhr.responseText));
        }
      };
      xhr.onerror = () => reject(new Error('Upload failed'));
      xhr.send(formData);
    });

    window.app.sessionId = response.session_id;
    window.app.filePath = response.file_path;
    window.app.fileName = response.file_name;

    progress.style.display = 'none';
    fileInfo.style.display = 'flex';
    fileName.textContent = response.file_name;
    fileSize.textContent = response.file_size_mb + ' MB';

    document.getElementById('startBtn').disabled = false;
    document.getElementById('sessionIdDisplay').textContent = response.session_id.substring(0, 8) + '...';

  } catch (err) {
    progress.style.display = 'none';
    alert('Upload failed: ' + err.message);
  }
}

window.uploadModule = { initUpload };
