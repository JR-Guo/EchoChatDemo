(async function () {
  const dz = document.getElementById('dropzone');
  const input = document.getElementById('file-input');
  const list = document.getElementById('file-list');
  const continueBtn = document.getElementById('continue-btn');
  const newBtn = document.getElementById('new-study-btn');

  let studyId = null;
  const fileRows = new Map();

  async function ensureStudy() {
    if (studyId) return studyId;
    const r = await fetch('/api/study', { method: 'POST' });
    studyId = (await r.json()).study_id;
    return studyId;
  }

  function row(name) {
    const el = document.createElement('div');
    el.className = 'card px-4 py-3 flex items-center justify-between';
    el.innerHTML = `
      <div class="min-w-0">
        <div class="font-medium truncate">${name}</div>
        <div class="text-xs text-muted mt-0.5" data-status>Queued</div>
      </div>
      <div class="w-32 bg-slate-100 rounded-full h-1 overflow-hidden">
        <div class="h-1 bg-primary" style="width:0%" data-bar></div>
      </div>
    `;
    list.appendChild(el);
    return el;
  }

  function setStatus(el, text) { el.querySelector('[data-status]').textContent = text; }
  function setBar(el, pct) { el.querySelector('[data-bar]').style.width = pct + '%'; }

  async function sniffDicom(file) {
    if (file.size < 132) return false;
    const buf = await file.slice(128, 132).arrayBuffer();
    const s = new TextDecoder().decode(buf);
    return s === 'DICM';
  }

  async function uploadOne(file) {
    if (!await sniffDicom(file)) {
      const el = row(file.name);
      setStatus(el, 'Rejected (not DICOM)');
      return;
    }
    const el = row(file.name);
    setStatus(el, 'Uploading…');
    const sid = await ensureStudy();
    const form = new FormData();
    form.append('file', file, file.name);
    const xhr = new XMLHttpRequest();
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) setBar(el, (e.loaded / e.total * 100) | 0);
    };
    await new Promise((resolve) => {
      xhr.onload = () => {
        if (xhr.status === 200) {
          const j = JSON.parse(xhr.responseText);
          fileRows.set(j.file_id, el);
          setStatus(el, 'Uploaded, awaiting classification…');
        } else {
          setStatus(el, 'Failed: ' + xhr.status);
        }
        resolve();
      };
      xhr.open('POST', `/api/study/${sid}/upload`);
      xhr.send(form);
    });
  }

  async function runProcess() {
    if (!studyId) return;
    const es = new EventSource(`/api/study/${studyId}/process`);
    es.addEventListener('phase', (evt) => {
      const data = JSON.parse(evt.data);
      if (data.file_id && fileRows.has(data.file_id)) {
        const el = fileRows.get(data.file_id);
        setStatus(el, `${data.phase.replace('_', ' ')}…`);
      }
    });
    es.addEventListener('clip', (evt) => {
      const data = JSON.parse(evt.data);
      const el = fileRows.get(data.file_id);
      if (el) setStatus(el, `View: ${data.view || 'Unknown'}`);
    });
    es.addEventListener('error', (evt) => { es.close(); });
    es.addEventListener('done', () => {
      es.close();
      continueBtn.disabled = false;
    });
  }

  async function handleFiles(files) {
    for (const f of files) await uploadOne(f);
    await runProcess();
  }

  dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('dropzone--hover'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('dropzone--hover'));
  dz.addEventListener('drop', (e) => {
    e.preventDefault();
    dz.classList.remove('dropzone--hover');
    if (e.dataTransfer.files) handleFiles(e.dataTransfer.files);
  });
  input.addEventListener('change', () => handleFiles(input.files));

  continueBtn.addEventListener('click', () => {
    if (studyId) location.href = `/workspace/${studyId}`;
  });
  newBtn.addEventListener('click', () => location.reload());
})();
