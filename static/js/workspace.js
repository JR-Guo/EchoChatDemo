(async function () {
  const sid = window.ECHOCHAT_STUDY_ID;

  document.querySelectorAll('.tab-item').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-item').forEach((b) => b.setAttribute('aria-selected', 'false'));
      btn.setAttribute('aria-selected', 'true');
      const key = btn.dataset.tab;
      document.querySelectorAll('.tab-panel').forEach((p) => {
        p.classList.toggle('hidden', p.dataset.panel !== key);
      });
    });
  });

  const constants = await (await fetch('/api/constants')).json();

  async function renderStudy() {
    const s = await (await fetch(`/api/study/${sid}`)).json();

    const clipsEl = document.getElementById('clips-list');
    clipsEl.innerHTML = '';
    for (const c of s.clips) {
      const view = c.user_view || c.view || null;
      const row = document.createElement('div');
      row.className = 'flex items-center gap-3 p-2 rounded-lg hover:bg-slate-50 transition';
      const badgeCls = view ? 'badge badge--ok' : 'badge badge--unknown';
      row.innerHTML = `
        <img src="/api/study/${sid}/clip/${c.file_id}/thumbnail" alt="" class="w-12 h-12 rounded-md object-cover bg-slate-100"/>
        <div class="min-w-0 flex-1">
          <div class="text-sm font-medium truncate">${c.original_filename}</div>
          <div class="mt-1 flex items-center gap-2">
            <span class="${badgeCls}">${view || 'Unknown'}</span>
            ${c.user_view ? '<span class="text-xs text-muted">(manual)</span>' : ''}
          </div>
        </div>
        <button class="text-xs text-muted hover:text-danger transition" data-delete="${c.file_id}">Remove</button>
      `;
      row.querySelector('span.badge').addEventListener('click', async () => {
        const choice = prompt('Set view (exact label from VIEW_LABELS, or blank for unknown):', view || '');
        if (choice === null) return;
        const body = choice.trim() ? { user_view: choice.trim() } : { user_view: null };
        const r = await fetch(`/api/study/${sid}/clip/${c.file_id}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (r.ok) renderStudy();
      });
      row.querySelector('[data-delete]').addEventListener('click', async () => {
        if (!confirm(`Remove ${c.original_filename}?`)) return;
        await fetch(`/api/study/${sid}/clip/${c.file_id}`, { method: 'DELETE' });
        renderStudy();
      });
      clipsEl.appendChild(row);
    }

    const avail = document.getElementById('tasks-avail');
    const missing = s.tasks.missing_groups || [];
    avail.innerHTML = `
      <div>Report: ${s.tasks.report ? '<span class="badge badge--ok">ready</span>' : '<span class="badge badge--warn">no clips</span>'}</div>
      <div>Measurement: ${s.tasks.measurement ? '<span class="badge badge--ok">ready</span>' : '<span class="badge badge--warn">no clips</span>'}</div>
      <div>Disease: ${s.tasks.disease ? '<span class="badge badge--ok">ready</span>' : '<span class="badge badge--warn">no clips</span>'}</div>
      <div>VQA: ${s.tasks.vqa ? '<span class="badge badge--ok">ready</span>' : '<span class="badge badge--warn">no clips</span>'}</div>
      ${missing.length ? `<div class="pt-2 text-xs text-muted">Missing views: ${missing.join(', ')} (soft warning)</div>` : ''}
    `;
  }
  await renderStudy();

  const measPresetsEl = document.getElementById('measure-presets');
  const measItemsEl = document.getElementById('measure-items');
  for (const [name, items] of Object.entries(constants.presets.measurements)) {
    const b = document.createElement('button');
    b.className = 'px-3 py-1.5 text-xs rounded-full border border-slate-200 hover:bg-slate-50 transition';
    b.textContent = name;
    b.addEventListener('click', () => {
      measItemsEl.querySelectorAll('input').forEach((inp) => {
        inp.checked = items.includes(inp.value);
      });
    });
    measPresetsEl.appendChild(b);
  }
  for (const m of constants.measurements) {
    const row = document.createElement('label');
    row.className = 'flex items-center gap-2 px-3 py-2 hover:bg-slate-50 transition cursor-pointer';
    row.innerHTML = `<input type="checkbox" value="${m}"/><span class="text-sm">${m}</span>`;
    measItemsEl.appendChild(row);
  }

  const disPresetsEl = document.getElementById('disease-presets');
  const disItemsEl = document.getElementById('disease-items');
  for (const [name, items] of Object.entries(constants.presets.diseases)) {
    const b = document.createElement('button');
    b.className = 'px-3 py-1.5 text-xs rounded-full border border-slate-200 hover:bg-slate-50 transition';
    b.textContent = name;
    b.addEventListener('click', () => {
      disItemsEl.querySelectorAll('input').forEach((inp) => { inp.checked = items.includes(inp.value); });
    });
    disPresetsEl.appendChild(b);
  }
  for (const d of constants.diseases) {
    const row = document.createElement('label');
    row.className = 'flex items-center gap-2 px-3 py-2 hover:bg-slate-50 transition cursor-pointer';
    row.innerHTML = `<input type="checkbox" value="${d}"/><span class="text-sm">${d}</span>`;
    disItemsEl.appendChild(row);
  }

  if (constants.presets.vqa_examples && constants.presets.vqa_examples.length) {
    const sel = document.getElementById('vqa-preset');
    sel.classList.remove('hidden');
    for (const q of constants.presets.vqa_examples) {
      const o = document.createElement('option'); o.value = q; o.textContent = q; sel.appendChild(o);
    }
    sel.addEventListener('change', () => {
      if (sel.value) document.getElementById('vqa-q').value = sel.value;
    });
  }

  function streamTask(taskId, onEvent) {
    const es = new EventSource(`/api/task/${taskId}/stream`);
    for (const kind of ['phase', 'partial', 'item', 'message', 'error', 'done']) {
      es.addEventListener(kind, (evt) => onEvent(kind, JSON.parse(evt.data)));
    }
    es.addEventListener('done', () => es.close());
    es.addEventListener('error', () => es.close());
    return es;
  }

  const reportRun = document.getElementById('report-run');
  const reportSections = document.getElementById('report-sections');
  const reportProgress = document.getElementById('report-progress');
  const reportPdf = document.getElementById('report-pdf');
  const reportDocx = document.getElementById('report-docx');

  async function runReport() {
    if (reportSections.children.length && !confirm('This will discard your current edits. Continue?')) return;
    reportSections.innerHTML = '';
    reportProgress.textContent = 'Preparing echocardiography context…';
    const r = await (await fetch(`/api/study/${sid}/task/report`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
    })).json();
    streamTask(r.task_id, (kind, data) => {
      if (kind === 'phase') {
        if (data.phase === 'inference') reportProgress.textContent = 'Generating report…';
        if (data.phase === 'preparing_context') reportProgress.textContent = 'Preparing echocardiography context…';
      } else if (kind === 'partial') {
        const wrap = document.createElement('div');
        wrap.className = 'report-section border border-slate-100 rounded-lg';
        wrap.innerHTML = `
          <div class="flex items-center justify-between px-4 py-2 text-sm text-muted">
            <span class="font-medium text-ink">${data.section}</span>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
          </div>
          <div class="px-4 pb-3 text-sm leading-relaxed whitespace-pre-wrap" data-section-body></div>
        `;
        wrap.querySelector('[data-section-body]').textContent = data.content;
        window.attachSectionEditor(wrap.querySelector('[data-section-body]'), sid, data.section);
        reportSections.appendChild(wrap);
      } else if (kind === 'done') {
        reportProgress.textContent = 'Report generated. You can edit any section above.';
        reportPdf.href = `/api/study/${sid}/report/export?format=pdf`;
        reportDocx.href = `/api/study/${sid}/report/export?format=docx`;
        reportPdf.classList.remove('hidden');
        reportDocx.classList.remove('hidden');
      } else if (kind === 'error') {
        reportProgress.textContent = 'Generation failed: ' + data.reason;
      }
    });
  }
  reportRun.addEventListener('click', runReport);

  document.getElementById('measure-run').addEventListener('click', async () => {
    const items = [...measItemsEl.querySelectorAll('input:checked')].map((i) => i.value);
    if (!items.length) return;
    const tbody = document.getElementById('measure-results');
    tbody.innerHTML = '';
    const progress = document.getElementById('measure-progress');
    const r = await (await fetch(`/api/study/${sid}/task/measurement`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ items }),
    })).json();
    streamTask(r.task_id, (kind, data) => {
      if (kind === 'phase' && data.phase === 'during') {
        progress.textContent = `Measuring ${data.name} (${data.i}/${data.n})…`;
      } else if (kind === 'item') {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td class="px-3 py-2">${data.name}</td><td class="px-3 py-2 text-right tabular-nums font-mono">${data.value ?? '—'}</td><td class="px-3 py-2 text-muted">${data.unit ?? ''}</td>`;
        tbody.appendChild(tr);
      } else if (kind === 'done') {
        progress.textContent = `${items.length} measurements complete.`;
      } else if (kind === 'error') {
        progress.textContent = 'Failed: ' + data.reason;
      }
    });
  });

  document.getElementById('disease-run').addEventListener('click', async () => {
    const items = [...disItemsEl.querySelectorAll('input:checked')].map((i) => i.value);
    if (!items.length) return;
    const results = document.getElementById('disease-results');
    results.innerHTML = '';
    const progress = document.getElementById('disease-progress');
    const r = await (await fetch(`/api/study/${sid}/task/disease`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ items }),
    })).json();
    streamTask(r.task_id, (kind, data) => {
      if (kind === 'phase' && data.phase === 'during') {
        progress.textContent = `Evaluating ${data.name} (${data.i}/${data.n})…`;
      } else if (kind === 'item') {
        const row = document.createElement('details');
        row.className = 'p-3';
        const cls = data.answer === 'yes' ? 'badge--warn'
                  : data.answer === 'no' ? 'badge--ok' : 'badge--unknown';
        row.innerHTML = `
          <summary class="flex items-center gap-3 cursor-pointer">
            <span class="badge ${cls}">${data.answer}</span>
            <span class="text-sm">${data.name}</span>
          </summary>
          <pre class="mt-2 px-3 py-2 bg-slate-50 rounded text-xs whitespace-pre-wrap">${data.raw}</pre>
        `;
        results.appendChild(row);
      } else if (kind === 'done') {
        progress.textContent = `${items.length} conditions evaluated.`;
      } else if (kind === 'error') {
        progress.textContent = 'Failed: ' + data.reason;
      }
    });
  });

  document.getElementById('vqa-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const q = document.getElementById('vqa-q').value.trim();
    if (!q) return;
    const log = document.getElementById('vqa-log');
    const ubub = document.createElement('div');
    ubub.className = 'flex justify-end';
    ubub.innerHTML = `<div class="max-w-[80%] bg-primary text-white rounded-2xl rounded-br-sm px-4 py-2 text-sm">${q}</div>`;
    log.appendChild(ubub);
    document.getElementById('vqa-q').value = '';
    const progress = document.getElementById('vqa-progress');
    progress.textContent = 'Thinking…';
    const r = await (await fetch(`/api/study/${sid}/task/vqa`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q }),
    })).json();
    streamTask(r.task_id, (kind, data) => {
      if (kind === 'message') {
        const bub = document.createElement('div');
        bub.className = 'flex';
        bub.innerHTML = `<div class="max-w-[80%] bg-slate-100 rounded-2xl rounded-bl-sm px-4 py-2 text-sm whitespace-pre-wrap">${data.content}</div>`;
        log.appendChild(bub);
      } else if (kind === 'done') {
        progress.textContent = '';
      } else if (kind === 'error') {
        progress.textContent = 'Failed: ' + data.reason;
      }
    });
  });
})();
