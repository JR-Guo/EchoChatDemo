window.attachSectionEditor = function (el, studyId, sectionName) {
  el.setAttribute('contenteditable', 'true');
  el.setAttribute('spellcheck', 'false');
  el.addEventListener('blur', async () => {
    const content = el.innerText.trim();
    const resp = await fetch(`/api/study/${studyId}/report/section`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ section: sectionName, content }),
    });
    if (resp.ok) {
      el.dataset.edited = '1';
    }
  });
};
