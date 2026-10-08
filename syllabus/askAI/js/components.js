// <master-prompt-card>
class MasterPromptCardElement extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.isOpen = false;
    this.promptText = '';
    this.label = '';
    this.icon = '📜';
  }

  setPrompt(text, label = '', icon = '📜', isOpen = false) {
    this.promptText = text || '(no master prompt template found)';
    this.label = label;
    this.icon = icon || '📜';
    this.isOpen = isOpen;
    this.render();
  }

  toggle() {
    this.isOpen = !this.isOpen;
    this.render();
    this.dispatchEvent(new CustomEvent('toggle-master-prompt', { detail: { isOpen: this.isOpen }, bubbles: true, composed: true }));
  }

  render() {
    this.shadowRoot.innerHTML = `
      <style>
        :host { display: block; }
        .card { background-color: rgba(15, 23, 42, 0.9); border: 1px solid #1e293b; border-radius: 1rem; overflow: hidden; font-family: Inter, sans-serif; }
        .header { display: flex; align-items: center; justify-content: space-between; padding: 0.875rem 1.25rem; border-bottom: 1px solid rgba(30, 41, 59, 0.8); background-color: rgba(15, 23, 42, 0.6); cursor: pointer; user-select: none; }
        .title { font-size: 0.8125rem; font-weight: 600; color: #818cf8; display: flex; align-items: center; gap: 0.5rem; }
        .toggle { font-size: 0.75rem; color: #94a3b8; }
        .body { padding: 1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #cbd5e1; line-height: 1.625; white-space: pre-wrap; word-break: break-word; max-height: 20rem; overflow-y: auto; background-color: rgba(2, 6, 23, 0.6); }
        .hidden { display: none; }
        .custom-scroll::-webkit-scrollbar { width: 6px; height: 6px; }
        .custom-scroll::-webkit-scrollbar-track { background: transparent; }
        .custom-scroll::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
      </style>
      <div class="card">
        <div class="header" id="toggle-btn">
          <span class="title"><span>${this.icon}</span> Master Prompt Template — <strong style="color: #ffffff;">${this.label || 'Selected Topic'}</strong></span>
          <span class="toggle">${this.isOpen ? '▼ collapse' : '▶ expand'}</span>
        </div>
        <div class="body custom-scroll ${this.isOpen ? '' : 'hidden'}" id="body-text"></div>
      </div>
    `;

    const bodyText = this.shadowRoot.getElementById('body-text');
    if (bodyText) bodyText.textContent = this.promptText;

    const toggleBtn = this.shadowRoot.getElementById('toggle-btn');
    if (toggleBtn) {
      toggleBtn.onclick = () => this.toggle();
    }
  }
}
customElements.define('master-prompt-card', MasterPromptCardElement);

// <prompt-editor>
class PromptEditorElement extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
  }

  connectedCallback() {
    this.render();
  }

  getValue() {
    const textarea = this.shadowRoot.getElementById('section-input');
    return textarea ? textarea.value : '';
  }

  setValue(val) {
    const textarea = this.shadowRoot.getElementById('section-input');
    if (textarea) textarea.value = val;
  }

  setCharCount(str) {
    const countEl = this.shadowRoot.getElementById('char-count');
    if (countEl) countEl.textContent = str;
  }

  render() {
    this.shadowRoot.innerHTML = `
      <style>
        :host { display: block; margin-top: 1.25rem; }
        .card { background-color: rgba(15, 23, 42, 0.9); border: 1px solid #1e293b; border-radius: 1rem; overflow: hidden; font-family: Inter, sans-serif; margin-bottom: 1.25rem; }
        .header { display: flex; align-items: center; justify-content: space-between; padding: 0.875rem 1.25rem; border-bottom: 1px solid rgba(30, 41, 59, 0.8); background-color: rgba(15, 23, 42, 0.6); }
        .title { font-size: 0.75rem; font-weight: 600; color: #e2e8f0; }
        .count { font-size: 0.75rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace; }
        textarea { width: 100%; min-height: 140px; background-color: rgba(2, 6, 23, 0.8); border: 0; outline: none; color: #f1f5f9; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; padding: 1rem; resize: vertical; box-sizing: border-box; }
        textarea::placeholder { color: #475569; }
        textarea:focus { background-color: #020617; }
        .actions { display: flex; gap: 0.75rem; align-items: center; }
        .btn-primary { flex: 1; display: flex; align-items: center; justify-content: center; gap: 0.5rem; padding: 0.75rem 1.5rem; border-radius: 0.75rem; font-weight: 600; font-size: 0.875rem; color: #ffffff; background: linear-gradient(135deg, #4f46e5, #6366f1, #9333ea); border: none; cursor: pointer; transition: all 0.2s; box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.2); }
        .btn-primary:active { transform: scale(0.99); }
        .btn-secondary { display: flex; align-items: center; justify-content: center; gap: 0.5rem; padding: 0.75rem 1.25rem; border-radius: 0.75rem; font-weight: 600; font-size: 0.75rem; color: #cbd5e1; background-color: #0f172a; border: 1px solid rgba(51, 65, 85, 0.8); cursor: pointer; transition: all 0.2s; }
        .btn-secondary:hover { background-color: #1e293b; }
      </style>
      <div class="card">
        <div class="header">
          <span class="title">📝 Paste Your Section Here</span>
          <small class="count" id="char-count">— characters in output</small>
        </div>
        <textarea id="section-input"
          placeholder="Paste or type the section number/text here, e.g.: &#10;7. Autograd &amp; Automatic Differentiation&#10;&#10;The full prompt is built + copied automatically on paste."
        ></textarea>
      </div>
      <div class="actions">
        <button class="btn-primary" id="copy-btn">📋 Build &amp; Copy Full Prompt</button>
        <button class="btn-secondary" id="clear-btn">🗑️ Clear</button>
      </div>
    `;

    const input = this.shadowRoot.getElementById('section-input');
    const copyBtn = this.shadowRoot.getElementById('copy-btn');
    const clearBtn = this.shadowRoot.getElementById('clear-btn');

    if (input) {
      input.oninput = () => this.dispatchEvent(new CustomEvent('editor-input', { detail: { value: input.value }, bubbles: true, composed: true }));
      input.onpaste = () => {
        setTimeout(() => {
          this.dispatchEvent(new CustomEvent('editor-paste', { detail: { value: input.value }, bubbles: true, composed: true }));
        }, 60);
      };
    }

    if (copyBtn) {
      copyBtn.onclick = () => this.dispatchEvent(new CustomEvent('copy-click', { bubbles: true, composed: true }));
    }
    if (clearBtn) {
      clearBtn.onclick = () => this.dispatchEvent(new CustomEvent('clear-click', { bubbles: true, composed: true }));
    }
  }
}
customElements.define('prompt-editor', PromptEditorElement);

// <prompt-preview>
class PromptPreviewElement extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
  }

  connectedCallback() {
    this.render();
  }

  updatePreview(htmlSnippet, badgeText) {
    const previewBody = this.shadowRoot.getElementById('preview-body');
    const charBadge = this.shadowRoot.getElementById('char-badge');
    if (previewBody) previewBody.innerHTML = htmlSnippet || '<span style="color:#475569;">Start typing or paste a section to preview the output…</span>';
    if (charBadge) charBadge.textContent = badgeText || '—';
  }

  render() {
    this.shadowRoot.innerHTML = `
      <style>
        :host { display: block; margin-top: 1.25rem; }
        .card { background-color: rgba(15, 23, 42, 0.9); border: 1px solid #1e293b; border-radius: 1rem; overflow: hidden; font-family: Inter, sans-serif; }
        .header { display: flex; align-items: center; justify-content: space-between; padding: 0.875rem 1.25rem; border-bottom: 1px solid rgba(30, 41, 59, 0.8); background-color: rgba(15, 23, 42, 0.6); }
        .title { font-size: 0.75rem; font-weight: 600; color: #a5b4fc; display: flex; align-items: center; gap: 0.5rem; }
        .badge { font-size: 0.75rem; color: #64748b; font-family: 'JetBrains Mono', monospace; }
        .body { padding: 1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94a3b8; line-height: 1.625; max-height: 13rem; overflow-y: auto; white-space: pre-wrap; word-break: break-word; background-color: rgba(2, 6, 23, 0.6); }
        .custom-scroll::-webkit-scrollbar { width: 6px; height: 6px; }
        .custom-scroll::-webkit-scrollbar-track { background: transparent; }
        .custom-scroll::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
      </style>
      <div class="card">
        <div class="header">
          <span class="title">👁️ Preview (first 15 lines of output)</span>
          <span class="badge" id="char-badge">—</span>
        </div>
        <div class="body custom-scroll" id="preview-body">
          <span style="color:#475569;">Start typing or paste a section to preview the output…</span>
        </div>
      </div>
    `;
  }
}
customElements.define('prompt-preview', PromptPreviewElement);
