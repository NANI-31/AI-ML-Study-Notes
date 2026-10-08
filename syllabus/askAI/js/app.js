// ─── 5. MAIN APPLICATION CONTROLLER ───────────────────────────────
class App {
  constructor() {
    this.store = new PromptStore(TOPICS, FOLDER_LABELS);
    this.toast = new ToastManager();
    this.virtualList = null;
  }

  init() {
    const container = document.getElementById("sidebar-list");
    this.virtualList = new VirtualizedList(
      container,
      () => this.store.getFilteredFlatList(),
      (item) => this.renderSidebarRow(item)
    );

    this.virtualList.render();

    const panel = document.getElementById("panel");
    if (panel) {
      panel.addEventListener('toggle-master-prompt', (e) => {
        this.store.roadmapOpen = e.detail.isOpen;
      });
      panel.addEventListener('editor-input', () => this.onInput());
      panel.addEventListener('editor-paste', () => {
        const editor = document.getElementById('prompt-editor');
        const val = editor ? editor.getValue() : '';
        this.updatePreview(val);
        if (val.trim()) this.doCopy(val);
      });
      panel.addEventListener('copy-click', () => this.copyPrompt());
      panel.addEventListener('clear-click', () => this.clearInput());
    }
  }

  onSearchInput() {
    const query = document.getElementById("search-input")?.value || "";
    this.store.setSearchQuery(query);
    this.virtualList.render();
  }

  renderSidebarRow(item) {
    if (item.type === 'header') {
      return `<div class="px-3 pt-2 text-[10px] font-bold text-slate-400 tracking-wider uppercase truncate">${esc(item.text)}</div>`;
    }
    const t = item.topic;
    const isActive = this.store.activeTopic?.id === t.id;
    return `
      <div class="flex items-center gap-2.5 px-3 py-1.5 rounded-lg cursor-pointer text-xs transition-all border-l-2 ${
        isActive 
          ? "bg-indigo-600/20 text-white font-medium border-indigo-500 shadow-sm" 
          : "text-slate-300 hover:bg-indigo-600/10 hover:text-white border-transparent"
      }" onclick="app.selectTopic('${t.id}')">
        <span class="text-sm">${t.icon}</span><span class="truncate">${t.label}</span>
      </div>
    `;
  }

  selectTopic(id) {
    this.store.selectTopic(id);
    this.virtualList.render();
    this.renderPanel();
  }

  renderPanel() {
    const t = this.store.activeTopic;
    const panel = document.getElementById("panel");
    if (!t || !panel) return;

    panel.innerHTML = `
      <master-prompt-card id="master-prompt-card"></master-prompt-card>
      <prompt-editor id="prompt-editor"></prompt-editor>
      <prompt-preview id="prompt-preview"></prompt-preview>
    `;

    const mpCard = document.getElementById('master-prompt-card');
    if (mpCard) {
      mpCard.setPrompt(t.prompt, t.label, t.icon, this.store.roadmapOpen);
    }
  }

  onInput() {
    const editor = document.getElementById('prompt-editor');
    const val = editor ? editor.getValue() : '';
    this.updatePreview(val);
  }

  updatePreview(val) {
    const editor = document.getElementById('prompt-editor');
    const preview = document.getElementById('prompt-preview');
    if (!preview) return;

    if (!val.trim()) {
      if (editor) editor.setCharCount("— characters in output");
      preview.updatePreview('', '—');
      return;
    }

    const full = this.store.buildPrompt(val);
    const chars = full.length.toLocaleString();
    if (editor) editor.setCharCount(`${chars} characters in output`);

    const lines = full.split("\n");
    const snippet = lines.slice(0, 15).join("\n");
    const htmlSnippet = esc(snippet) +
      (lines.length > 15 ? `\n<span style="color:#818cf8; font-style:italic;">…[${lines.length - 15} more lines]</span>` : "");
    preview.updatePreview(htmlSnippet, `${chars} chars`);
  }

  copyPrompt() {
    const editor = document.getElementById('prompt-editor');
    const val = editor ? editor.getValue() : '';
    if (!val.trim()) return;
    this.doCopy(val);
  }

  doCopy(val) {
    const full = this.store.buildPrompt(val);
    navigator.clipboard.writeText(full).then(() => this.toast.showSuccess(), () => {
      const ta = document.createElement("textarea");
      ta.value = full; document.body.appendChild(ta); ta.select();
      document.execCommand("copy"); document.body.removeChild(ta);
      this.toast.showSuccess();
    });
  }

  clearInput() {
    const editor = document.getElementById('prompt-editor');
    if (editor) editor.setValue("");
    this.updatePreview("");
  }
}

function esc(s) {
  return (s || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

// ─── INITIALIZATION ───────────────────────────────────────────────
const app = new App();
document.addEventListener("DOMContentLoaded", () => app.init());
if (document.readyState === "complete" || document.readyState === "interactive") {
  app.init();
}
