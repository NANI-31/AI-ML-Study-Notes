// ─── 1. PROMPT STORE (Data & State Management) ──────────────────────
class PromptStore {
  constructor(topics, folderLabels) {
    this.topics = topics || [];
    this.folderLabels = folderLabels || {};
    this.activeTopic = null;
    this.searchQuery = "";
    this.roadmapOpen = false;
  }

  setSearchQuery(query) {
    this.searchQuery = (query || "").toLowerCase().trim();
  }

  getFilteredFlatList() {
    const query = this.searchQuery;
    const groups = {};
    this.topics.forEach(t => {
      if (!query || t.label.toLowerCase().includes(query) || t.id.toLowerCase().includes(query)) {
        if (!groups[t.folder]) groups[t.folder] = [];
        groups[t.folder].push(t);
      }
    });

    const flatItems = [];
    Object.entries(groups).forEach(([f, list]) => {
      if (list.length === 0) return;
      flatItems.push({ type: 'header', text: this.folderLabels[f] || f, height: 32 });
      list.forEach(t => {
        flatItems.push({ type: 'item', topic: t, height: 36 });
      });
    });

    return flatItems;
  }

  selectTopic(id) {
    this.activeTopic = this.topics.find(t => t.id === id) || null;
    return this.activeTopic;
  }

  buildPrompt(sectionText) {
    if (!this.activeTopic || !sectionText.trim()) return "";
    return this.activeTopic.prompt.split(this.activeTopic.placeholder).join(sectionText.trim());
  }
}

// ─── 4. TOAST NOTIFICATION CONTROLLER ─────────────────────────────
class ToastManager {
  constructor() {
    this.timer = null;
  }

  showSuccess() {
    const toast = document.getElementById("toast");
    const editorComp = document.getElementById("prompt-editor");
    let btn = null;
    if (editorComp && editorComp.shadowRoot) {
      btn = editorComp.shadowRoot.getElementById("copy-btn");
    }

    if (toast) {
      toast.classList.remove("translate-y-20", "opacity-0");
      toast.classList.add("translate-y-0", "opacity-100");
      clearTimeout(this.timer);
      this.timer = setTimeout(() => {
        toast.classList.remove("translate-y-0", "opacity-100");
        toast.classList.add("translate-y-20", "opacity-0");
      }, 2800);
    }
    if (btn) {
      btn.style.background = "linear-gradient(135deg, #059669, #10b981)";
      btn.innerHTML = "✅ Copied!";
      setTimeout(() => { 
        btn.style.background = "";
        btn.innerHTML = "📋 Build &amp; Copy Full Prompt"; 
      }, 2500);
    }
  }
}
