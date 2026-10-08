// ─── 2. VIRTUALIZED LIST CONTROLLER (60 FPS Performance) ──────────
class VirtualizedList {
  constructor(container, getItemsFn, renderRowFn) {
    this.container = container;
    this.getItemsFn = getItemsFn;
    this.renderRowFn = renderRowFn;
    this.ticking = false;

    if (this.container) {
      this.container.addEventListener('scroll', () => {
        if (!this.ticking) {
          requestAnimationFrame(() => {
            this.render();
            this.ticking = false;
          });
          this.ticking = true;
        }
      });
    }
  }

  render() {
    if (!this.container) return;
    const items = this.getItemsFn();
    if (!items || items.length === 0) {
      this.container.innerHTML = `<div class="px-4 py-8 text-center text-xs text-slate-500">No topics match search</div>`;
      return;
    }

    let currentY = 0;
    const positions = items.map(item => {
      const top = currentY;
      currentY += item.height;
      return { top, height: item.height, item };
    });
    const totalHeight = currentY;

    const scrollTop = this.container.scrollTop || 0;
    const clientHeight = this.container.clientHeight || 600;
    const buffer = 120; // Virtual scroll buffer margin in pixels

    const visibleItems = positions.filter(pos => 
      (pos.top + pos.height) >= (scrollTop - buffer) && pos.top <= (scrollTop + clientHeight + buffer)
    );

    let html = `<div style="height:${totalHeight}px; position:relative; width:100%;">`;
    visibleItems.forEach(pos => {
      html += `<div style="position:absolute; top:${pos.top}px; left:0; right:0; height:${pos.height}px;">
        ${this.renderRowFn(pos.item)}
      </div>`;
    });
    html += `</div>`;

    this.container.innerHTML = html;
  }
}
