// Automatically route to http://127.0.0.1:3210 if loaded via VS Code Live Server (:5500) or file://
const API_BASE = window.location.port === "3210" ? "" : "http://127.0.0.1:3210";

let currentData = null;
let countdownInterval = null;
let autoRefreshTimer = 30;

// Fetch status from local server
async function fetchStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/status`);
    const data = await res.json();
    currentData = data;
    renderDashboard(data);
  } catch (e) {
    console.error("Failed to fetch status:", e);
    showToast("Error connecting to backend server (3210): " + e.message, true);
  }
}

// Format fraction to percentage
function formatPct(fraction) {
  if (fraction === undefined || fraction === null) return "--";
  return (fraction * 100).toFixed(1) + "%";
}

// Get color class based on percentage
function getColorClass(fraction) {
  if (fraction === undefined || fraction === null) return "green";
  if (fraction > 0.4) return "green";
  if (fraction > 0.15) return "yellow";
  return "red";
}

// Format duration from reset ISO timestamp
function formatTimeRemaining(isoString, remainingFraction) {
  if (
    remainingFraction !== undefined &&
    remainingFraction !== null &&
    remainingFraction >= 0.999
  ) {
    return "Ready";
  }
  if (!isoString) return "Ready";
  const target = new Date(isoString).getTime();
  const now = Date.now();
  const diff = target - now;

  if (diff <= 0) return "Ready";

  const totalSeconds = Math.floor(diff / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (hours >= 24) {
    const days = Math.floor(hours / 24);
    const remHours = hours % 24;
    return `${days}d ${remHours}h ${minutes}m ${seconds}s`;
  }

  if (hours > 0) {
    return `${hours}h ${minutes}m ${seconds}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
}

// Format exact reset calendar date and time (e.g. "Wed, Sep 16 at 2:33 PM" or "Today at 7:33 PM")
function formatResetDateTime(isoString) {
  if (!isoString) return "";
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return "";
    const now = new Date();

    const isToday = d.toDateString() === now.toDateString();
    const tomorrow = new Date(now);
    tomorrow.setDate(tomorrow.getDate() + 1);
    const isTomorrow = d.toDateString() === tomorrow.toDateString();

    const timeStr = d.toLocaleTimeString(undefined, {
      hour: "numeric",
      minute: "2-digit",
    });

    if (isToday) {
      return `Today at ${timeStr}`;
    } else if (isTomorrow) {
      return `Tomorrow at ${timeStr}`;
    } else {
      const dateStr = d.toLocaleDateString(undefined, {
        weekday: "short",
        month: "short",
        day: "numeric",
      });
      return `${dateStr} at ${timeStr}`;
    }
  } catch (e) {
    return "";
  }
}

// Sort Profile Engine & Multi-Tier Preferences
let currentSortProfile = localStorage.getItem("quota_sort_profile") || "custom";

const DEFAULT_SORT_RULES = {
  primary: { field: "claude_5h", direction: "desc" },
  secondary: { field: "claude_weekly", direction: "desc" },
};

const SORT_FIELD_LABELS = {
  claude_5h: "Claude 5-Hour %",
  claude_weekly: "Claude Weekly %",
  gemini_5h: "Gemini 5-Hour %",
  gemini_weekly: "Gemini Weekly %",
  reset_soonest: "Reset Recovery Time",
  active_first: "Active in IDE",
  oauth_expiry: "OAuth Token Remaining",
  none: "None",
};

function getCustomSortRules() {
  try {
    const raw = localStorage.getItem("quota_custom_sort_rules");
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed.primary && parsed.primary.field) return parsed;
    }
  } catch {}
  return DEFAULT_SORT_RULES;
}

function saveCustomSortRules(rules) {
  try {
    localStorage.setItem("quota_custom_sort_rules", JSON.stringify(rules));
  } catch {}
}

function getAccountExtendedMetrics(acc, currentDataRef) {
  const isCurrentActive = acc.isActive;
  const accQuotas = acc.quotas || (isCurrentActive && currentDataRef ? currentDataRef.quotas : null);
  const groups = accQuotas?.groups || [];
  const geminiGroup = groups.find((g) => g.displayName?.includes("Gemini"));
  const claudeGroup = groups.find(
    (g) => g.displayName?.includes("Claude") || g.displayName?.includes("GPT")
  );
  const gemini5h = geminiGroup?.buckets?.find((b) => b.window === "5h");
  const geminiWeekly = geminiGroup?.buckets?.find((b) => b.window === "weekly");
  const claude5h = claudeGroup?.buckets?.find((b) => b.window === "5h");
  const claudeWeekly = claudeGroup?.buckets?.find((b) => b.window === "weekly");

  const now = Date.now();
  const getEarliestActiveReset = () => {
    const resets = [claude5h?.resetTime, gemini5h?.resetTime, claudeWeekly?.resetTime, geminiWeekly?.resetTime]
      .filter(Boolean)
      .map((iso) => new Date(iso).getTime() - now)
      .filter((diff) => diff > 0);
    return resets.length > 0 ? Math.min(...resets) : Infinity;
  };

  return {
    claude_5h: claude5h?.remainingFraction ?? -1,
    claude_weekly: claudeWeekly?.remainingFraction ?? -1,
    gemini_5h: gemini5h?.remainingFraction ?? -1,
    gemini_weekly: geminiWeekly?.remainingFraction ?? -1,
    reset_soonest: getEarliestActiveReset(),
    active_first: acc.isActive ? 1 : 0,
    oauth_expiry: acc.expiryDate ? acc.expiryDate - now : 0,
  };
}

function compareAccountsByRule(a, b, rule, currentDataRef) {
  if (!rule || !rule.field || rule.field === "none") return 0;
  const metricsA = getAccountExtendedMetrics(a, currentDataRef);
  const metricsB = getAccountExtendedMetrics(b, currentDataRef);

  const valA = metricsA[rule.field] ?? 0;
  const valB = metricsB[rule.field] ?? 0;

  let diff = 0;
  if (rule.field === "reset_soonest") {
    diff = valA - valB;
    if (rule.direction === "asc") diff = -diff;
  } else {
    diff = rule.direction === "asc" ? valA - valB : valB - valA;
  }

  if (Math.abs(diff) < 0.0001) return 0;
  return diff;
}

function sortAccountsByProfile(accounts, profileId, currentDataRef) {
  if (!Array.isArray(accounts)) return accounts;

  if (profileId === "multi-tier") {
    const rules = getCustomSortRules();
    accounts.sort((a, b) => {
      // 1. Primary rule
      const primaryDiff = compareAccountsByRule(a, b, rules.primary, currentDataRef);
      if (primaryDiff !== 0) return primaryDiff;

      // 2. Secondary tie-breaker rule
      const secondaryDiff = compareAccountsByRule(a, b, rules.secondary, currentDataRef);
      if (secondaryDiff !== 0) return secondaryDiff;

      // 3. Fallback: preserve stable ID sequence
      return (a.fileId || "").localeCompare(b.fileId || "", undefined, { numeric: true, sensitivity: "base" });
    });
  } else if (profileId === "claude-highest") {
    // Highest Claude & GPT percentage first
    accounts.sort((a, b) => {
      const bFrac = getAccountExtendedMetrics(b, currentDataRef).claude_5h;
      const aFrac = getAccountExtendedMetrics(a, currentDataRef).claude_5h;
      return bFrac - aFrac;
    });
  } else if (profileId === "gemini-highest") {
    // Highest Gemini percentage first
    accounts.sort((a, b) => {
      const bFrac = getAccountExtendedMetrics(b, currentDataRef).gemini_5h;
      const aFrac = getAccountExtendedMetrics(a, currentDataRef).gemini_5h;
      return bFrac - aFrac;
    });
  } else if (profileId === "reset-soonest") {
    // Accounts recovering soonest first
    accounts.sort((a, b) => {
      const bDiff = getAccountExtendedMetrics(b, currentDataRef).reset_soonest;
      const aDiff = getAccountExtendedMetrics(a, currentDataRef).reset_soonest;
      return aDiff - bDiff;
    });
  } else if (profileId === "active-first") {
    // Active IDE account first, then standby
    accounts.sort((a, b) => {
      if (a.isActive && !b.isActive) return -1;
      if (!a.isActive && b.isActive) return 1;
      return 0;
    });
  } else {
    // "custom" - manual drag-and-drop order from localStorage / disk
    try {
      const savedOrder = JSON.parse(localStorage.getItem("quota_account_order") || "[]");
      if (Array.isArray(savedOrder) && savedOrder.length > 0) {
        accounts.sort((a, b) => {
          const idxA = savedOrder.indexOf(a.fileId);
          const idxB = savedOrder.indexOf(b.fileId);
          if (idxA === -1 && idxB === -1) return 0;
          if (idxA === -1) return 1;
          if (idxB === -1) return -1;
          return idxA - idxB;
        });
      }
    } catch (e) {
      console.warn("Failed to apply saved account order:", e);
    }
  }

  return accounts;
}

function updateSortProfileUI() {
  const profileBtns = {
    custom: document.getElementById("sortProfileCustom"),
    "claude-highest": document.getElementById("sortProfileClaude"),
    "gemini-highest": document.getElementById("sortProfileGemini"),
    "reset-soonest": document.getElementById("sortProfileReset"),
    "active-first": document.getElementById("sortProfileActive"),
    "multi-tier": document.getElementById("sortProfileMultiTier"),
  };

  Object.entries(profileBtns).forEach(([key, btn]) => {
    if (btn) {
      if (key === currentSortProfile) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    }
  });
}

function setSortProfile(profileId) {
  currentSortProfile = profileId;
  try {
    localStorage.setItem("quota_sort_profile", profileId);
  } catch {}

  updateSortProfileUI();

  if (currentData && Array.isArray(currentData.accounts)) {
    sortAccountsByProfile(currentData.accounts, currentSortProfile, currentData);
    renderCards(currentData);
    renderTable(currentData);

    const labels = {
      custom: "Custom (Drag & Drop)",
      "claude-highest": "Highest Claude Quota",
      "gemini-highest": "Highest Gemini Quota",
      "reset-soonest": "Reset Recovery Soonest",
      "active-first": "Active Account First",
      "multi-tier": "Custom Multi-Tier Rules",
    };
    showToast(`Sort Preset: ${labels[profileId] || profileId}`, false);
  }
}
window.setSortProfile = setSortProfile;

// Multi-Tier Sort Preferences Modal Handlers
function openSortPreferencesModal() {
  const modal = document.getElementById("sortPreferencesModal");
  if (!modal) return;

  const rules = getCustomSortRules();
  const primaryField = document.getElementById("sortPrimaryField");
  const primaryDir = document.getElementById("sortPrimaryDir");
  const secondaryField = document.getElementById("sortSecondaryField");
  const secondaryDir = document.getElementById("sortSecondaryDir");

  if (primaryField) primaryField.value = rules.primary.field;
  if (primaryDir) primaryDir.value = rules.primary.direction || "desc";
  if (secondaryField) secondaryField.value = rules.secondary?.field || "none";
  if (secondaryDir) secondaryDir.value = rules.secondary?.direction || "desc";

  updateModalSummary();
  modal.style.display = "flex";
}

function closeSortPreferencesModal() {
  const modal = document.getElementById("sortPreferencesModal");
  if (modal) modal.style.display = "none";
}

function handleModalBackdropClick(e) {
  if (e.target.id === "sortPreferencesModal") {
    closeSortPreferencesModal();
  }
}

function updateModalSummary() {
  const summaryEl = document.getElementById("sortRuleSummaryText");
  const pField = document.getElementById("sortPrimaryField")?.value || "claude_5h";
  const pDir = document.getElementById("sortPrimaryDir")?.value || "desc";
  const sField = document.getElementById("sortSecondaryField")?.value || "none";
  const sDir = document.getElementById("sortSecondaryDir")?.value || "desc";

  const pLabel = `${SORT_FIELD_LABELS[pField] || pField} (${pDir === "desc" ? "Highest First" : "Lowest First"})`;
  let text = `Active Rule: <strong>${pLabel}</strong>`;

  if (sField && sField !== "none") {
    const sLabel = `${SORT_FIELD_LABELS[sField] || sField} (${sDir === "desc" ? "Highest First" : "Lowest First"})`;
    text += `, then tie-break with <strong>${sLabel}</strong>.`;
  } else {
    text += ` (No secondary tie-breaker).`;
  }

  if (summaryEl) summaryEl.innerHTML = text;
}

function saveSortPreferences() {
  const pField = document.getElementById("sortPrimaryField")?.value || "claude_5h";
  const pDir = document.getElementById("sortPrimaryDir")?.value || "desc";
  const sField = document.getElementById("sortSecondaryField")?.value || "none";
  const sDir = document.getElementById("sortSecondaryDir")?.value || "desc";

  const newRules = {
    primary: { field: pField, direction: pDir },
    secondary: { field: sField, direction: sDir },
  };

  saveCustomSortRules(newRules);
  closeSortPreferencesModal();

  // Automatically activate multi-tier sort profile and re-sort
  setSortProfile("multi-tier");
}

function resetSortPreferences() {
  saveCustomSortRules(DEFAULT_SORT_RULES);
  const primaryField = document.getElementById("sortPrimaryField");
  const primaryDir = document.getElementById("sortPrimaryDir");
  const secondaryField = document.getElementById("sortSecondaryField");
  const secondaryDir = document.getElementById("sortSecondaryDir");

  if (primaryField) primaryField.value = DEFAULT_SORT_RULES.primary.field;
  if (primaryDir) primaryDir.value = DEFAULT_SORT_RULES.primary.direction;
  if (secondaryField) secondaryField.value = DEFAULT_SORT_RULES.secondary.field;
  if (secondaryDir) secondaryDir.value = DEFAULT_SORT_RULES.secondary.direction;

  updateModalSummary();
  showToast("Sort rules reset to defaults", false);
}

// Close modal on Escape key
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    closeSortPreferencesModal();
  }
});

window.openSortPreferencesModal = openSortPreferencesModal;
window.closeSortPreferencesModal = closeSortPreferencesModal;
window.handleModalBackdropClick = handleModalBackdropClick;
window.updateModalSummary = updateModalSummary;
window.saveSortPreferences = saveSortPreferences;
window.resetSortPreferences = resetSortPreferences;

// Render Dashboard
function renderDashboard(data) {
  // 0. Apply persistent sort profile
  sortAccountsByProfile(data.accounts, currentSortProfile, data);
  updateSortProfileUI();

  // 1. Connection & Token Daemon Status
  const connectionBadge = document.getElementById("connectionBadge");
  const connectionText = document.getElementById("connectionText");
  if (data.languageServer?.connected) {
    connectionBadge.className = "status-indicator online";
    connectionText.textContent = `Language Server Active (: ${data.languageServer.port})`;
  } else {
    connectionBadge.className = "status-indicator offline";
    connectionText.textContent = "Language Server Offline";
  }

  const tokenDaemonBadge = document.getElementById("tokenDaemonBadge");
  const tokenDaemonText = document.getElementById("tokenDaemonText");
  if (tokenDaemonBadge && tokenDaemonText) {
    if (data.tokenDaemon?.running) {
      tokenDaemonBadge.className = "status-indicator online";
      const totalAcc = data.tokenDaemon.accountCount || data.accounts.length;
      tokenDaemonText.textContent = `⚡ 24/7 Token Sync (${totalAcc} Active)`;
      tokenDaemonBadge.title = `24/7 Background Token Daemon Active. Cycle count: ${data.tokenDaemon.cycleCount}. Click to cycle Google OAuth tokens now!`;
    } else {
      tokenDaemonBadge.className = "status-indicator offline";
      tokenDaemonText.textContent = "Token Sync Inactive";
    }
  }

  // 2. Active Account & Counters Display
  const activeEmailEl = document.getElementById("activeAccountEmail");
  if (activeEmailEl) activeEmailEl.textContent = data.activeEmail || "None";
  const accountCountEl = document.getElementById("accountCount");
  if (accountCountEl) accountCountEl.textContent = data.accounts.length;

  // 3. Render Accounts Grid
  renderCards(data);

  // 4. Render Table Matrix
  renderTable(data);

  // 5. Restore preferred view mode
  switchView(currentView);
}

// Render Accounts Grid (Cards View)
function renderCards(data) {
  const grid = document.getElementById("accountsGrid");
  if (!grid) return;
  grid.innerHTML = "";

  if (!data.accounts || data.accounts.length === 0) {
    grid.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-muted);">
                <p>No accounts found in <code>.gemini/quota_accounts/</code>.</p>
                <p style="margin-top: 8px;">Click <strong>"+ Add Account"</strong> above to authenticate your accounts.</p>
            </div>
        `;
    return;
  }

  data.accounts.forEach((acc, index) => {
    const isCurrentActive = acc.isActive;
    const card = document.createElement("div");
    card.className = `account-card ${isCurrentActive ? "active-card" : ""}`;

    // Extract quotas for THIS specific account
    const accQuotas = acc.quotas || (isCurrentActive ? data.quotas : null);
    const accGroups = accQuotas?.groups || [];
    const accGeminiGroup = accGroups.find((g) =>
      g.displayName?.includes("Gemini"),
    );
    const accClaudeGroup = accGroups.find(
      (g) =>
        g.displayName?.includes("Claude") || g.displayName?.includes("GPT"),
    );

    const aGemini5h = accGeminiGroup?.buckets?.find((b) => b.window === "5h");
    const aGeminiWeekly = accGeminiGroup?.buckets?.find(
      (b) => b.window === "weekly",
    );
    const aClaude5h = accClaudeGroup?.buckets?.find((b) => b.window === "5h");
    const aClaudeWeekly = accClaudeGroup?.buckets?.find(
      (b) => b.window === "weekly",
    );

    // Model Quota Content
    let quotaContent = "";
    if (accGroups.length > 0) {
      const isLive = accQuotas?.isLive;
      const syncLabel = isCurrentActive
        ? "Active IDE Session"
        : isLive
          ? "● Live Direct Sync"
          : "⏱ Cached from Session";

      quotaContent = `
                <!-- Claude & GPT Block -->
                <div class="model-block">
                    <div class="model-header">
                        <div class="model-name-wrapper">
                            <span class="model-bullet claude"></span>
                            <div>
                                <div class="model-title">Claude & GPT Models</div>
                                <div class="model-subtitle">Claude 3.7 Sonnet, Opus 3.5, GPT-OSS</div>
                            </div>
                        </div>
                        <span style="font-size: 0.7rem; color: var(--text-muted);">${syncLabel}</span>
                    </div>

                    <!-- 5 Hour -->
                    <div class="quota-row">
                        <div class="quota-label-row">
                            <span class="quota-label">5-Hour Sprint Limit</span>
                            <div class="quota-value-group">
                                <span class="quota-value ${getColorClass(aClaude5h?.remainingFraction)}">
                                    ${formatPct(aClaude5h?.remainingFraction)}
                                </span>
                                <span class="reset-timer-chip ${aClaude5h?.remainingFraction < 0.999 && aClaude5h?.resetTime ? "cooling" : "ready"}" 
                                     data-reset="${aClaude5h?.resetTime || ""}"
                                     data-fraction="${aClaude5h?.remainingFraction !== undefined && aClaude5h?.remainingFraction !== null ? aClaude5h.remainingFraction : ""}">
                                    ⏱ ${formatTimeRemaining(aClaude5h?.resetTime, aClaude5h?.remainingFraction)}
                                </span>
                            </div>
                        </div>
                        <div class="progress-track">
                            <div class="progress-fill ${getColorClass(aClaude5h?.remainingFraction)}" 
                                 style="width: ${(aClaude5h?.remainingFraction || 0) * 100}%"></div>
                        </div>
                        ${
                          aClaude5h?.resetTime &&
                          aClaude5h?.remainingFraction < 0.999
                            ? `
                            <div class="quota-timer-container">
                                <div class="reset-date-chip" title="Exact reset time">
                                    🗓 ${formatResetDateTime(aClaude5h.resetTime)}
                                </div>
                            </div>
                        `
                            : ""
                        }
                    </div>

                    <!-- Weekly -->
                    <div class="quota-row" style="margin-top: 14px;">
                        <div class="quota-label-row">
                            <span class="quota-label">7-Day Baseline Cap</span>
                            <div class="quota-value-group">
                                <span class="quota-value ${getColorClass(aClaudeWeekly?.remainingFraction)}">
                                    ${formatPct(aClaudeWeekly?.remainingFraction)}
                                </span>
                                <span class="reset-timer-chip ${aClaudeWeekly?.remainingFraction < 0.999 && aClaudeWeekly?.resetTime ? "cooling" : "ready"}" 
                                     data-reset="${aClaudeWeekly?.resetTime || ""}"
                                     data-fraction="${aClaudeWeekly?.remainingFraction !== undefined && aClaudeWeekly?.remainingFraction !== null ? aClaudeWeekly.remainingFraction : ""}">
                                    ⏱ ${formatTimeRemaining(aClaudeWeekly?.resetTime, aClaudeWeekly?.remainingFraction)}
                                </span>
                            </div>
                        </div>
                        <div class="progress-track">
                            <div class="progress-fill ${getColorClass(aClaudeWeekly?.remainingFraction)}" 
                                 style="width: ${(aClaudeWeekly?.remainingFraction || 0) * 100}%"></div>
                        </div>
                        ${
                          aClaudeWeekly?.resetTime &&
                          aClaudeWeekly?.remainingFraction < 0.999
                            ? `
                            <div class="quota-timer-container">
                                <div class="reset-date-chip" title="Exact weekly baseline refresh date and time">
                                    🗓 ${formatResetDateTime(aClaudeWeekly.resetTime)}
                                </div>
                            </div>
                        `
                            : ""
                        }
                    </div>
                </div>

                <!-- Gemini Block -->
                <div class="model-block">
                    <div class="model-header">
                        <div class="model-name-wrapper">
                            <span class="model-bullet gemini"></span>
                            <div>
                                <div class="model-title">Gemini Models</div>
                                <div class="model-subtitle">Gemini 3.1 Pro, Gemini 3.8 Flash</div>
                            </div>
                        </div>
                    </div>

                    <!-- 5 Hour -->
                    <div class="quota-row">
                        <div class="quota-label-row">
                            <span class="quota-label">5-Hour Sprint Limit</span>
                            <div class="quota-value-group">
                                <span class="quota-value ${getColorClass(aGemini5h?.remainingFraction)}">
                                    ${formatPct(aGemini5h?.remainingFraction)}
                                </span>
                                <span class="reset-timer-chip ${aGemini5h?.remainingFraction < 0.999 && aGemini5h?.resetTime ? "cooling" : "ready"}" 
                                     data-reset="${aGemini5h?.resetTime || ""}"
                                     data-fraction="${aGemini5h?.remainingFraction !== undefined && aGemini5h?.remainingFraction !== null ? aGemini5h.remainingFraction : ""}">
                                    ⏱ ${formatTimeRemaining(aGemini5h?.resetTime, aGemini5h?.remainingFraction)}
                                </span>
                            </div>
                        </div>
                        <div class="progress-track">
                            <div class="progress-fill ${getColorClass(aGemini5h?.remainingFraction)}" 
                                 style="width: ${(aGemini5h?.remainingFraction || 0) * 100}%"></div>
                        </div>
                        ${
                          aGemini5h?.resetTime &&
                          aGemini5h?.remainingFraction < 0.999
                            ? `
                            <div class="quota-timer-container">
                                <div class="reset-date-chip" title="Exact reset time">
                                    🗓 ${formatResetDateTime(aGemini5h.resetTime)}
                                </div>
                            </div>
                        `
                            : ""
                        }
                    </div>

                    <!-- Weekly -->
                    <div class="quota-row" style="margin-top: 14px;">
                        <div class="quota-label-row">
                            <span class="quota-label">7-Day Baseline Cap</span>
                            <div class="quota-value-group">
                                <span class="quota-value ${getColorClass(aGeminiWeekly?.remainingFraction)}">
                                    ${formatPct(aGeminiWeekly?.remainingFraction)}
                                </span>
                                <span class="reset-timer-chip ${aGeminiWeekly?.remainingFraction < 0.999 && aGeminiWeekly?.resetTime ? "cooling" : "ready"}" 
                                     data-reset="${aGeminiWeekly?.resetTime || ""}"
                                     data-fraction="${aGeminiWeekly?.remainingFraction !== undefined && aGeminiWeekly?.remainingFraction !== null ? aGeminiWeekly.remainingFraction : ""}">
                                    ⏱ ${formatTimeRemaining(aGeminiWeekly?.resetTime, aGeminiWeekly?.remainingFraction)}
                                </span>
                            </div>
                        </div>
                        <div class="progress-track">
                            <div class="progress-fill ${getColorClass(aGeminiWeekly?.remainingFraction)}" 
                                 style="width: ${(aGeminiWeekly?.remainingFraction || 0) * 100}%"></div>
                        </div>
                        ${
                          aGeminiWeekly?.resetTime &&
                          aGeminiWeekly?.remainingFraction < 0.999
                            ? `
                            <div class="quota-timer-container">
                                <div class="reset-date-chip" title="Exact weekly baseline refresh date and time">
                                    🗓 ${formatResetDateTime(aGeminiWeekly.resetTime)}
                                </div>
                            </div>
                        `
                            : ""
                        }
                    </div>
                </div>
            `;
    } else if (acc.isDuplicate) {
      // Duplicate account alert
      quotaContent = `
                <div class="model-block" style="text-align: center; padding: 20px 16px; background: rgba(245, 158, 11, 0.05); border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.2);">
                    <p style="font-size: 0.88rem; color: #fbbf24; font-weight: 600; margin-bottom: 6px;">
                        ⚠️ Duplicate of ${acc.duplicateOf ? acc.duplicateOf.replace(".json", "") : "account1"}
                    </p>
                    <p style="font-size: 0.78rem; color: var(--text-muted); line-height: 1.4;">
                        This file contains identical credentials. Click "Log In Real Account" below to authenticate!
                    </p>
                </div>
            `;
    } else {
      // Standby account without synced quota yet
      quotaContent = `
                <div class="model-block" style="text-align: center; padding: 22px 16px;">
                    <p style="font-size: 0.88rem; color: var(--text-secondary); margin-bottom: 6px; font-weight: 600;">
                        Rate Limits Not Synced Yet
                    </p>
                    <p style="font-size: 0.78rem; color: var(--text-muted); line-height: 1.45; margin-bottom: 14px;">
                        Click <strong>"1-Click Link"</strong> for continuous 24/7 background quota sync.
                    </p>
                    <div style="display: flex; gap: 8px; justify-content: center;">
                        <button class="btn btn-secondary" style="font-size: 0.8rem; padding: 7px 14px;" onclick="triggerAddAccount('${acc.fileId}')">
                            🔗 1-Click Link
                        </button>
                    </div>
                </div>
            `;
    }

    let badgeClass = "badge-standby";
    let badgeText = "STANDBY";

    if (isCurrentActive) {
      badgeClass = "badge-active";
      badgeText = "ACTIVE IN IDE";
    } else if (acc.isDuplicate) {
      badgeClass = "badge-duplicate";
      badgeText = "DUPLICATE (ACTION REQUIRED)";
    } else if (accQuotas?.isLive) {
      badgeClass = "badge-live";
      badgeText = "STANDBY • LIVE SYNC";
    } else if (accQuotas?.isCached) {
      badgeClass = "badge-standby";
      badgeText = "STANDBY • CACHED";
    }

    const expiryMin = acc.expiryDate ? Math.round((acc.expiryDate - Date.now()) / 60000) : null;
    const tokenTagHtml = expiryMin !== null
      ? `<span style="font-family: var(--font-mono); font-size: 0.72rem; color: ${expiryMin > 10 ? 'var(--accent-emerald)' : 'var(--accent-amber)'}; margin-left: 6px;" title="Token automatically cycles when under 10m">● OAuth ${expiryMin}m</span>`
      : "";

    card.innerHTML = `
            <div class="card-top-accent ${isCurrentActive ? "" : "inactive"}"></div>
            <div class="card-header">
                <div class="account-profile">
                    <div class="account-avatar">
                        ${acc.picture ? `<img src="${acc.picture}" style="width:100%;height:100%;border-radius:50%;" alt="avatar" />` : acc.name ? acc.name.charAt(0).toUpperCase() : "U"}
                    </div>
                    <div class="account-title">
                        <h3>${acc.fileId.toUpperCase()} • ${acc.name || "Pro Account"}</h3>
                        <div class="account-email">${acc.email} ${tokenTagHtml}</div>
                    </div>
                </div>
                <span class="badge ${badgeClass}">
                    ${badgeText}
                </span>
            </div>

            ${quotaContent}

            <div class="card-actions">
                ${
                  isCurrentActive
                    ? `
                    <button class="btn btn-active-status" disabled style="width: 100%; justify-content: center; opacity: 1;">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width: 16px; height: 16px;">
                            <polyline points="20 6 9 17 4 12"></polyline>
                        </svg>
                        Currently Active in IDE
                    </button>
                `
                    : acc.isDuplicate
                      ? `
                    <button class="btn btn-primary" style="width: 100%; justify-content: center;" onclick="triggerAddAccount('${acc.fileId}')">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width: 16px; height: 16px;">
                            <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"></path>
                            <polyline points="10 17 15 12 10 7"></polyline>
                            <line x1="15" y1="12" x2="3" y2="12"></line>
                        </svg>
                        Log In Real ${acc.fileId.toUpperCase()}
                    </button>
                `
                      : `
                    <div style="display: flex; gap: 8px; width: 100%;">
                        <button class="btn btn-secondary" style="flex: 1; justify-content: center; cursor: default; opacity: 0.85;" disabled>
                            <span style="font-size: 0.85em; margin-right: 4px;">●</span> Standby Account
                        </button>
                        <button class="btn btn-secondary btn-reauth" onclick="triggerAddAccount('${acc.fileId}')" title="Re-link / Refresh OAuth for background sync">
                            🔄 Re-link
                        </button>
                    </div>
                `
                }
            </div>
        `;

    grid.appendChild(card);
  });

  // 6. Always show "+ Ingest Account N" slot to allow N accounts
  const nextSlotNum = data.accounts.length + 1;
  const emptySlot = document.createElement("div");
  emptySlot.className = "account-card empty-slot";
  emptySlot.onclick = () => triggerAddAccount(`account${nextSlotNum}`);
  emptySlot.innerHTML = `
        <div class="empty-slot-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width: 24px; height: 24px;">
                <line x1="12" y1="5" x2="12" y2="19"></line>
                <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
        </div>
        <div class="empty-slot-title">+ Ingest Account ${nextSlotNum}</div>
        <div class="empty-slot-desc">Authenticate additional Google account via isolated 1-Click OAuth</div>
    `;
  grid.appendChild(emptySlot);
}

// View switcher state & handler
let currentView = localStorage.getItem("quotaDashboardView") || "cards";

function switchView(mode) {
  currentView = mode;
  try {
    localStorage.setItem("quotaDashboardView", mode);
  } catch {}

  const cardsGrid = document.getElementById("accountsGrid");
  const tableContainer = document.getElementById("accountsTableContainer");
  const cardsBtn = document.getElementById("viewCardsBtn");
  const tableBtn = document.getElementById("viewTableBtn");

  if (mode === "table") {
    if (cardsGrid) cardsGrid.style.display = "none";
    if (tableContainer) tableContainer.style.display = "block";
    if (cardsBtn) cardsBtn.classList.remove("active");
    if (tableBtn) tableBtn.classList.add("active");
  } else {
    if (cardsGrid) cardsGrid.style.display = "grid";
    if (tableContainer) tableContainer.style.display = "none";
    if (cardsBtn) cardsBtn.classList.add("active");
    if (tableBtn) tableBtn.classList.remove("active");
  }
}
window.switchView = switchView;

// Render High-Density Quota Table Matrix
function renderTable(data) {
  const tbody = document.getElementById("quotaTableBody");
  if (!tbody) return;
  tbody.innerHTML = "";

  if (data.accounts.length === 0) {
    tbody.innerHTML = `
            <tr>
                <td colspan="6" style="text-align: center; padding: 36px; color: var(--text-muted);">
                    No accounts found in <code>.gemini/quota_accounts/</code>. Click "+ Add Account" to start.
                </td>
            </tr>
        `;
    return;
  }

  data.accounts.forEach((acc) => {
    const isCurrentActive = acc.isActive;
    const accQuotas = acc.quotas || (isCurrentActive ? data.quotas : null);
    const accGroups = accQuotas?.groups || [];
    const accGemini = accGroups.find((g) => g.displayName?.includes("Gemini"));
    const accClaude = accGroups.find(
      (g) =>
        g.displayName?.includes("Claude") || g.displayName?.includes("GPT"),
    );

    const gemini5h = accGemini?.buckets?.find((b) => b.window === "5h");
    const geminiWeekly = accGemini?.buckets?.find((b) => b.window === "weekly");
    const claude5h = accClaude?.buckets?.find((b) => b.window === "5h");
    const claudeWeekly = accClaude?.buckets?.find((b) => b.window === "weekly");

    const tr = document.createElement("tr");
    tr.setAttribute("data-file-id", acc.fileId);
    tr.setAttribute("draggable", "true");
    if (isCurrentActive) tr.className = "row-active";

    const hasQuota = accGroups.length > 0;

    let badgeHtml = "";
    if (isCurrentActive) {
      badgeHtml = `<span class="badge badge-active">ACTIVE IN IDE</span>`;
    } else if (acc.isDuplicate) {
      badgeHtml = `<span class="badge badge-duplicate">DUPLICATE</span>`;
    } else if (accQuotas?.isLive) {
      badgeHtml = `<span class="badge badge-live">LIVE SYNC</span>`;
    } else if (accQuotas?.isCached) {
      badgeHtml = `<span class="badge badge-standby">CACHED</span>`;
    } else {
      badgeHtml = `<span class="badge badge-standby">STANDBY</span>`;
    }

    const renderMetricCell = (bucket, showTimer = false) => {
      if (!hasQuota || !bucket) {
        return `<td style="color: var(--text-muted); font-size: 0.8rem; text-align: center; background: rgba(0,0,0,0.15);">--</td>`;
      }
      const pctStr = formatPct(bucket.remainingFraction);
      const colorClass = getColorClass(bucket.remainingFraction);
      const widthPct = Math.max(
        0,
        Math.min(100, (bucket.remainingFraction || 0) * 100),
      );
      const resetDateStr =
        bucket.resetTime && bucket.remainingFraction < 0.999
          ? formatResetDateTime(bucket.resetTime)
          : "";
      const isCooling = bucket.remainingFraction < 0.999 && bucket.resetTime;

      return `
                <td class="cell-metric">
                    <div class="metric-header">
                        <span class="metric-val ${colorClass}">${pctStr}</span>
                        ${
                          showTimer
                            ? `
                            <span class="mini-timer reset-timer-chip ${isCooling ? "cooling" : "ready"}" 
                                 data-reset="${bucket.resetTime || ""}"
                                 data-fraction="${bucket.remainingFraction !== undefined && bucket.remainingFraction !== null ? bucket.remainingFraction : ""}">
                                ⏱ ${formatTimeRemaining(bucket.resetTime, bucket.remainingFraction)}
                            </span>
                        `
                            : ""
                        }
                    </div>
                    <div class="mini-progress">
                        <div class="mini-fill ${colorClass}" style="width: ${widthPct}%"></div>
                    </div>
                    ${
                      showTimer && resetDateStr
                        ? `
                        <div class="mini-date" title="Exact reset date and time: ${resetDateStr}">
                            🗓 ${resetDateStr}
                        </div>
                    `
                        : ""
                    }
                </td>
            `;
    };

    const tExpiryMin = acc.expiryDate ? Math.round((acc.expiryDate - Date.now()) / 60000) : null;
    const tTokenHtml = tExpiryMin !== null
      ? `<span style="font-family: var(--font-mono); font-size: 0.72rem; color: ${tExpiryMin > 10 ? 'var(--accent-emerald)' : 'var(--accent-amber)'}; margin-left: 6px;">● OAuth ${tExpiryMin}m</span>`
      : "";

    tr.innerHTML = `
            <td>
                <div class="cell-account">
                    <span class="drag-handle" title="Drag to reorder account row">
                        <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
                            <circle cx="8.5" cy="5" r="1.6" />
                            <circle cx="15.5" cy="5" r="1.6" />
                            <circle cx="8.5" cy="12" r="1.6" />
                            <circle cx="15.5" cy="12" r="1.6" />
                            <circle cx="8.5" cy="19" r="1.6" />
                            <circle cx="15.5" cy="19" r="1.6" />
                        </svg>
                    </span>
                    <div class="account-avatar" style="width: 36px; height: 36px; font-size: 0.85rem;">
                        ${acc.picture ? `<img src="${acc.picture}" style="width:100%;height:100%;border-radius:50%;" alt="avatar" />` : acc.name ? acc.name.charAt(0).toUpperCase() : "U"}
                    </div>
                    <div>
                        <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 2px;">
                            <strong style="color: var(--text-primary); font-size: 0.9rem;">${acc.name || acc.fileId.toUpperCase()}</strong>
                            ${badgeHtml}
                        </div>
                        <div style="font-family: var(--font-mono); font-size: 0.78rem; color: var(--text-secondary);">${acc.email} ${tTokenHtml}</div>
                    </div>
                </div>
            </td>
            ${renderMetricCell(gemini5h, true)}
            ${renderMetricCell(geminiWeekly, true)}
            ${renderMetricCell(claude5h, true)}
            ${renderMetricCell(claudeWeekly, true)}
            <td class="table-action-cell">
                ${
                  isCurrentActive
                    ? `
                    <span class="badge badge-active" style="font-size: 0.8rem; padding: 5px 12px; display: inline-flex; align-items: center; gap: 5px;">
                        <span style="font-size: 0.9em;">●</span> Active in IDE
                    </span>
                `
                    : acc.isDuplicate
                      ? `
                    <button class="btn btn-primary" style="font-size: 0.76rem; padding: 6px 10px;" onclick="triggerAddAccount('${acc.fileId}')">
                        Log In Real
                    </button>
                `
                      : `
                    <div style="display: inline-flex; align-items: center; gap: 6px;">
                        <span class="badge badge-standby" style="font-size: 0.8rem; padding: 5px 10px;">
                            Standby
                        </span>
                        <button class="btn btn-secondary btn-reauth" style="font-size: 0.76rem; padding: 5px 8px;" onclick="triggerAddAccount('${acc.fileId}')" title="Re-link / Refresh OAuth">
                            🔄
                        </button>
                    </div>
                `
                }
            </td>
        `;

    tbody.appendChild(tr);
  });

  // Always offer an inline "+ Ingest Account N" row to support N accounts
  const nextNum = data.accounts.length + 1;
  const addTr = document.createElement("tr");
  addTr.style.background = "rgba(255, 255, 255, 0.015)";
  addTr.setAttribute("draggable", "false");
  addTr.innerHTML = `
        <td colspan="6" style="text-align: center; padding: 14px;">
            <button class="btn btn-secondary" style="font-size: 0.8rem; padding: 6px 16px; border-style: dashed;" onclick="triggerAddAccount('account${nextNum}')">
                + Ingest Account ${nextNum} via 1-Click OAuth
            </button>
        </td>
    `;
  tbody.appendChild(addTr);

  // Initialize or rebind drag-and-drop table listeners
  setupTableDragAndDrop();
}

// Drag and Drop Table Row Reordering
let draggedRow = null;

function setupTableDragAndDrop() {
  const tbody = document.getElementById("quotaTableBody");
  if (!tbody || tbody.dataset.dragInitialized) return;
  tbody.dataset.dragInitialized = "true";

  tbody.addEventListener("dragstart", (e) => {
    // Do not initiate drag if user interacted with a button, link, or input
    if (e.target.closest("button, a, input, select")) {
      e.preventDefault();
      return;
    }

    const row = e.target.closest('tr[draggable="true"]');
    if (!row) return;

    draggedRow = row;
    row.classList.add("dragging");
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", row.getAttribute("data-file-id") || "");
  });

  tbody.addEventListener("dragend", () => {
    tbody.querySelectorAll("tr").forEach((r) => {
      r.classList.remove("dragging", "drag-over-top", "drag-over-bottom");
    });
    draggedRow = null;
  });

  tbody.addEventListener("dragover", (e) => {
    e.preventDefault();
    if (!draggedRow) return;
    e.dataTransfer.dropEffect = "move";

    const targetRow = e.target.closest("tr");
    if (!targetRow || !tbody.contains(targetRow)) return;

    // Reset highlights on all other rows
    tbody.querySelectorAll("tr").forEach((r) => {
      if (r !== targetRow) {
        r.classList.remove("drag-over-top", "drag-over-bottom");
      }
    });

    if (targetRow === draggedRow) return;

    // Hovering over the bottom "+ Ingest Account N" row
    if (!targetRow.getAttribute("data-file-id")) {
      targetRow.classList.add("drag-over-top");
      targetRow.classList.remove("drag-over-bottom");
      return;
    }

    // Determine upper half vs lower half
    const rect = targetRow.getBoundingClientRect();
    const offset = e.clientY - rect.top;
    const isTop = offset < rect.height / 2;
    targetRow.classList.toggle("drag-over-top", isTop);
    targetRow.classList.toggle("drag-over-bottom", !isTop);
  });

  tbody.addEventListener("dragleave", (e) => {
    if (!tbody.contains(e.relatedTarget)) {
      tbody.querySelectorAll("tr").forEach((r) => {
        r.classList.remove("drag-over-top", "drag-over-bottom");
      });
    }
  });

  tbody.addEventListener("drop", (e) => {
    e.preventDefault();
    if (!draggedRow) return;

    const targetRow = e.target.closest("tr");
    if (!targetRow || !tbody.contains(targetRow) || targetRow === draggedRow) {
      tbody.querySelectorAll("tr").forEach((r) => {
        r.classList.remove("dragging", "drag-over-top", "drag-over-bottom");
      });
      draggedRow = null;
      return;
    }

    // If dropped on the bottom "+ Ingest Account N" row, insert before it
    if (!targetRow.getAttribute("data-file-id")) {
      tbody.insertBefore(draggedRow, targetRow);
    } else {
      const isTop = targetRow.classList.contains("drag-over-top");
      if (isTop) {
        tbody.insertBefore(draggedRow, targetRow);
      } else {
        tbody.insertBefore(draggedRow, targetRow.nextSibling);
      }
    }

    tbody.querySelectorAll("tr").forEach((r) => {
      r.classList.remove("dragging", "drag-over-top", "drag-over-bottom");
    });
    draggedRow = null;

    saveTableRowOrder(tbody);
  });
}

function saveTableRowOrder(tbody) {
  const rows = tbody.querySelectorAll("tr[data-file-id]");
  const newOrder = Array.from(rows)
    .map((r) => r.getAttribute("data-file-id"))
    .filter(Boolean);

  try {
    localStorage.setItem("quota_account_order", JSON.stringify(newOrder));
  } catch (err) {
    console.error("Failed to save row order to localStorage:", err);
  }

  // Persist to server disk so order survives cache wipes, incognito, and PC restarts
  try {
    fetch(`${API_BASE}/api/accounts/reorder`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ order: newOrder }),
    }).catch(() => {});
  } catch {}

  // Switching order via drag & drop automatically activates "custom" preset
  if (currentSortProfile !== "custom") {
    currentSortProfile = "custom";
    try {
      localStorage.setItem("quota_sort_profile", "custom");
    } catch {}
    updateSortProfileUI();
  }

  // Synchronize currentData.accounts in-memory order
  if (currentData && Array.isArray(currentData.accounts)) {
    currentData.accounts.sort((a, b) => {
      const idxA = newOrder.indexOf(a.fileId);
      const idxB = newOrder.indexOf(b.fileId);
      if (idxA === -1 && idxB === -1) return 0;
      if (idxA === -1) return 1;
      if (idxB === -1) return -1;
      return idxA - idxB;
    });
    // Immediately re-render Cards View so views remain identical
    renderCards(currentData);
  }

  showToast("Custom account order updated", false);
}

// Trigger Add Account Flow from UI
async function triggerAddAccount(targetSlot = null) {
  try {
    const targetLabel = targetSlot ? targetSlot.toUpperCase() : "Account";
    showToast(`Opening Google login for ${targetLabel} in browser...`, false);
    const res = await fetch(`${API_BASE}/api/auth/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target: targetSlot }),
    });
    const data = await res.json();
    if (data.success && data.authUrl) {
      // Open Google Account Chooser in a new browser tab immediately
      window.open(data.authUrl, "_blank");
      showToast(
        `Google login opened in a new tab! Select your ${targetLabel}.`,
        false,
      );
      let attempts = 0;
      const checkInt = setInterval(() => {
        fetchStatus();
        if (++attempts > 20) clearInterval(checkInt);
      }, 2500);
    } else {
      showToast(
        "Failed to launch auth: " + (data.error || "Unknown error"),
        true,
      );
    }
  } catch (e) {
    showToast("Auth launch error: " + e.message, true);
  }
}

// Trigger Manual Background Token Cycle
async function triggerTokenCycle() {
  try {
    showToast("Cycling Google OAuth tokens across all accounts...", false);
    const res = await fetch(`${API_BASE}/api/token-daemon/cycle`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force: false })
    });
    const data = await res.json();
    if (data.success) {
      showToast(`Token cycle complete! ${data.result?.results?.length || 0} accounts synchronized.`, false);
      fetchStatus();
    } else {
      showToast("Token cycle error: " + (data.error || "Unknown"), true);
    }
  } catch (e) {
    showToast("Token cycle request failed: " + e.message, true);
  }
}

// Switch Active Account in IDE
// Account switching is manual directly within Antigravity IDE
function switchAccount(fileId) {
  showToast(
    "Account switching is manual. Switch account directly in Antigravity IDE.",
    false,
  );
}

// Toast notification helper
function showToast(msg, isError = false) {
  const toast = document.getElementById("toast");
  toast.textContent = msg;
  toast.style.borderColor = isError
    ? "var(--accent-rose)"
    : "var(--accent-emerald)";
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 3500);
}

// Event Listeners
document.getElementById("refreshBtn").addEventListener("click", () => {
  fetchStatus();
  showToast("Quotas refreshed!", false);
});

// Auto-refresh countdown every second
setInterval(() => {
  autoRefreshTimer--;
  const timerSpan = document.getElementById("autoTimer");
  if (timerSpan) timerSpan.textContent = autoRefreshTimer;

  // Update ticking chips
  document.querySelectorAll(".reset-timer-chip").forEach((chip) => {
    const resetIso = chip.getAttribute("data-reset");
    const fracStr = chip.getAttribute("data-fraction");
    const fraction =
      fracStr !== "" && fracStr !== null && fracStr !== undefined
        ? parseFloat(fracStr)
        : undefined;
    if (resetIso) {
      const formatted = formatTimeRemaining(resetIso, fraction);
      if (formatted) {
        chip.textContent = `⏱ ${formatted}`;
        if (formatted === "Ready") {
          chip.classList.remove("cooling");
          chip.classList.add("ready");
        }
      }
    }
  });

  if (autoRefreshTimer <= 0) {
    autoRefreshTimer = 30;
    fetchStatus();
  }
}, 1000);

// Initial load
fetchStatus();
