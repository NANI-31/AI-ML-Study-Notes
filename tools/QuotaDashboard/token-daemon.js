/**
 * Automated Background Token Refresh Daemon (token-daemon.js)
 *
 * Lightweight background worker to automatically cycle Google OAuth refresh tokens
 * before their expiration window to ensure 24/7 background sync across all N accounts.
 *
 * Features:
 * - Automatically discovers all account*.json files in ~/.gemini/quota_accounts/
 * - Proactively refreshes tokens 10 minutes before expiration
 * - Keeps active ~/.gemini/oauth_creds.json in sync
 * - Automatically warms CloudCode quota cache (account*_quota.json) after refresh
 * - Exposes health metrics and manual cycle triggers
 */

const fs = require("fs");
const path = require("path");
const https = require("https");

// Official Antigravity CloudCode OAuth Credentials
const CLIENT_ID = process.env.GOOGLE_CLIENT_ID || "";
const CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET || "";

const USER_PROFILE = process.env.USERPROFILE || "C:\\Users\\Chowd";
const ACCOUNTS_DIR = path.join(USER_PROFILE, ".gemini", "quota_accounts");
const OAUTH_FILE = path.join(USER_PROFILE, ".gemini", "oauth_creds.json");

// Refresh threshold: 10 minutes before expiration
const REFRESH_THRESHOLD_MS = 10 * 60 * 1000;
// Scan interval: every 2 minutes
const DEFAULT_CYCLE_INTERVAL_MS = 2 * 60 * 1000;

class TokenDaemon {
    constructor(options = {}) {
        this.cycleIntervalMs = options.cycleIntervalMs || DEFAULT_CYCLE_INTERVAL_MS;
        this.refreshThresholdMs = options.refreshThresholdMs || REFRESH_THRESHOLD_MS;
        this.timer = null;
        this.isRunning = false;
        this.isCycling = false;

        this.stats = {
            cycleCount: 0,
            lastCycleTime: null,
            nextCycleTime: null,
            totalRefreshes: 0,
            lastError: null,
            accountHistory: {} // fileId -> { lastRefreshed, lastStatus, refreshCount }
        };
    }

    /**
     * Start the automated background worker
     */
    start() {
        if (this.isRunning) return;
        this.isRunning = true;
        console.log(`[TOKEN-DAEMON] Starting background token refresh worker (interval: ${this.cycleIntervalMs / 1000}s, threshold: ${this.refreshThresholdMs / 60000}m)`);

        // Run first cycle shortly after start (after 3 seconds)
        setTimeout(() => {
            if (this.isRunning) this.cycleAll();
        }, 3000);

        this.timer = setInterval(() => {
            if (this.isRunning) this.cycleAll();
        }, this.cycleIntervalMs);

        this.stats.nextCycleTime = new Date(Date.now() + 3000).toISOString();
    }

    /**
     * Stop the background worker
     */
    stop() {
        this.isRunning = false;
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
        }
        console.log("[TOKEN-DAEMON] Background token refresh worker stopped");
    }

    /**
     * Exchange a refresh token with Google OAuth for a fresh access token
     */
    async refreshGoogleToken(refreshToken) {
        if (!refreshToken) throw new Error("Missing refresh token");

        const postBody = new URLSearchParams({
            client_id: CLIENT_ID,
            client_secret: CLIENT_SECRET,
            refresh_token: refreshToken,
            grant_type: "refresh_token"
        }).toString();

        return new Promise((resolve, reject) => {
            const req = https.request("https://oauth2.googleapis.com/token", {
                method: "POST",
                headers: {
                    "Content-Type": "application/x-www-form-urlencoded",
                    "User-Agent": "AntigravityTokenDaemon/1.0"
                },
                timeout: 9000
            }, res => {
                let raw = "";
                res.on("data", chunk => raw += chunk);
                res.on("end", () => {
                    try {
                        const json = JSON.parse(raw);
                        if (json.access_token) {
                            resolve(json);
                        } else {
                            reject(new Error(json.error_description || json.error || `HTTP ${res.statusCode}: ${raw}`));
                        }
                    } catch (e) {
                        reject(new Error(`Failed to parse token response: ${raw}`));
                    }
                });
            });

            req.on("error", err => reject(err));
            req.on("timeout", () => {
                req.destroy();
                reject(new Error("Timeout connecting to oauth2.googleapis.com"));
            });

            req.write(postBody);
            req.end();
        });
    }

    /**
     * Fetch fresh quota summary directly from CloudCode API and update disk cache
     */
    async warmQuotaCache(fileId, accessToken) {
        if (!accessToken) return null;
        return new Promise((resolve) => {
            const req = https.request("https://cloudcode-pa.googleapis.com/v1internal:retrieveUserQuotaSummary", {
                method: "POST",
                headers: {
                    "Authorization": "Bearer " + accessToken,
                    "Content-Type": "application/json",
                    "User-Agent": "AntigravityTokenDaemon/1.0"
                },
                timeout: 8000
            }, res => {
                let b = "";
                res.on("data", c => b += c);
                res.on("end", () => {
                    try {
                        const json = JSON.parse(b);
                        if (res.statusCode === 200 && json.groups) {
                            const quotaCachePath = path.join(ACCOUNTS_DIR, `${fileId}_quota.json`);
                            const result = {
                                ...json,
                                isLive: true,
                                source: "cloudcode_api",
                                lastSynced: Date.now()
                            };
                            try {
                                fs.writeFileSync(quotaCachePath, JSON.stringify(result, null, 2), "utf8");
                                console.log(`[TOKEN-DAEMON] Warmed quota cache for ${fileId}`);
                            } catch {}
                            resolve(result);
                        } else {
                            resolve(null);
                        }
                    } catch {
                        resolve(null);
                    }
                });
            });
            req.on("error", () => resolve(null));
            req.on("timeout", () => { req.destroy(); resolve(null); });
            req.write(JSON.stringify({}));
            req.end();
        });
    }

    /**
     * Run a single pass across all N accounts, checking expirations and cycling tokens
     */
    async cycleAll(options = {}) {
        if (this.isCycling) {
            console.log("[TOKEN-DAEMON] Cycle already in progress, skipping overlapping run");
            return this.getStatus();
        }

        this.isCycling = true;
        const now = Date.now();
        this.stats.cycleCount++;
        this.stats.lastCycleTime = new Date(now).toISOString();
        this.stats.nextCycleTime = new Date(now + this.cycleIntervalMs).toISOString();

        try {
            if (!fs.existsSync(ACCOUNTS_DIR)) {
                fs.mkdirSync(ACCOUNTS_DIR, { recursive: true });
            }

            // Find all account*.json files dynamically
            const files = fs.readdirSync(ACCOUNTS_DIR)
                .filter(f => f.endsWith(".json") && !f.endsWith("_quota.json") && f !== "order.json")
                .sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }));

            // Also check oauth_creds.json
            let activeOauthData = null;
            if (fs.existsSync(OAUTH_FILE)) {
                try {
                    activeOauthData = JSON.parse(fs.readFileSync(OAUTH_FILE, "utf8"));
                } catch {}
            }

            const results = [];

            for (const file of files) {
                const filePath = path.join(ACCOUNTS_DIR, file);
                const fileId = file.replace(".json", "");

                try {
                    const raw = fs.readFileSync(filePath, "utf8");
                    const acc = JSON.parse(raw);

                    if (!acc.refresh_token) {
                        results.push({
                            fileId,
                            email: acc.email || "Unknown",
                            action: "SKIPPED_NO_REFRESH_TOKEN",
                            expiresInSec: null
                        });
                        continue;
                    }

                    const expiryDate = acc.expiry_date || 0;
                    const diffMs = expiryDate - now;
                    const expiresInSec = Math.round(diffMs / 1000);
                    const shouldRefresh = options.force || !acc.access_token || diffMs <= this.refreshThresholdMs;

                    if (shouldRefresh) {
                        console.log(`[TOKEN-DAEMON] Cycling token for ${fileId} (${acc.email}) - expires in ${expiresInSec}s (threshold: ${this.refreshThresholdMs / 1000}s)`);

                        const tokenResponse = await this.refreshGoogleToken(acc.refresh_token);

                        acc.access_token = tokenResponse.access_token;
                        acc.expiry_date = now + ((tokenResponse.expires_in || 3600) * 1000);
                        if (tokenResponse.id_token) acc.id_token = tokenResponse.id_token;
                        if (tokenResponse.scope) acc.scope = tokenResponse.scope;
                        if (tokenResponse.token_type) acc.token_type = tokenResponse.token_type;

                        // Save updated credentials atomically
                        fs.writeFileSync(filePath, JSON.stringify(acc, null, 2), "utf8");

                        // If this account's refresh_token matches oauth_creds.json, update oauth_creds.json as well!
                        if (activeOauthData && activeOauthData.refresh_token === acc.refresh_token) {
                            try {
                                activeOauthData.access_token = acc.access_token;
                                activeOauthData.expiry_date = acc.expiry_date;
                                if (acc.id_token) activeOauthData.id_token = acc.id_token;
                                fs.writeFileSync(OAUTH_FILE, JSON.stringify(activeOauthData, null, 2), "utf8");
                                console.log(`[TOKEN-DAEMON] Synchronized fresh token to active IDE session (${OAUTH_FILE})`);
                            } catch (oe) {
                                console.warn(`[TOKEN-DAEMON] Failed to sync oauth_creds.json: ${oe.message}`);
                            }
                        }

                        // Warm CloudCode quota cache immediately with new token
                        await this.warmQuotaCache(fileId, acc.access_token);

                        this.stats.totalRefreshes++;
                        this.stats.accountHistory[fileId] = {
                            lastRefreshed: new Date().toISOString(),
                            lastStatus: "SUCCESS",
                            email: acc.email,
                            refreshCount: (this.stats.accountHistory[fileId]?.refreshCount || 0) + 1
                        };

                        results.push({
                            fileId,
                            email: acc.email,
                            action: "REFRESHED",
                            newExpiryDate: acc.expiry_date,
                            expiresInSec: Math.round((acc.expiry_date - now) / 1000)
                        });
                    } else {
                        results.push({
                            fileId,
                            email: acc.email,
                            action: "VALID",
                            expiresInSec: expiresInSec
                        });
                    }
                } catch (accErr) {
                    console.error(`[TOKEN-DAEMON] Error processing ${fileId}:`, accErr.message);
                    this.stats.lastError = `${fileId}: ${accErr.message}`;
                    this.stats.accountHistory[fileId] = {
                        ...(this.stats.accountHistory[fileId] || {}),
                        lastStatus: `ERROR: ${accErr.message}`
                    };
                    results.push({
                        fileId,
                        action: "ERROR",
                        error: accErr.message
                    });
                }
            }

            // Also check if oauth_creds.json itself is expired and not covered by any account file
            if (activeOauthData && activeOauthData.refresh_token) {
                const diffMs = (activeOauthData.expiry_date || 0) - now;
                if (diffMs <= this.refreshThresholdMs) {
                    try {
                        console.log(`[TOKEN-DAEMON] Cycling standalone IDE session token (oauth_creds.json) - expires in ${Math.round(diffMs / 1000)}s`);
                        const tr = await this.refreshGoogleToken(activeOauthData.refresh_token);
                        activeOauthData.access_token = tr.access_token;
                        activeOauthData.expiry_date = now + ((tr.expires_in || 3600) * 1000);
                        if (tr.id_token) activeOauthData.id_token = tr.id_token;
                        fs.writeFileSync(OAUTH_FILE, JSON.stringify(activeOauthData, null, 2), "utf8");
                        console.log(`[TOKEN-DAEMON] Standalone IDE session token refreshed successfully`);
                    } catch (e) {
                        console.warn(`[TOKEN-DAEMON] Could not refresh standalone oauth_creds.json: ${e.message}`);
                    }
                }
            }

            return {
                success: true,
                cycleCount: this.stats.cycleCount,
                results
            };
        } catch (globalErr) {
            console.error("[TOKEN-DAEMON] Fatal cycle error:", globalErr.message);
            this.stats.lastError = globalErr.message;
            return {
                success: false,
                error: globalErr.message
            };
        } finally {
            this.isCycling = false;
        }
    }

    /**
     * Get daemon state and detailed metrics across all monitored accounts
     */
    getStatus() {
        const now = Date.now();
        let monitoredAccounts = [];

        try {
            if (fs.existsSync(ACCOUNTS_DIR)) {
                const files = fs.readdirSync(ACCOUNTS_DIR)
                    .filter(f => f.endsWith(".json") && !f.endsWith("_quota.json"))
                    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }));

                monitoredAccounts = files.map(file => {
                    const filePath = path.join(ACCOUNTS_DIR, file);
                    const fileId = file.replace(".json", "");
                    try {
                        const acc = JSON.parse(fs.readFileSync(filePath, "utf8"));
                        const expiry = acc.expiry_date || 0;
                        const diffSec = Math.round((expiry - now) / 1000);
                        const isExpiringSoon = diffSec <= (this.refreshThresholdMs / 1000);
                        const isExpired = diffSec <= 0;

                        return {
                            fileId,
                            email: acc.email || "Unknown",
                            name: acc.name || "",
                            hasRefreshToken: !!acc.refresh_token,
                            expiryDate: expiry,
                            expiresInSec: diffSec,
                            expiresInMinutes: parseFloat((diffSec / 60).toFixed(1)),
                            status: isExpired ? "EXPIRED" : isExpiringSoon ? "EXPIRING_SOON" : "HEALTHY",
                            history: this.stats.accountHistory[fileId] || null
                        };
                    } catch {
                        return { fileId, status: "READ_ERROR" };
                    }
                });
            }
        } catch (e) {
            console.error("[TOKEN-DAEMON] Error getting account statuses:", e.message);
        }

        return {
            running: this.isRunning,
            isCycling: this.isCycling,
            cycleIntervalMs: this.cycleIntervalMs,
            refreshThresholdMinutes: this.refreshThresholdMs / 60000,
            cycleCount: this.stats.cycleCount,
            lastCycleTime: this.stats.lastCycleTime,
            nextCycleTime: this.stats.nextCycleTime,
            totalRefreshes: this.stats.totalRefreshes,
            lastError: this.stats.lastError,
            accountCount: monitoredAccounts.length,
            accounts: monitoredAccounts
        };
    }
}

// Singleton instance
const defaultDaemon = new TokenDaemon();

// Allow running directly as a standalone CLI worker script
if (require.main === module) {
    console.log("=======================================================");
    console.log(" Antigravity Background Token Refresh Daemon (CLI Mode)");
    console.log("=======================================================");

    const args = process.argv.slice(2);
    if (args.includes("--once") || args.includes("-1")) {
        defaultDaemon.cycleAll({ force: args.includes("--force") }).then(res => {
            console.log("\nCycle complete:", JSON.stringify(res, null, 2));
            process.exit(0);
        });
    } else {
        defaultDaemon.start();
        process.on("SIGINT", () => {
            defaultDaemon.stop();
            process.exit(0);
        });
    }
}

module.exports = defaultDaemon;
