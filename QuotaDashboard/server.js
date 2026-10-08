const http = require("http");
const https = require("https");
const fs = require("fs");
const path = require("path");
const { execFileSync, exec } = require("child_process");

const PORT = 3210;
const USER_PROFILE = process.env.USERPROFILE || "C:\\Users\\Chowd";
const ACCOUNTS_DIR = path.join(USER_PROFILE, ".gemini", "quota_accounts");
const ORDER_FILE = path.join(ACCOUNTS_DIR, "order.json");
const OAUTH_FILE = path.join(USER_PROFILE, ".gemini", "oauth_creds.json");
const GOOGLE_ACCOUNTS_FILE = path.join(USER_PROFILE, ".gemini", "google_accounts.json");
const crypto = require("crypto");
const tokenDaemon = require("./token-daemon");

// Official Antigravity CloudCode OAuth Credentials
const CLIENT_ID = process.env.GOOGLE_CLIENT_ID || "";
const CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET || "";
const OAUTH_PORT = 8989;
const REDIRECT_URI = `http://localhost:${OAUTH_PORT}/oauth-callback`;

let activeAuthSession = null;
let activeAuthServer = null;

let cachedConfig = null;
let lastConfigCheck = 0;

// Discover live language server port & CSRF token reliably
function getLanguageServerConfig() {
    const now = Date.now();
    if (cachedConfig && now - lastConfigCheck < 60000) {
        return cachedConfig;
    }

    lastConfigCheck = now;

    try {
        const psScript = "$ProgressPreference = 'SilentlyContinue'; Get-CimInstance Win32_Process -Filter \"Name = 'language_server_windows_x64.exe'\" | Select-Object -ExpandProperty CommandLine";
        const b64 = Buffer.from(psScript, "utf16le").toString("base64");
        const out = execFileSync("powershell.exe", ["-NoProfile", "-EncodedCommand", b64], { encoding: "utf8", timeout: 4000 });

        for (const line of out.split(/\r?\n/)) {
            const portMatch = line.match(/--https_server_port\s+(\d+)/);
            const csrfMatch = line.match(/--csrf_token\s+([a-zA-Z0-9\-]+)/);
            if (portMatch && csrfMatch) {
                cachedConfig = { port: parseInt(portMatch[1]), csrfToken: csrfMatch[1] };
                return cachedConfig;
            }
        }
    } catch (e) {
        console.warn("Process inspection error:", e.message);
    }

    // Default fallback
    cachedConfig = { port: 55737, csrfToken: "1e122aca-096e-4e99-8fb6-2238d0bdff4e" };
    return cachedConfig;
}

// Query local Antigravity Language Server via ConnectRPC
function queryLanguageServerQuota(port, csrfToken) {
    return new Promise((resolve, reject) => {
        const agent = new https.Agent({ rejectUnauthorized: false });
        const req = https.request({
            hostname: "127.0.0.1",
            port: port,
            path: "/exa.language_server_pb.LanguageServerService/RetrieveUserQuotaSummary",
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Connect-Protocol-Version": "1",
                "X-Csrf-Token": csrfToken,
                "x-codeium-csrf-token": csrfToken
            },
            agent: agent,
            timeout: 5000
        }, res => {
            let data = "";
            res.on("data", chunk => data += chunk);
            res.on("end", () => {
                try {
                    const json = JSON.parse(data);
                    resolve(json);
                } catch (e) {
                    reject(new Error("Invalid JSON: " + data));
                }
            });
        });

        req.on("error", err => {
            cachedConfig = null;
            reject(err);
        });
        req.on("timeout", () => {
            req.destroy();
            cachedConfig = null;
            reject(new Error("Request timeout querying language server"));
        });
        req.write(JSON.stringify({ forceRefresh: true }));
        req.end();
    });
}

// Refresh Google OAuth Access Token using official client credentials
async function refreshAccessToken(accountData, filePath) {
    if (!accountData.refresh_token) return null;
    return new Promise((resolve) => {
        const body = new URLSearchParams({
            client_id: CLIENT_ID,
            client_secret: CLIENT_SECRET,
            refresh_token: accountData.refresh_token,
            grant_type: "refresh_token"
        }).toString();

        const req = https.request("https://oauth2.googleapis.com/token", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            timeout: 7000
        }, res => {
            let b = "";
            res.on("data", c => b += c);
            res.on("end", () => {
                try {
                    const json = JSON.parse(b);
                    if (json.access_token) {
                        accountData.access_token = json.access_token;
                        accountData.expiry_date = Date.now() + ((json.expires_in || 3600) * 1000);
                        if (filePath && fs.existsSync(filePath)) {
                            try {
                                fs.writeFileSync(filePath, JSON.stringify(accountData, null, 2), "utf8");
                            } catch {}
                        }
                        resolve(json.access_token);
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
        req.write(body);
        req.end();
    });
}

// Fetch live quota summary directly from Google Cloud Code API
async function fetchGoogleQuotaDirect(accessToken) {
    if (!accessToken) return null;
    return new Promise((resolve) => {
        const req = https.request("https://cloudcode-pa.googleapis.com/v1internal:retrieveUserQuotaSummary", {
            method: "POST",
            headers: {
                "Authorization": "Bearer " + accessToken,
                "Content-Type": "application/json",
                "User-Agent": "AntigravityIDE"
            },
            timeout: 7000
        }, res => {
            let b = "";
            res.on("data", c => b += c);
            res.on("end", () => {
                try {
                    const json = JSON.parse(b);
                    if (res.statusCode === 200 && json.groups) {
                        resolve(json);
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

// Get quota summary for a specific account (direct API, language server, or cached)
async function getQuotaForAccount(acc, isActive, lsQuotaData, activeEmail) {
    const quotaCachePath = path.join(ACCOUNTS_DIR, `${acc.fileId}_quota.json`);
    const accFilePath = path.join(ACCOUNTS_DIR, acc.fileName);

    // 1. Prioritize Direct Google Cloud Code API with fresh access token (Authoritative source)
    try {
        if (fs.existsSync(accFilePath)) {
            const raw = fs.readFileSync(accFilePath, "utf8");
            const accData = JSON.parse(raw);
            let token = accData.access_token;
            const isExpired = !accData.expiry_date || Date.now() >= (accData.expiry_date - 60000);

            if (isExpired || !token) {
                token = await refreshAccessToken(accData, accFilePath);
            }

            if (token) {
                const directQuota = await fetchGoogleQuotaDirect(token);
                if (directQuota && directQuota.groups) {
                    const result = {
                        ...directQuota,
                        isLive: true,
                        source: "cloudcode_api",
                        lastSynced: Date.now()
                    };
                    try { fs.writeFileSync(quotaCachePath, JSON.stringify(result, null, 2), "utf8"); } catch {}
                    return result;
                }
            }
        }
    } catch (e) {
        console.warn(`[QUOTA] Direct fetch error for ${acc.fileId}:`, e.message);
    }

    // 2. Secondary fallback: Language Server IF and ONLY IF this account is the active session
    if (isActive && lsQuotaData && lsQuotaData.groups && (!activeEmail || acc.email === activeEmail)) {
        const result = {
            ...lsQuotaData,
            isLive: true,
            source: "language_server",
            lastSynced: Date.now()
        };
        try { fs.writeFileSync(quotaCachePath, JSON.stringify(result, null, 2), "utf8"); } catch {}
        return result;
    }

    // 3. Fallback to cached quota on disk
    if (fs.existsSync(quotaCachePath)) {
        try {
            const cached = JSON.parse(fs.readFileSync(quotaCachePath, "utf8"));
            if (cached && cached.groups) {
                return {
                    ...cached,
                    isLive: false,
                    isCached: true
                };
            }
        } catch {}
    }

    return null;
}

// Decode base64url JWT payload
function decodeJwtPayload(token) {
    try {
        const parts = token.split(".");
        if (parts.length < 2) return {};
        const base64 = parts[1].replace(/-/g, "+").replace(/_/g, "/");
        const jsonStr = Buffer.from(base64, "base64").toString("utf8");
        return JSON.parse(jsonStr);
    } catch {
        return {};
    }
}

// Read all accounts from quota_accounts directory
function getSavedAccounts() {
    if (!fs.existsSync(ACCOUNTS_DIR)) {
        fs.mkdirSync(ACCOUNTS_DIR, { recursive: true });
    }

    let activeEmail = "";
    if (fs.existsSync(GOOGLE_ACCOUNTS_FILE)) {
        try {
            const accData = JSON.parse(fs.readFileSync(GOOGLE_ACCOUNTS_FILE, "utf8"));
            activeEmail = accData.active || "";
        } catch {}
    }

    let activeRefreshToken = "";
    if (fs.existsSync(OAUTH_FILE)) {
        try {
            const oData = JSON.parse(fs.readFileSync(OAUTH_FILE, "utf8"));
            activeRefreshToken = oData.refresh_token || "";
            if (!activeEmail && oData.id_token) {
                const j = decodeJwtPayload(oData.id_token);
                activeEmail = j.email || "";
            }
        } catch {}
    }

    const files = fs.readdirSync(ACCOUNTS_DIR)
        .filter(f => f.endsWith(".json") && !f.endsWith("_quota.json") && f !== "order.json")
        .sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }));

    // --- Pass 1: collect all account data (no isActive yet) ---
    const rawAccounts = [];
    const seenEmails = new Map();
    let tokenMatchFileId = null; // fileId of the account whose refresh_token matches oauth_creds.json

    for (const file of files) {
        const filePath = path.join(ACCOUNTS_DIR, file);
        try {
            const raw = fs.readFileSync(filePath, "utf8");
            const data = JSON.parse(raw);
            const jwtInfo = data.id_token ? decodeJwtPayload(data.id_token) : {};
            const email = jwtInfo.email || data.email || file.replace(".json", "");
            const name = jwtInfo.name || data.name || email.split("@")[0];
            const picture = jwtInfo.picture || "";

            const isDuplicate = seenEmails.has(email);
            const originalFile = seenEmails.get(email) || null;
            if (!isDuplicate) seenEmails.set(email, file);

            const fileId = file.replace(".json", "");

            // Refresh token match is the ground truth — exactly one account can win
            if (activeRefreshToken && data.refresh_token === activeRefreshToken && !tokenMatchFileId) {
                tokenMatchFileId = fileId;
            }

            rawAccounts.push({
                fileId,
                fileName: file,
                email,
                name,
                picture,
                isDuplicate,
                duplicateOf: originalFile,
                hasRefreshToken: !!data.refresh_token,
                expiryDate: data.expiry_date,
                refreshToken: data.refresh_token || ""
            });
        } catch (e) {
            console.error("Error reading account file:", file, e.message);
        }
    }

    // --- Pass 2: assign isActive to exactly ONE account ---
    // Priority: refresh token match > email match (only if no token match found)
    let activeAssigned = false;
    const accounts = rawAccounts.map(acc => {
        let isActive = false;
        if (tokenMatchFileId) {
            // Refresh token winner takes all — ignore email entirely
            isActive = (acc.fileId === tokenMatchFileId);
        } else if (!activeAssigned && !acc.isDuplicate && acc.email === activeEmail) {
            // Fallback: first non-duplicate account whose email matches google_accounts.json
            isActive = true;
            activeAssigned = true;
        }
        const { refreshToken, ...rest } = acc; // don't expose raw token in API response
        return { ...rest, isActive };
    });

    // Apply saved custom order if present
    if (fs.existsSync(ORDER_FILE)) {
        try {
            const savedOrder = JSON.parse(fs.readFileSync(ORDER_FILE, "utf8"));
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
        } catch {}
    }

    return { accounts, activeEmail };
}

// Start HTTP Server
const server = http.createServer(async (req, res) => {
    // CORS headers for VS Code Live Server (port 5500) and other frontends
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");

    if (req.method === "OPTIONS") {
        res.writeHead(204);
        return res.end();
    }

    const url = new URL(req.url, `http://${req.headers.host}`);

    // API: Status
    if (url.pathname === "/api/status") {
        try {
            const { accounts, activeEmail } = getSavedAccounts();
            const lsConfig = getLanguageServerConfig();
            let quotaData = null;
            let error = null;

            try {
                quotaData = await queryLanguageServerQuota(lsConfig.port, lsConfig.csrfToken);
            } catch (err) {
                error = err.message;
            }

            const lsResponse = quotaData?.response || null;

            // Fetch / attach quota for ALL accounts
            for (const acc of accounts) {
                acc.quotas = await getQuotaForAccount(acc, acc.isActive, lsResponse, activeEmail);
            }

            // Active quota priority: active account's quota or language server response
            const activeAcc = accounts.find(a => a.isActive);
            const primaryQuotas = activeAcc?.quotas || lsResponse || null;

            res.writeHead(200, { "Content-Type": "application/json" });
            return res.end(JSON.stringify({
                success: true,
                serverTime: new Date().toISOString(),
                languageServer: { port: lsConfig.port, connected: !error },
                activeEmail: activeEmail,
                accounts: accounts,
                quotas: primaryQuotas,
                tokenDaemon: tokenDaemon.getStatus(),
                error: error
            }));
        } catch (e) {
            res.writeHead(500, { "Content-Type": "application/json" });
            return res.end(JSON.stringify({ success: false, error: e.message }));
        }
    }

    // API: Start Ingestion Flow
    if (url.pathname === "/api/auth/start" && req.method === "POST") {
        let body = "";
        req.on("data", chunk => body += chunk);
        req.on("end", () => {
            try {
                const { target } = body ? JSON.parse(body) : {};
                let chosenTarget = target;
                if (!chosenTarget) {
                    const { accounts } = getSavedAccounts();
                    const dup = accounts.find(a => a.isDuplicate);
                    if (dup) {
                        chosenTarget = dup.fileId;
                    } else {
                        let n = 1;
                        while (fs.existsSync(path.join(ACCOUNTS_DIR, `account${n}.json`))) {
                            n++;
                        }
                        chosenTarget = `account${n}`;
                    }
                }

                // Generate PKCE credentials
                const verifier = crypto.randomBytes(32).toString("base64url");
                const challenge = crypto.createHash("sha256").update(verifier).digest("base64url");
                const stateToken = crypto.randomBytes(16).toString("hex");

                activeAuthSession = {
                    target: chosenTarget,
                    verifier: verifier,
                    stateToken: stateToken
                };

                // Close any existing temporary auth listener on port 8989
                if (activeAuthServer) {
                    try { activeAuthServer.close(); } catch {}
                    activeAuthServer = null;
                }

                // Start isolated OAuth callback listener on port 8989
                activeAuthServer = http.createServer(async (cbReq, cbRes) => {
                    const cbUrl = new URL(cbReq.url, `http://${cbReq.headers.host}`);
                    if (cbUrl.pathname === "/oauth-callback" || cbUrl.pathname === "/oauth/callback") {
                        const code = cbUrl.searchParams.get("code");
                        const error = cbUrl.searchParams.get("error");

                        if (error) {
                            cbRes.writeHead(400, { "Content-Type": "text/html; charset=utf-8" });
                            cbRes.end(`<body style="background:#0f172a;color:#ef4444;text-align:center;padding:50px;font-family:sans-serif;"><h2>Authentication Canceled</h2><p>${error}</p></body>`);
                            try { activeAuthServer.close(); } catch {}
                            activeAuthServer = null;
                            return;
                        }

                        if (!code) {
                            cbRes.writeHead(400, { "Content-Type": "text/plain" });
                            return cbRes.end("Missing authorization code.");
                        }

                        try {
                            const postBody = new URLSearchParams({
                                client_id: CLIENT_ID,
                                client_secret: CLIENT_SECRET,
                                code: code,
                                code_verifier: activeAuthSession.verifier,
                                grant_type: "authorization_code",
                                redirect_uri: REDIRECT_URI
                            }).toString();

                            const tokenData = await new Promise((resolve, reject) => {
                                const tokenReq = https.request({
                                    hostname: "oauth2.googleapis.com",
                                    path: "/token",
                                    method: "POST",
                                    headers: { "Content-Type": "application/x-www-form-urlencoded" }
                                }, tokenRes => {
                                    let raw = "";
                                    tokenRes.on("data", c => raw += c);
                                    tokenRes.on("end", () => {
                                        try {
                                            const p = JSON.parse(raw);
                                            if (p.error) reject(new Error(p.error_description || p.error));
                                            else resolve(p);
                                        } catch (e) { reject(e); }
                                    });
                                });
                                tokenReq.on("error", reject);
                                tokenReq.write(postBody);
                                tokenReq.end();
                            });

                            const jwtInfo = tokenData.id_token ? decodeJwtPayload(tokenData.id_token) : {};
                            const email = jwtInfo.email || "Unknown";
                            const name = jwtInfo.name || email.split("@")[0];
                            const picture = jwtInfo.picture || "";

                            const targetFile = path.join(ACCOUNTS_DIR, `${activeAuthSession.target}.json`);
                            const credsToSave = {
                                access_token: tokenData.access_token,
                                scope: tokenData.scope || "https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile openid",
                                token_type: tokenData.token_type || "Bearer",
                                id_token: tokenData.id_token,
                                expiry_date: Date.now() + ((tokenData.expires_in || 3600) * 1000),
                                refresh_token: tokenData.refresh_token,
                                email: email,
                                name: name
                            };

                            if (!fs.existsSync(ACCOUNTS_DIR)) fs.mkdirSync(ACCOUNTS_DIR, { recursive: true });
                            fs.writeFileSync(targetFile, JSON.stringify(credsToSave, null, 2), "utf8");

                            cbRes.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
                            cbRes.end(`
                                <!DOCTYPE html><html><head><title>Authentication Successful</title>
                                <style>body{background:#0b0f19;color:#f8fafc;font-family:-apple-system,sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;}
                                .card{background:#131d31;border:1px solid #1e293b;padding:40px;border-radius:20px;text-align:center;max-width:440px;box-shadow:0 20px 40px rgba(0,0,0,0.6);}
                                .badge{background:#10b981;color:#022c22;font-weight:700;padding:6px 14px;border-radius:99px;font-size:13px;display:inline-block;margin-bottom:20px;}
                                .avatar{width:64px;height:64px;border-radius:50%;border:3px solid #38bdf8;margin:0 auto 16px;display:block;}
                                h2{margin:0 0 8px;font-size:24px;}p{color:#94a3b8;font-size:15px;margin:6px 0;}
                                .email{color:#38bdf8;font-weight:600;font-family:monospace;font-size:16px;background:rgba(56,189,248,0.1);padding:4px 10px;border-radius:6px;display:inline-block;margin:8px 0;}
                                .file-box{margin-top:16px;padding:12px;background:#090d16;border-radius:8px;font-size:13px;color:#cbd5e1;border:1px solid #1e293b;}
                                .tip{margin-top:20px;font-size:13px;color:#64748b;}</style></head>
                                <body><div class="card">
                                    <span class="badge">&#10003; AUTHENTICATION SUCCESSFUL</span>
                                    ${picture ? `<img src="${picture}" class="avatar" alt="Avatar">` : ""}
                                    <h2>Account Ingested!</h2>
                                    <p>Welcome, <strong>${name}</strong></p>
                                    <div class="email">${email}</div>
                                    <div class="file-box">Saved to: <code>quota_accounts/${path.basename(targetFile)}</code></div>
                                    <p class="tip">You can close this tab and return to the Quota Dashboard!</p>
                                </div></body></html>
                            `);

                            console.log(`[AUTH] Successfully ingested ${email} into ${targetFile}`);
                            setTimeout(() => {
                                try { activeAuthServer.close(); } catch {}
                                activeAuthServer = null;
                            }, 1000);
                        } catch (e) {
                            cbRes.writeHead(500, { "Content-Type": "text/html; charset=utf-8" });
                            cbRes.end(`<body style="background:#0f172a;color:#ef4444;padding:40px;text-align:center;"><h2>Exchange Failed</h2><p>${e.message}</p></body>`);
                        }
                    } else {
                        cbRes.writeHead(404); cbRes.end();
                    }
                });

                activeAuthServer.on("error", (err) => {
                    console.error("[AUTH] Listener error:", err.message);
                });

                activeAuthServer.listen(OAUTH_PORT, "127.0.0.1", () => {
                    console.log(`[AUTH] Ingestion listener active on http://localhost:${OAUTH_PORT}/oauth-callback`);
                });

                // Construct Google OAuth URL
                const authParams = new URLSearchParams({
                    client_id: CLIENT_ID,
                    redirect_uri: REDIRECT_URI,
                    response_type: "code",
                    scope: "https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/cclog https://www.googleapis.com/auth/experimentsandconfigs openid",
                    code_challenge: challenge,
                    code_challenge_method: "S256",
                    state: stateToken,
                    access_type: "offline",
                    prompt: "consent select_account"
                }).toString();

                const authUrl = `https://accounts.google.com/o/oauth2/v2/auth?${authParams}`;

                res.writeHead(200, { "Content-Type": "application/json" });
                return res.end(JSON.stringify({
                    success: true,
                    authUrl: authUrl,
                    target: chosenTarget,
                    message: `OAuth ready for ${chosenTarget}!`
                }));
            } catch (e) {
                res.writeHead(500, { "Content-Type": "application/json" });
                return res.end(JSON.stringify({ success: false, error: e.message }));
            }
        });
        return;
    }

    // API: Switch Active Account (Disabled - switching is manual)
    if (url.pathname === "/api/switch-account" && req.method === "POST") {
        res.writeHead(400, { "Content-Type": "application/json" });
        return res.end(JSON.stringify({ success: false, error: "Account switching via dashboard is disabled. Switch accounts manually in Antigravity IDE." }));
    }

    // API: Token Daemon Status
    if (url.pathname === "/api/token-daemon/status" && req.method === "GET") {
        res.writeHead(200, { "Content-Type": "application/json" });
        return res.end(JSON.stringify({ success: true, ...tokenDaemon.getStatus() }));
    }

    // API: Token Daemon Trigger Manual Cycle
    if (url.pathname === "/api/token-daemon/cycle" && req.method === "POST") {
        let body = "";
        req.on("data", chunk => body += chunk);
        req.on("end", async () => {
            try {
                const { force } = body ? JSON.parse(body) : {};
                const result = await tokenDaemon.cycleAll({ force: !!force });
                res.writeHead(200, { "Content-Type": "application/json" });
                return res.end(JSON.stringify({ success: true, result, daemon: tokenDaemon.getStatus() }));
            } catch (e) {
                res.writeHead(500, { "Content-Type": "application/json" });
                return res.end(JSON.stringify({ success: false, error: e.message }));
            }
        });
        return;
    }

    // API: Save Account Order to disk
    if (url.pathname === "/api/accounts/reorder" && req.method === "POST") {
        let body = "";
        req.on("data", chunk => body += chunk);
        req.on("end", () => {
            try {
                const { order } = JSON.parse(body || "{}");
                if (Array.isArray(order)) {
                    fs.writeFileSync(ORDER_FILE, JSON.stringify(order, null, 2), "utf8");
                    res.writeHead(200, { "Content-Type": "application/json" });
                    return res.end(JSON.stringify({ success: true, order }));
                }
                res.writeHead(400, { "Content-Type": "application/json" });
                return res.end(JSON.stringify({ success: false, error: "Invalid order array" }));
            } catch (e) {
                res.writeHead(500, { "Content-Type": "application/json" });
                return res.end(JSON.stringify({ success: false, error: e.message }));
            }
        });
        return;
    }

    // Static Files
    let filePath = path.join(__dirname, url.pathname === "/" ? "index.html" : url.pathname);
    if (!fs.existsSync(filePath)) {
        res.writeHead(404, { "Content-Type": "text/plain" });
        return res.end("404 Not Found");
    }

    const ext = path.extname(filePath);
    const contentTypes = {
        ".html": "text/html",
        ".css": "text/css",
        ".js": "application/javascript",
        ".json": "application/json",
        ".png": "image/png",
        ".svg": "image/svg+xml"
    };

    res.writeHead(200, { "Content-Type": contentTypes[ext] || "text/plain" });
    fs.createReadStream(filePath).pipe(res);
});

process.on("uncaughtException", err => {
    console.error("Uncaught Exception:", err.message);
});

process.on("unhandledRejection", reason => {
    console.error("Unhandled Rejection:", reason);
});

server.listen(PORT, "127.0.0.1", () => {
    tokenDaemon.start();
    console.log(`=======================================================`);
    console.log(` Antigravity Multi-Account Quota Dashboard is running!`);
    console.log(` URL: http://localhost:${PORT}`);
    console.log(` Token Daemon: 24/7 Background Sync Active across N accounts`);
    console.log(` CORS: Enabled for all origins (Live Server :5500)`);
    console.log(` Watching Accounts Dir: ${ACCOUNTS_DIR}`);
    console.log(`=======================================================`);
});

process.on("SIGINT", () => {
    tokenDaemon.stop();
    process.exit(0);
});

process.on("SIGTERM", () => {
    tokenDaemon.stop();
    process.exit(0);
});
