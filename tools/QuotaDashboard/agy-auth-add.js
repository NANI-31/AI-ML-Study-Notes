/**
 * Antigravity Automated Account Ingestion CLI Tool (agy-auth-add)
 * 
 * Authenticates secondary and tertiary Google accounts directly into
 * ~/.gemini/quota_accounts/accountX.json using an isolated OAuth loopback listener.
 * This runs independently of the IDE, requiring NO IDE sign-out or session loss.
 */

const http = require("http");
const https = require("https");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const { exec } = require("child_process");

// Official Antigravity CloudCode OAuth credentials
const CLIENT_ID = process.env.GOOGLE_CLIENT_ID || "";
const CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET || "";

const PORT = 8989;
const REDIRECT_URI = `http://localhost:${PORT}/oauth-callback`;

const USER_PROFILE = process.env.USERPROFILE || "C:\\Users\\Chowd";
const ACCOUNTS_DIR = path.join(USER_PROFILE, ".gemini", "quota_accounts");

// Parse target account name from CLI arguments (default: next available account)
let targetName = process.argv[2];
if (!targetName) {
    let n = 1;
    while (fs.existsSync(path.join(ACCOUNTS_DIR, `account${n}.json`))) {
        n++;
    }
    targetName = `account${n}`;
}
if (targetName.endsWith(".json")) {
    targetName = targetName.replace(".json", "");
}

const targetFile = path.join(ACCOUNTS_DIR, `${targetName}.json`);

// 1. Generate PKCE Verifier and S256 Challenge
const verifier = crypto.randomBytes(32).toString("base64url");
const challenge = crypto.createHash("sha256").update(verifier).digest("base64url");
const stateToken = crypto.randomBytes(16).toString("hex");

// 2. Decode JWT payload helper
function decodeJwt(token) {
    try {
        const base64 = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
        return JSON.parse(Buffer.from(base64, "base64").toString("utf8"));
    } catch {
        return {};
    }
}

// 3. Start Loopback HTTP Server
const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, `http://${req.headers.host}`);

    // Handle both /oauth-callback and /oauth/callback
    if (url.pathname === "/oauth-callback" || url.pathname === "/oauth/callback") {
        const code = url.searchParams.get("code");
        const error = url.searchParams.get("error");

        if (error) {
            res.writeHead(400, { "Content-Type": "text/html; charset=utf-8" });
            res.end(`
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Authentication Canceled</title>
                    <style>
                        body { background: #0b0f19; color: #f8fafc; font-family: -apple-system, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
                        .card { background: #1e293b; border: 1px solid #ef4444; padding: 40px; border-radius: 16px; text-align: center; max-width: 440px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
                        .badge { background: #ef4444; color: #fff; font-weight: bold; padding: 6px 14px; border-radius: 99px; font-size: 14px; display: inline-block; margin-bottom: 20px; }
                        h2 { margin: 0 0 10px; font-size: 22px; }
                        p { color: #94a3b8; font-size: 15px; }
                    </style>
                </head>
                <body>
                    <div class="card">
                        <span class="badge">AUTH FAILED</span>
                        <h2>Authentication Canceled</h2>
                        <p>Error: ${error}</p>
                        <p style="margin-top:20px; font-size:13px; color:#64748b;">You may close this tab and try again.</p>
                    </div>
                </body>
                </html>
            `);
            console.error("\n[ERROR] Google OAuth Error:", error);
            setTimeout(() => server.close(() => process.exit(1)), 1000);
            return;
        }

        if (!code) {
            res.writeHead(400, { "Content-Type": "text/plain" });
            return res.end("Missing authorization code.");
        }

        console.log("\n[INFO] Authorization code received! Exchanging with Google OAuth...");

        try {
            // Exchange code for tokens
            const postBody = new URLSearchParams({
                client_id: CLIENT_ID,
                client_secret: CLIENT_SECRET,
                code: code,
                code_verifier: verifier,
                grant_type: "authorization_code",
                redirect_uri: REDIRECT_URI
            }).toString();

            const tokenData = await new Promise((resolve, reject) => {
                const tokenReq = https.request({
                    hostname: "oauth2.googleapis.com",
                    path: "/token",
                    method: "POST",
                    headers: {
                        "Content-Type": "application/x-www-form-urlencoded"
                    }
                }, tokenRes => {
                    let raw = "";
                    tokenRes.on("data", c => raw += c);
                    tokenRes.on("end", () => {
                        try {
                            const parsed = JSON.parse(raw);
                            if (parsed.error) reject(new Error(parsed.error_description || parsed.error));
                            else resolve(parsed);
                        } catch (e) {
                            reject(new Error("Invalid token JSON response: " + raw));
                        }
                    });
                });
                tokenReq.on("error", reject);
                tokenReq.write(postBody);
                tokenReq.end();
            });

            // Decode account identity from id_token
            const jwtInfo = tokenData.id_token ? decodeJwt(tokenData.id_token) : {};
            const email = jwtInfo.email || "Unknown";
            const name = jwtInfo.name || email.split("@")[0];
            const picture = jwtInfo.picture || "";

            // Build credentials JSON matching Antigravity IDE format
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

            // Ensure directory exists and write
            if (!fs.existsSync(ACCOUNTS_DIR)) {
                fs.mkdirSync(ACCOUNTS_DIR, { recursive: true });
            }
            fs.writeFileSync(targetFile, JSON.stringify(credsToSave, null, 2), "utf8");

            // Success page in browser
            res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
            res.end(`
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Account Authenticated - Antigravity Quota Hub</title>
                    <style>
                        body { background: #0b0f19; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
                        .card { background: #131d31; border: 1px solid #1e293b; padding: 40px; border-radius: 20px; text-align: center; max-width: 460px; box-shadow: 0 20px 40px rgba(0,0,0,0.6); }
                        .badge { background: #10b981; color: #022c22; font-weight: 700; padding: 6px 14px; border-radius: 99px; font-size: 13px; display: inline-block; margin-bottom: 20px; letter-spacing: 0.5px; }
                        .avatar { width: 64px; height: 64px; border-radius: 50%; border: 3px solid #38bdf8; margin: 0 auto 16px; display: block; object-fit: cover; }
                        h2 { margin: 0 0 8px; font-size: 24px; font-weight: 700; }
                        p { color: #94a3b8; font-size: 15px; margin: 6px 0; }
                        .email { color: #38bdf8; font-weight: 600; font-family: ui-monospace, monospace; font-size: 16px; background: rgba(56, 189, 248, 0.1); padding: 4px 10px; border-radius: 6px; display: inline-block; margin: 8px 0; }
                        .file-box { margin-top: 18px; padding: 12px; background: #090d16; border-radius: 8px; font-size: 13px; color: #cbd5e1; border: 1px solid #1e293b; word-break: break-all; }
                        .tip { margin-top: 24px; padding-top: 18px; border-top: 1px solid #1e293b; font-size: 13px; color: #64748b; }
                    </style>
                </head>
                <body>
                    <div class="card">
                        <span class="badge">&#10003; AUTHENTICATION SUCCESSFUL</span>
                        ${picture ? `<img src="${picture}" class="avatar" alt="Avatar">` : ""}
                        <h2>Account Ingested!</h2>
                        <p>Welcome, <strong>${name}</strong></p>
                        <div class="email">${email}</div>
                        <div class="file-box">Saved to: <code>quota_accounts/${path.basename(targetFile)}</code></div>
                        <p class="tip">You can now close this tab. The Quota Dashboard is ready to monitor this account!</p>
                    </div>
                </body>
                </html>
            `);

            console.log("\n=======================================================");
            console.log(` [SUCCESS] Google Account Successfully Ingested!`);
            console.log(` User   : ${name}`);
            console.log(` Email  : ${email}`);
            console.log(` File   : ${targetFile}`);
            console.log(` Status : Ready in Quota Dashboard at http://localhost:3210`);
            console.log("=======================================================\n");

            setTimeout(() => {
                server.close(() => process.exit(0));
            }, 1000);

        } catch (err) {
            console.error("\n[ERROR] Failed to exchange authorization code:", err.message);
            res.writeHead(500, { "Content-Type": "text/html; charset=utf-8" });
            res.end(`
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Token Exchange Error</title>
                    <style>
                        body { background: #0b0f19; color: #f8fafc; font-family: -apple-system, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
                        .card { background: #1e293b; border: 1px solid #ef4444; padding: 40px; border-radius: 16px; text-align: center; max-width: 440px; }
                        h2 { color: #ef4444; margin-top: 0; }
                        p { color: #94a3b8; font-size: 14px; }
                    </style>
                </head>
                <body>
                    <div class="card">
                        <h2>Token Exchange Failed</h2>
                        <p>${err.message}</p>
                    </div>
                </body>
                </html>
            `);
            setTimeout(() => server.close(() => process.exit(1)), 1000);
        }
    } else {
        res.writeHead(404);
        res.end();
    }
});

server.listen(PORT, "127.0.0.1", () => {
    // 4. Construct Auth URL with prompt=select_account to force account chooser
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

    console.log("=======================================================");
    console.log(` Antigravity Account Ingestion Tool`);
    console.log(` Target File: ${targetFile}`);
    console.log(` Loopback Listener: ${REDIRECT_URI}`);
    console.log("=======================================================");
    console.log("\nLaunching default browser for Google Account selection...");
    console.log("If your browser did not open automatically, navigate to:\n");
    console.log(authUrl + "\n");

    // Launch default browser on Windows
    exec(`start "" "${authUrl}"`);
});

server.on("error", (err) => {
    if (err.code === "EADDRINUSE") {
        console.error(`\n[ERROR] Port ${PORT} is already in use by another process.`);
        console.error(`Please make sure no other instance of agy-auth-add is running.\n`);
    } else {
        console.error("\n[ERROR] Server error:", err.message);
    }
    process.exit(1);
});
