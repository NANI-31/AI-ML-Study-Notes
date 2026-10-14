import subprocess
import re
import json
import urllib.request
import ssl
import sys
import time
from datetime import datetime, timezone

def get_language_server_config():
    try:
        ps_cmd = "$ProgressPreference = 'SilentlyContinue'; Get-CimInstance Win32_Process -Filter \"Name = 'language_server_windows_x64.exe'\" | Select-Object -ExpandProperty CommandLine"
        out = subprocess.check_output(["powershell.exe", "-NoProfile", "-Command", ps_cmd], text=True, timeout=5)
        for line in out.splitlines():
            port_match = re.search(r'--https_server_port\s+(\d+)', line)
            csrf_match = re.search(r'--csrf_token\s+([a-zA-Z0-9\-]+)', line)
            if port_match and csrf_match:
                return int(port_match.group(1)), csrf_match.group(1)
    except Exception:
        pass
    
    targets = [
        (62491, "3d40a5a8-2720-4350-94a6-47c209e464e1"),
        (65355, "9cc1a65c-df9e-40ff-823f-e50aa23e69a2"),
        (55737, "1e122aca-096e-4e99-8fb6-2238d0bdff4e")
    ]
    return targets[0]

def query_quota():
    port, csrf = get_language_server_config()
    url = f"https://127.0.0.1:{port}/exa.language_server_pb.LanguageServerService/RetrieveUserQuotaSummary"
    headers = {
        "Content-Type": "application/json",
        "Connect-Protocol-Version": "1",
        "X-Csrf-Token": csrf,
        "x-codeium-csrf-token": csrf
    }
    body = json.dumps({"forceRefresh": True}).encode("utf-8")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, context=ctx, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_gemini_5h_status():
    q = query_quota()
    groups = q.get("response", {}).get("groups", [])
    for g in groups:
        if "gemini" in g.get("displayName", "").lower():
            for b in g.get("buckets", []):
                if b.get("bucketId") == "gemini-5h" or b.get("window") == "5h":
                    rem = b.get("remainingFraction", 1.0)
                    reset_time = b.get("resetTime")
                    return rem, reset_time
    return 1.0, None

def inspect_full_quota():
    q = query_quota()
    groups = q.get("response", {}).get("groups", [])
    res = {}
    for g in groups:
        name = g.get("displayName")
        res[name] = {}
        for b in g.get("buckets", []):
            res[name][b.get("bucketId")] = {
                "displayName": b.get("displayName"),
                "remainingFraction": b.get("remainingFraction"),
                "resetTime": b.get("resetTime")
            }
    return res

if __name__ == "__main__":
    rem, reset_time = get_gemini_5h_status()
    print(f"GEMINI_5H_REMAINING: {rem * 100:.2f}%")
    print(f"RESET_TIME: {reset_time}")
    if rem <= 0.04:
        print("ALERT: Gemini 5-hour quota is at or below 4%! Pause required.")
    else:
        print("STATUS: Gemini 5-hour quota healthy.")
