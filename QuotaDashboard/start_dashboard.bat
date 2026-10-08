@echo off
title Antigravity Quota Dashboard
cd /d "%~dp0"
start "" http://localhost:3210
node server.js
pause
