param(
    [string]$CommitMessage = "Update lessons and study notes"
)

$ErrorActionPreference = "Stop"

Write-Host "`n🚀 --- Syncing AI-ML Study Notes & Streamlit Deploy Branch ---" -ForegroundColor Cyan

# 1. Ensure we are on main
$currentBranch = (git branch --show-current).Trim()
if ($currentBranch -ne "main") {
    Write-Host "Switching to 'main' branch..." -ForegroundColor Yellow
    git checkout main
}

# 2. Stage and commit current work on main
Write-Host "`n📦 Step 1: Staging and committing all files to 'main'..." -ForegroundColor Cyan
git add -A
$status = git status --porcelain
if ($status) {
    git commit -m "$CommitMessage"
} else {
    Write-Host "No changes to commit on main." -ForegroundColor Gray
}

# 3. Push main to remote origin
Write-Host "`n🌐 Step 2: Pushing 'main' to GitHub..." -ForegroundColor Cyan
git push origin main

# 4. Prepare isolated deploy branch
Write-Host "`n🧹 Step 3: Preparing clean 'deploy' branch with only app folders..." -ForegroundColor Cyan
# Create a detached orphan branch
git checkout --orphan deploy-sync-temp 2>$null
git rm -rf . >$null

# Pull only the deployable folders, configuration, and .gitignore from main
git checkout main -- .gitignore app.py requirements.txt packages.txt .streamlit/ 01_Data_Science_Foundations/ 02_Machine_Learning/ 03_Data_Visualization/ 04_Generative_AI/ 05_Deployment/
git add .gitignore app.py requirements.txt packages.txt .streamlit "01_Data_Science_Foundations" "02_Machine_Learning" "03_Data_Visualization" "04_Generative_AI" "05_Deployment"
git commit -m "Auto-deploy Streamlit Hub [skip ci]"

# 5. Push clean branch to remote deploy branch
Write-Host "`n☁️  Step 4: Pushing 'deploy' branch to GitHub..." -ForegroundColor Cyan
git push origin deploy-sync-temp:deploy --force
$pushSuccess = ($LASTEXITCODE -eq 0)

# 6. Safely return back to main
Write-Host "`n🔄 Step 5: Returning to 'main' branch..." -ForegroundColor Cyan
git checkout main
git branch -D deploy-sync-temp 2>$null

if ($pushSuccess) {
    Write-Host "`n✅ Done! The 'deploy' branch has been updated on GitHub." -ForegroundColor Green
    Write-Host "Streamlit Community Cloud (tracking 'deploy') will automatically deploy.`n" -ForegroundColor Yellow
} else {
    Write-Host "`n❌ Error: Failed to push to 'deploy' branch on GitHub. Check error output above.`n" -ForegroundColor Red
    exit 1
}
