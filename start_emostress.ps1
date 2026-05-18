Write-Host "Starting EmoStress Backend..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd emostress-app\backend; .\venv\Scripts\activate; uvicorn main:app --reload --port 8000"

Write-Host "Starting EmoStress Frontend..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd emostress-app\frontend; npm run dev"

Write-Host "Both servers are starting in new windows!"
