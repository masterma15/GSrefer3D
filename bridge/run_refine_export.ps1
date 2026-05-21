# Refine all objects then export data2_sft. Run in Windows envGS from repo root.
$ErrorActionPreference = "Stop"
Set-Location "E:\3DGS-VLM"

$dirs = Get-ChildItem "training_data\data2_*" -Directory |
    Where-Object { $_.Name -ne "data2_sft" }

foreach ($d in $dirs) {
    if (-not (Test-Path (Join-Path $d.FullName "question.json"))) {
        Write-Warning "skip refine (no question.json): $($d.Name)"
        continue
    }
    Write-Host "refine $($d.Name) ..."
    python bridge/gen_training_data.py --stage refine --out $d.FullName
}

$inputs = ($dirs | ForEach-Object { $_.FullName }) -join " "
Write-Host "export -> training_data/data2_sft ..."
python bridge/export_spatial_train.py `
    --inputs $($dirs | ForEach-Object { $_.FullName }) `
    --views-root 3DGS/test2 `
    --out training_data/data2_sft `
    --source question

Write-Host "[done] refine + export"
