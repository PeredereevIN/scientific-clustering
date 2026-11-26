param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [switch]$EnglishOnly,
    [switch]$RussianOnly
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path $Python)) { throw "Python interpreter not found: $Python" }
& $Python -m spacy validate
if ($LASTEXITCODE -ne 0) { throw "spaCy validation failed" }
$models = @("en_core_web_sm", "ru_core_news_sm")
if ($EnglishOnly) { $models = @("en_core_web_sm") }
if ($RussianOnly) { $models = @("ru_core_news_sm") }
foreach ($model in $models) {
    Write-Host "Installing spaCy model: $model"
    & $Python -m spacy download $model
    if ($LASTEXITCODE -ne 0) { throw "Could not install $model" }
}
Write-Host "Linguistic models are ready."
