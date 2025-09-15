# ==========================
# YOLOX-s setup script (Windows PowerShell)
# ==========================

# Останавливаем выполнение при ошибках
$ErrorActionPreference = "Stop"

# --- 0. Настройка .gitignore ---
$gitignorePath = ".gitignore"
$ignoreRules = @(
    "/models/",
    "/outputs/",
    "/assets/dog.jpg"
)

if (-Not (Test-Path $gitignorePath)) {
    New-Item -Path $gitignorePath -ItemType File -Force | Out-Null
}

foreach ($rule in $ignoreRules) {
    $exists = Select-String -Path $gitignorePath -Pattern [Regex]::Escape($rule) -Quiet
    if (-Not $exists) {
        Add-Content -Path $gitignorePath -Value $rule
        Write-Host "Added '$rule' to .gitignore"
    }
}

# --- 1. Папки ---
$modelsDir = "models"
$assetsDir = "assets"
$outputsDir = "outputs"

New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
New-Item -ItemType Directory -Force -Path $assetsDir | Out-Null
New-Item -ItemType Directory -Force -Path $outputsDir | Out-Null

# --- 2. Ссылки ---
$yoloxPthUrl  = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.0/yolox_s.pth"
$yoloxOnnxUrl = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.0/yolox_s.onnx"
$yoloxXmlUrl  = "https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/models/public/yolox-s/yolox-s.xml"
$yoloxBinUrl  = "https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/models/public/yolox-s/yolox-s.bin"
$dogImgUrl    = "https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/assets/dog.jpg"

# --- 3. Скачивание моделей ---
if (-Not (Test-Path "$modelsDir\yolox_s.pth")) {
    Write-Host "Downloading YOLOX-s PyTorch checkpoint..."
    Invoke-WebRequest -Uri $yoloxPthUrl -OutFile "$modelsDir\yolox_s.pth"
} else {
    Write-Host "✔ yolox_s.pth already exists, skipping download."
}

if (-Not (Test-Path "$modelsDir\yolox_s.onnx")) {
    Write-Host "Downloading YOLOX-s ONNX model..."
    Invoke-WebRequest -Uri $yoloxOnnxUrl -OutFile "$modelsDir\yolox_s.onnx"
} else {
    Write-Host "✔ yolox_s.onnx already exists, skipping download."
}

if (-Not (Test-Path "$modelsDir\yolox_s.xml")) {
    Write-Host "Downloading YOLOX-s OpenVINO IR (xml)..."
    Invoke-WebRequest -Uri $yoloxXmlUrl -OutFile "$modelsDir\yolox_s.xml"
} else {
    Write-Host "✔ yolox_s.xml already exists, skipping download."
}

if (-Not (Test-Path "$modelsDir\yolox_s.bin")) {
    Write-Host "Downloading YOLOX-s OpenVINO IR (bin)..."
    Invoke-WebRequest -Uri $yoloxBinUrl -OutFile "$modelsDir\yolox_s.bin"
} else {
    Write-Host "✔ yolox_s.bin already exists, skipping download."
}

# --- 4. Картинка для теста ---
if (-Not (Test-Path "$assetsDir\dog.jpg")) {
    Write-Host "Downloading test image..."
    Invoke-WebRequest -Uri $dogImgUrl -OutFile "$assetsDir\dog.jpg"
}
else {
    Write-Host "✔ dog.jpg already exists, skipping download."
}

# --- 5. Установка зависимостей ---
Write-Host "Installing Python dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# --- 6. Запуск инференса через ONNXRuntime ---
Write-Host "Running ONNXRuntime inference..."
python src\inference_onnx.py -m "$modelsDir\yolox_s.onnx" -i "$assetsDir\dog.jpg" -o "$outputsDir"

Write-Host "✅ All done! Results saved in $outputsDir"
