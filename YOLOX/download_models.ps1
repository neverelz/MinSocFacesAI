# ==========================
# YOLOX-s setup script (Windows PowerShell, ASCII clean)
# ==========================

$ErrorActionPreference = "Stop"

# --- .gitignore ---
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
    $pattern = [Regex]::Escape($rule)
    $exists = Select-String -Path $gitignorePath -Pattern $pattern -Quiet -ErrorAction SilentlyContinue
    if (-Not $exists) {
        Add-Content -Path $gitignorePath -Value $rule
        Write-Host "Added '$rule' to .gitignore"
    }
}

# --- папки ---
$modelsDir = "models"
$assetsDir = "assets"
$outputsDir = "outputs"

New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
New-Item -ItemType Directory -Force -Path $assetsDir | Out-Null
New-Item -ItemType Directory -Force -Path $outputsDir | Out-Null

# --- ссылки ---
$yoloxOnnxUrlPrimary = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"
$yoloxOnnxUrlHf = "https://huggingface.co/hr16/yolox-onnx/resolve/main/yolox_s.onnx"

$dogImgUrl = "https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/assets/dog.jpg"

# --- скачивание ONNX ---
$onnxPath = Join-Path $modelsDir "yolox_s.onnx"
if (-Not (Test-Path $onnxPath)) {
    Write-Host "Downloading YOLOX-s ONNX model (primary)..."
    try {
        Invoke-WebRequest -Uri $yoloxOnnxUrlPrimary -OutFile $onnxPath -UseBasicParsing -ErrorAction Stop
        Write-Host "Downloaded ONNX from primary URL."
    } catch {
        Write-Host "Primary ONNX download failed -- trying HuggingFace mirror..."
        try {
            Invoke-WebRequest -Uri $yoloxOnnxUrlHf -OutFile $onnxPath -UseBasicParsing -ErrorAction Stop
            Write-Host "Downloaded ONNX from HuggingFace mirror."
        } catch {
            Write-Error "Failed to download ONNX model from both primary and mirror. Please download manually and put in $onnxPath."
            exit 1
        }
    }
}
else {
    Write-Host "$onnxPath already exists, skipping download."
}

# --- тестовая картинка ---
$imgPath = Join-Path $assetsDir "dog.jpg"
if (-Not (Test-Path $imgPath)) {
    Write-Host "Downloading test image..."
    Invoke-WebRequest -Uri $dogImgUrl -OutFile $imgPath -UseBasicParsing
}
else {
    Write-Host "$imgPath already exists, skipping download."
}

# --- установка зависимостей ---
Write-Host "Installing Python dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# --- конвертация ONNX -> OpenVINO IR ---
$xmlPath = Join-Path $modelsDir "yolox_s.xml"
$binPath = Join-Path $modelsDir "yolox_s.bin"

if ((-Not (Test-Path $xmlPath)) -or (-Not (Test-Path $binPath))) {
    Write-Host "OpenVINO IR not found -- converting ONNX -> OpenVINO IR..."
    if (-Not (Test-Path "src\convert_to_openvino.py")) {
        Write-Error "src\convert_to_openvino.py not found. Place the converter script in src/ and re-run."
        exit 1
    }
    python src\convert_to_openvino.py -i $onnxPath -o $xmlPath
    if ((-Not (Test-Path $xmlPath)) -or (-Not (Test-Path $binPath))) {
        Write-Error "Conversion failed or output files not found ($xmlPath / $binPath)."
        exit 1
    }
    else {
        Write-Host "Conversion successful: $xmlPath and corresponding .bin created."
    }
}
else {
    Write-Host "OpenVINO IR files already exist, skipping conversion."
}

# --- запуск тестового инференса ---
Write-Host "Running ONNXRuntime inference test..."
python src\inference_onnx.py -m $onnxPath -i $imgPath -o $outputsDir

Write-Host "All done! Results (if any) in $outputsDir"
