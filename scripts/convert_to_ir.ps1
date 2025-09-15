# Скрипт конвертации ONNX → OpenVINO IR (FP16)
# Использование: запустите в корне проекта после помещения ONNX в models\yolox_s\model.onnx
# Требуется: pip install openvino-dev

param(
    [string]$OnnxPath = "models/yolox_s/model.onnx",
    [string]$OutDir = "models/yolox_s",
    [string]$InputShape = "[1,3,640,640]",
    [switch]$UseMO
)

Write-Host "[convert_to_ir] ONNX:" $OnnxPath
Write-Host "[convert_to_ir] OUT_DIR:" $OutDir

if (!(Test-Path $OnnxPath)) {
    Write-Error "ONNX файл не найден: $OnnxPath"
    exit 1
}

# Создать папку вывода
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# Проверка наличия ovc/mo
$ovc = (Get-Command ovc -ErrorAction SilentlyContinue)
$mo = (Get-Command mo -ErrorAction SilentlyContinue)

if ($ovc -and -not $UseMO) {
    Write-Host "[convert_to_ir] Использую ovc"
    ovc $OnnxPath --output_dir $OutDir --compress_to_fp16 --input_shape $InputShape
}
elseif ($mo) {
    Write-Host "[convert_to_ir] Использую mo"
    mo --input_model $OnnxPath --output_dir $OutDir --data_type FP16 --input_shape $InputShape
}
else {
    Write-Error "Не найдены утилиты ovc/mo. Установите: pip install openvino-dev"
    exit 1
}

Write-Host "[convert_to_ir] Готово. Проверьте файлы .xml/.bin в" $OutDir
