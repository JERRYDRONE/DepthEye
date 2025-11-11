# Convierte `ingresosXAUUSD.csv` e `ingresosXAUUSD40.csv` in-place a UTC usando offset fijo -6
$python = Join-Path -Path $PSScriptRoot -ChildPath '.venv\Scripts\python.exe'
$script = Join-Path -Path $PSScriptRoot -ChildPath 'adjust_timestamps.py'

if (-not (Test-Path $python)) {
    Write-Error "No se encontró el ejecutable de Python en $python. Activa el virtualenv o ajusta la ruta."
    exit 1
}

$files = @('ingresosXAUUSD.csv','ingresosXAUUSD40.csv')
foreach ($f in $files) {
    $in = Join-Path -Path $PSScriptRoot -ChildPath $f
    if (-not (Test-Path $in)) {
        Write-Warning "Archivo no encontrado: $in - se omite"
        continue
    }
    Write-Host "Procesando: $in"
    & $python $script --input $in --timestamp-col 0 --fixed-offset -6 --inplace
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Error procesando $in (exit $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
}

Write-Host "Conversión finalizada. Los archivos se modificaron in-place."
