# Compilation Script for Windows
# LZ77 Compression Project - Build All Implementations

param(
    [switch]$All = $false,
    [switch]$Serial = $false,
    [switch]$OpenMP = $false,
    [switch]$MPI = $false,
    [switch]$Hybrid = $false,
    [switch]$CUDA = $false,
    [switch]$Clean = $false,
    [switch]$Help = $false
)

$ErrorActionPreference = "Continue"

function Show-Help {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "  LZ77 Compression Build Script" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\scripts\build.ps1 [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Green
    Write-Host "  -All       Build all implementations"
    Write-Host "  -Serial    Build serial version only"
    Write-Host "  -OpenMP    Build OpenMP version only"
    Write-Host "  -MPI       Build MPI version only"
    Write-Host "  -Hybrid    Build MPI+OpenMP hybrid only"
    Write-Host "  -CUDA      Build CUDA version only"
    Write-Host "  -Clean     Remove all compiled executables"
    Write-Host "  -Help      Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  .\scripts\build.ps1 -All           # Build everything"
    Write-Host "  .\scripts\build.ps1 -Serial        # Build serial only"
    Write-Host "  .\scripts\build.ps1 -OpenMP -MPI   # Build OpenMP and MPI"
    Write-Host "  .\scripts\build.ps1 -Clean         # Clean all builds"
    Write-Host ""
}

function Build-Serial {
    Write-Host "`n=== Building Serial LZ77 ===" -ForegroundColor Cyan
    Push-Location serial
    
    if (Get-Command gcc -ErrorAction SilentlyContinue) {
        Write-Host "Compiling with GCC..." -ForegroundColor Yellow
        gcc -O3 -o serial_lz77.exe serial_lz77.c
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Serial build successful!" -ForegroundColor Green
        } else {
            Write-Host "✗ Serial build failed!" -ForegroundColor Red
        }
    } else {
        Write-Host "✗ GCC not found. Please install MinGW." -ForegroundColor Red
    }
    
    Pop-Location
}

function Build-OpenMP {
    Write-Host "`n=== Building OpenMP LZ77 ===" -ForegroundColor Cyan
    Push-Location openmp
    
    if (Get-Command gcc -ErrorAction SilentlyContinue) {
        Write-Host "Compiling with GCC and OpenMP..." -ForegroundColor Yellow
        gcc -O3 -fopenmp -o openmp_lz77.exe openmp_lz77.c
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ OpenMP build successful!" -ForegroundColor Green
        } else {
            Write-Host "✗ OpenMP build failed!" -ForegroundColor Red
        }
    } else {
        Write-Host "✗ GCC not found. Please install MinGW." -ForegroundColor Red
    }
    
    Pop-Location
}

function Build-PThread {
    Write-Host "`n=== Building POSIX Threads LZ77 ===" -ForegroundColor Cyan
    Push-Location pthread
    
    Write-Host "Note: POSIX Threads may have limited support on Windows." -ForegroundColor Yellow
    Write-Host "Consider using WSL or Linux for PThreads implementation." -ForegroundColor Yellow
    
    if (Get-Command gcc -ErrorAction SilentlyContinue) {
        Write-Host "Attempting to compile with pthread..." -ForegroundColor Yellow
        gcc -O3 -pthread -o pthread_lz77.exe pthread_lz77.c
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ PThreads build successful!" -ForegroundColor Green
        } else {
            Write-Host "✗ PThreads build failed (expected on Windows without pthread library)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "✗ GCC not found." -ForegroundColor Red
    }
    
    Pop-Location
}

function Build-MPI {
    Write-Host "`n=== Building MPI LZ77 ===" -ForegroundColor Cyan
    Push-Location mpi
    
    if (Get-Command mpicc -ErrorAction SilentlyContinue) {
        Write-Host "Compiling with MPICC..." -ForegroundColor Yellow
        mpicc -O3 -o mpi_lz77.exe mpi_lz77.c
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ MPI build successful!" -ForegroundColor Green
        } else {
            Write-Host "✗ MPI build failed!" -ForegroundColor Red
        }
    } elseif (Get-Command cl -ErrorAction SilentlyContinue) {
        Write-Host "Attempting to compile with MSVC and MS-MPI..." -ForegroundColor Yellow
        $MPIInclude = "C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
        $MPILib = "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"
        
        if (Test-Path $MPIInclude) {
            cl /O2 /I"$MPIInclude" mpi_lz77.c /link /LIBPATH:"$MPILib" msmpi.lib /OUT:mpi_lz77.exe
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✓ MPI build successful!" -ForegroundColor Green
            } else {
                Write-Host "✗ MPI build failed!" -ForegroundColor Red
            }
        } else {
            Write-Host "✗ MS-MPI SDK not found at expected location." -ForegroundColor Red
            Write-Host "  Please install MS-MPI SDK from:" -ForegroundColor Yellow
            Write-Host "  https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi" -ForegroundColor Yellow
        }
    } else {
        Write-Host "✗ Neither mpicc nor MSVC found." -ForegroundColor Red
    }
    
    Pop-Location
}

function Build-Hybrid {
    Write-Host "`n=== Building MPI+OpenMP Hybrid LZ77 ===" -ForegroundColor Cyan
    Push-Location hybrid
    
    if (Get-Command mpicc -ErrorAction SilentlyContinue) {
        Write-Host "Compiling with MPICC and OpenMP..." -ForegroundColor Yellow
        mpicc -O3 -fopenmp -o mpi_openmp_lz77.exe mpi_openmp_lz77.c
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ MPI+OpenMP hybrid build successful!" -ForegroundColor Green
        } else {
            Write-Host "✗ Hybrid build failed!" -ForegroundColor Red
        }
    } else {
        Write-Host "✗ MPI compiler not found." -ForegroundColor Red
    }
    
    Pop-Location
}

function Build-CUDA {
    Write-Host "`n=== Building CUDA LZ77 ===" -ForegroundColor Cyan
    Push-Location cuda
    
    if (Get-Command nvcc -ErrorAction SilentlyContinue) {
        Write-Host "Detecting GPU compute capability..." -ForegroundColor Yellow
        
        # Try to detect GPU compute capability
        $gpuInfo = nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null
        if ($gpuInfo) {
            $computeCap = $gpuInfo.Trim().Replace(".", "")
            $arch = "sm_$computeCap"
            Write-Host "Detected compute capability: $arch" -ForegroundColor Green
        } else {
            $arch = "sm_75"  # Default to RTX 20 series
            Write-Host "Could not detect GPU, using default: $arch" -ForegroundColor Yellow
        }
        
        Write-Host "Compiling with NVCC (arch=$arch)..." -ForegroundColor Yellow
        nvcc -O3 -arch=$arch -o cuda_lz77.exe cuda_lz77.cu
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ CUDA build successful!" -ForegroundColor Green
        } else {
            Write-Host "✗ CUDA build failed!" -ForegroundColor Red
            Write-Host "  Try adjusting the compute capability (-arch=sm_XX)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "✗ NVCC not found. CUDA Toolkit may not be installed." -ForegroundColor Red
        Write-Host "  Download from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
    }
    
    Pop-Location
}

function Build-CUDAHybrid {
    Write-Host "`n=== Building MPI+CUDA Hybrid LZ77 ===" -ForegroundColor Cyan
    Push-Location hybrid
    
    if ((Get-Command nvcc -ErrorAction SilentlyContinue) -and 
        (Get-Command mpicc -ErrorAction SilentlyContinue)) {
        
        $arch = "sm_75"  # Default
        Write-Host "Compiling with NVCC and MPI (arch=$arch)..." -ForegroundColor Yellow
        nvcc -O3 -arch=$arch -o mpi_cuda_lz77.exe mpi_cuda_lz77.cu -lmsmpi
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ MPI+CUDA hybrid build successful!" -ForegroundColor Green
        } else {
            Write-Host "✗ MPI+CUDA hybrid build failed!" -ForegroundColor Red
        }
    } else {
        Write-Host "✗ Both NVCC and MPI are required for this build." -ForegroundColor Red
    }
    
    Pop-Location
}

function Clean-Builds {
    Write-Host "`n=== Cleaning Build Artifacts ===" -ForegroundColor Cyan
    
    $executables = @(
        "serial\serial_lz77.exe",
        "openmp\openmp_lz77.exe",
        "pthread\pthread_lz77.exe",
        "mpi\mpi_lz77.exe",
        "hybrid\mpi_openmp_lz77.exe",
        "cuda\cuda_lz77.exe",
        "hybrid\mpi_cuda_lz77.exe"
    )
    
    foreach ($exe in $executables) {
        if (Test-Path $exe) {
            Remove-Item $exe -Force
            Write-Host "Removed: $exe" -ForegroundColor Yellow
        }
    }
    
    # Clean output directory
    if (Test-Path "output") {
        Remove-Item "output\*.lz77" -Force -ErrorAction SilentlyContinue
        Remove-Item "output\*.txt" -Force -ErrorAction SilentlyContinue
        Write-Host "Cleaned output directory" -ForegroundColor Yellow
    }
    
    Write-Host "✓ Clean complete!" -ForegroundColor Green
}

# Main script logic
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  LZ77 Compression Build Script" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

if ($Help) {
    Show-Help
    exit 0
}

if ($Clean) {
    Clean-Builds
    exit 0
}

# Create output directory if it doesn't exist
if (-not (Test-Path "output")) {
    New-Item -ItemType Directory -Force -Path "output" | Out-Null
}

# If no specific flags, assume -All
if (-not ($Serial -or $OpenMP -or $MPI -or $Hybrid -or $CUDA)) {
    $All = $true
}

if ($All -or $Serial) {
    Build-Serial
}

if ($All -or $OpenMP) {
    Build-OpenMP
}

if ($All) {
    Build-PThread
}

if ($All -or $MPI) {
    Build-MPI
}

if ($All -or $Hybrid) {
    Build-Hybrid
}

if ($All -or $CUDA) {
    Build-CUDA
    Build-CUDAHybrid
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Build process complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Generate test data: python scripts\generate_test_data.py"
Write-Host "2. Run a test: .\serial\serial_lz77.exe datasets\test_files\1MB_mixed.txt output\test.lz77"
Write-Host "3. See QUICKSTART.md for more examples"
Write-Host ""
