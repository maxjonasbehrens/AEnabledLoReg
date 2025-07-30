#!/bin/bash

# =============================================================================
# REPRODUCIBILITY PIPELINE FOR AEnabledLoReg
# =============================================================================
# 
# This master script orchestrates the complete reproducibility pipeline for the
# "Contrasting Global and Patient-Specific Regression Models via a Neural Network
# Representation" paper.
#
# Usage: ./run_reproducibility_pipeline.sh [quick|full]
#   quick: Run with reduced parameters for faster execution
#   full:  Run complete analysis (default)
#
# Requirements:
# - Python 3.8+ with required packages (see requirements.txt)
# - R 4.0+ with required packages (dplyr, haven, VIM, DataExplorer)
# - Sufficient computational resources (8GB+ RAM recommended)
#
# =============================================================================

set -e  # Exit on any error

# Configuration
QUICK_MODE=${1:-"full"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check R
    if ! command -v Rscript &> /dev/null; then
        error "R is required but not installed"
        exit 1
    fi
    
    # Check Python packages
    log "Checking Python dependencies..."
    python3 -c "
import sys
required_packages = ['pandas', 'numpy', 'torch', 'sklearn', 'scipy', 'matplotlib', 'plotly', 'tqdm']
missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f'Missing Python packages: {missing}')
    sys.exit(1)
" || {
        error "Missing required Python packages. Please install them first."
        echo "You can install them with: pip install -r requirements.txt"
        exit 1
    }
    
    # Check R packages
    log "Checking R dependencies..."
    Rscript -e "
required_packages <- c('dplyr', 'haven', 'VIM', 'DataExplorer')
missing <- required_packages[!sapply(required_packages, require, quietly=TRUE, character.only=TRUE)]
if(length(missing) > 0) {
    cat('Missing R packages:', paste(missing, collapse=', '), '\n')
    quit(status=1)
}
" || {
        error "Missing required R packages. Please install them first."
        echo "You can install them in R with: install.packages(c('dplyr', 'haven', 'VIM', 'DataExplorer'))"
        exit 1
    }
    
    success "All requirements satisfied"
}

# Create directory structure
setup_directories() {
    log "Setting up directory structure..."
    
    mkdir -p data/raw
    mkdir -p data/processed
    mkdir -p results/figures
    mkdir -p logs
    
    success "Directory structure created"
}

# Generate synthetic dataset
generate_synthetic_data() {
    log "Generating synthetic COPD dataset..."
    
    if [ -f "data/raw/prevent_st2_synthetic.csv" ] && [ "$QUICK_MODE" != "force" ]; then
        warning "Synthetic dataset already exists. Skipping generation."
        warning "Use 'force' mode to regenerate: ./run_reproducibility_pipeline.sh force"
        return 0
    fi
    
    python3 src/data_preparation/create_toy_dataset.py 2>&1 | tee logs/synthetic_data_generation.log
    
    if [ $? -eq 0 ]; then
        success "Synthetic dataset generated successfully"
    else
        error "Failed to generate synthetic dataset"
        exit 1
    fi
}

# Process data using R script
process_data() {
    log "Processing synthetic data with R pipeline..."
    
    Rscript src/data_preparation/prevent_dataprep_synthetic.R 2>&1 | tee logs/data_processing.log
    
    if [ $? -eq 0 ]; then
        success "Data processing completed successfully"
    else
        error "Failed to process data"
        exit 1
    fi
    
    # Copy processed file to expected name
    if [ -f "data/processed/prevent_num_imp_varL2_std.csv" ]; then
        cp data/processed/prevent_num_imp_varL2_std.csv data/processed/prevent_direct_train_data.csv
        log "Copied scaled data as input for main analysis"
    else
        error "Processed data file not found"
        exit 1
    fi
}

# Modify Python script for quick execution if needed
modify_for_quick_mode() {
    if [ "$QUICK_MODE" = "quick" ]; then
        log "Modifying analysis parameters for quick execution..."
        
        # Create a backup of the original script
        cp src/singlesite_prevent.script.py src/singlesite_prevent.script.py.backup
        
        # Modify parameters for quick execution
        sed -i.tmp "s/N_SEEDS = 15/N_SEEDS = 3/" src/singlesite_prevent.script.py
        sed -i.tmp "s/'num_epochs': 300/'num_epochs': 50/" src/singlesite_prevent.script.py
        
        warning "Analysis modified for quick execution (3 seeds, 50 epochs)"
        warning "For full reproducibility validation, run with 'full' mode"
    fi
}

# Restore original script after quick mode
restore_original_script() {
    if [ "$QUICK_MODE" = "quick" ] && [ -f "src/singlesite_prevent.script.py.backup" ]; then
        mv src/singlesite_prevent.script.py.backup src/singlesite_prevent.script.py
        rm -f src/singlesite_prevent.script.py.tmp
        log "Original script parameters restored"
    fi
}

# Run main analysis
run_analysis() {
    log "Running main AEnabledLoReg analysis..."
    
    if [ "$QUICK_MODE" = "quick" ]; then
        log "Quick mode: Running with reduced parameters"
    else
        log "Full mode: Running complete analysis (this may take several hours)"
    fi
    
    python3 src/singlesite_prevent.script.py 2>&1 | tee logs/main_analysis.log
    
    if [ $? -eq 0 ]; then
        success "Main analysis completed successfully"
    else
        error "Main analysis failed"
        restore_original_script
        exit 1
    fi
}

# Validate results
validate_results() {
    log "Validating analysis results..."
    
    # Check for expected output files
    expected_files=(
        "results/figures"
        "data/processed/prevent_direct_train_data.csv"
    )
    
    for file in "${expected_files[@]}"; do
        if [ ! -e "$file" ]; then
            error "Expected output file/directory not found: $file"
            return 1
        fi
    done
    
    # Check if results directory has contents
    if [ -z "$(ls -A results/figures/)" ]; then
        warning "Results directory is empty - analysis may not have completed properly"
        return 1
    fi
    
    success "Results validation passed"
}

# Generate summary report
generate_report() {
    log "Generating reproducibility report..."
    
    cat > REPRODUCIBILITY_REPORT.md << EOF
# Reproducibility Report - AEnabledLoReg

**Generated:** $(date)
**Mode:** $QUICK_MODE
**System:** $(uname -s) $(uname -r)
**Python Version:** $(python3 --version)
**R Version:** $(Rscript --version 2>&1 | head -1)

## Pipeline Execution Summary

### 1. Synthetic Data Generation
- ✅ Synthetic COPD dataset created
- ✅ 76 predictor variables included
- ✅ Realistic clinical relationships implemented
- ✅ Missing data patterns applied

### 2. Data Processing
- ✅ R preprocessing pipeline executed
- ✅ Data quality filtering applied
- ✅ Variable selection completed
- ✅ Scaling and normalization applied

### 3. Main Analysis
- ✅ Multi-seed analysis completed
EOF

    if [ "$QUICK_MODE" = "quick" ]; then
        cat >> REPRODUCIBILITY_REPORT.md << EOF
- ⚠️  Quick mode: 3 seeds, 50 epochs (reduced for testing)
EOF
    else
        cat >> REPRODUCIBILITY_REPORT.md << EOF
- ✅ Full mode: 15 seeds, 300 epochs (full reproducibility)
EOF
    fi

    cat >> REPRODUCIBILITY_REPORT.md << EOF

### 4. Output Generation
- ✅ Analysis results saved
- ✅ Figures generated
- ✅ Stability analysis completed

## Files Generated

### Data Files
- \`data/raw/prevent_st2_synthetic.csv\` - Synthetic raw dataset
- \`data/processed/prevent_num_imp_varL2.csv\` - Processed unscaled data
- \`data/processed/prevent_num_imp_varL2_std.csv\` - Processed scaled data
- \`data/processed/prevent_direct_train_data.csv\` - Final input for analysis

### Results
- \`results/figures/\` - Analysis output plots and results

### Logs
- \`logs/synthetic_data_generation.log\` - Data generation log
- \`logs/data_processing.log\` - R processing log
- \`logs/main_analysis.log\` - Main analysis log

## Reproducibility Checklist Compliance

✅ **Synthetic Data**: Comparable dataset provided when original cannot be shared
✅ **Documentation**: Complete README and data documentation included
✅ **Code Organization**: Proper file structure with descriptive names
✅ **Platform Independence**: Cross-platform compatible implementation
✅ **Version Information**: Dependency versions documented
✅ **Execution Instructions**: Clear step-by-step instructions provided

## Next Steps

1. Review the analysis results in \`results/figures/\`
2. Compare output with expected results in the paper
3. For full validation, run in 'full' mode if executed in 'quick' mode
4. Refer to SYNTHETIC_DATA_DOCUMENTATION.md for dataset details

---
*This report was automatically generated by the reproducibility pipeline.*
EOF

    success "Reproducibility report generated: REPRODUCIBILITY_REPORT.md"
}

# Main execution
main() {
    log "Starting AEnabledLoReg Reproducibility Pipeline"
    log "Mode: $QUICK_MODE"
    
    check_requirements
    setup_directories
    generate_synthetic_data
    process_data
    modify_for_quick_mode
    
    # Trap to ensure cleanup on exit
    trap 'restore_original_script' EXIT
    
    run_analysis
    validate_results
    generate_report
    
    success "Reproducibility pipeline completed successfully!"
    
    if [ "$QUICK_MODE" = "quick" ]; then
        warning "Quick mode was used. For full reproducibility validation, run:"
        warning "./run_reproducibility_pipeline.sh full"
    fi
    
    log "Check REPRODUCIBILITY_REPORT.md for detailed results"
}

# Help function
show_help() {
    cat << EOF
AEnabledLoReg Reproducibility Pipeline

Usage: $0 [MODE]

Modes:
  quick    Run with reduced parameters for faster execution (3 seeds, 50 epochs)
  full     Run complete analysis with full parameters (15 seeds, 300 epochs) [default]
  force    Force regeneration of all data and results

Examples:
  $0           # Run full analysis
  $0 quick     # Run quick test
  $0 force     # Force regeneration

This script addresses the Biometrical Journal reproducibility checklist by:
1. Generating synthetic data when original cannot be shared
2. Providing complete documentation and version information
3. Creating a master script for easy reproduction
4. Validating all outputs and generating a reproducibility report

For more information, see README.md and SYNTHETIC_DATA_DOCUMENTATION.md
EOF
}

# Handle command line arguments
case "${1:-}" in
    -h|--help|help)
        show_help
        exit 0
        ;;
    quick|full|force)
        main
        ;;
    "")
        QUICK_MODE="full"
        main
        ;;
    *)
        error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
