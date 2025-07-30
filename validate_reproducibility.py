#!/usr/bin/env python3
"""
Reproducibility Validation Script for AEnabledLoReg

This script performs comprehensive validation checks to ensure the reproducibility
pipeline is properly configured and ready for execution. It addresses several
items from the Biometrical Journal checklist by validating:

1. File organization and naming conventions
2. Code accessibility and documentation
3. Dependency availability
4. Data pipeline integrity
5. Output directory structure

Usage: python validate_reproducibility.py
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def check_mark():
    return "✅"

def cross_mark():
    return "❌"

def warning_mark():
    return "⚠️"

class ReproducibilityValidator:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.errors = []
        self.warnings = []
        self.passed_checks = []
        
    def log_error(self, message):
        self.errors.append(message)
        print(f"{cross_mark()} {message}")
        
    def log_warning(self, message):
        self.warnings.append(message)
        print(f"{warning_mark()} {message}")
        
    def log_success(self, message):
        self.passed_checks.append(message)
        print(f"{check_mark()} {message}")
    
    def check_directory_structure(self):
        """Validate the expected directory structure exists."""
        print("\n1. Checking Directory Structure...")
        
        required_dirs = [
            "src",
            "src/data_preparation",
            "src/methods",
            "src/utilities",
            "data",
            "data/raw",
            "data/processed",
            "results",
            "results/figures",
        ]
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                self.log_success(f"Directory exists: {dir_path}")
            else:
                self.log_error(f"Missing required directory: {dir_path}")
                
    def check_required_files(self):
        """Check that all required files are present."""
        print("\n2. Checking Required Files...")
        
        required_files = [
            "README.md",
            "requirements.txt",
            "run_reproducibility_pipeline.sh",
            "SYNTHETIC_DATA_DOCUMENTATION.md",
            "src/singlesite_prevent.script.py",
            "src/data_preparation/create_toy_dataset.py",
            "src/data_preparation/prevent_dataprep_synthetic.R",
            "src/data_preparation/prevent_dataprep_allinone.R",
            "src/methods/single_site_ae_loreg_module.py",
            "src/utilities/loregs.py",
            "src/utilities/weights.py",
            "all_variables.md",
            "checklist-for-authors.md"
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.log_success(f"File exists: {file_path}")
            else:
                self.log_error(f"Missing required file: {file_path}")
                
    def check_python_dependencies(self):
        """Check Python dependencies are available."""
        print("\n3. Checking Python Dependencies...")
        
        required_packages = [
            'pandas',
            'numpy', 
            'torch',
            'sklearn',
            'scipy',
            'matplotlib',
            'plotly',
            'tqdm'
        ]
        
        for package in required_packages:
            try:
                importlib.import_module(package)
                self.log_success(f"Python package available: {package}")
            except ImportError:
                self.log_error(f"Missing Python package: {package}")
                
    def check_r_availability(self):
        """Check R and required R packages."""
        print("\n4. Checking R Environment...")
        
        # Check R installation
        try:
            result = subprocess.run(['Rscript', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.log_success("R is available")
            else:
                self.log_error("R is not properly installed")
                return
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.log_error("R is not available in PATH")
            return
            
        # Check R packages
        r_packages = ['dplyr', 'haven', 'VIM', 'DataExplorer']
        r_check_script = """
required_packages <- c('dplyr', 'haven', 'VIM', 'DataExplorer')
for(pkg in required_packages) {
    if(require(pkg, quietly=TRUE, character.only=TRUE)) {
        cat('SUCCESS:', pkg, '\n')
    } else {
        cat('MISSING:', pkg, '\n')
    }
}
"""
        
        try:
            result = subprocess.run(['Rscript', '-e', r_check_script],
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('SUCCESS:'):
                        pkg = line.split(':')[1].strip()
                        self.log_success(f"R package available: {pkg}")
                    elif line.startswith('MISSING:'):
                        pkg = line.split(':')[1].strip()
                        self.log_error(f"Missing R package: {pkg}")
            else:
                self.log_error("Failed to check R packages")
        except subprocess.TimeoutExpired:
            self.log_error("R package check timed out")
            
    def check_script_permissions(self):
        """Check that scripts have proper permissions."""
        print("\n5. Checking Script Permissions...")
        
        executable_scripts = [
            "run_reproducibility_pipeline.sh"
        ]
        
        for script in executable_scripts:
            script_path = self.project_root / script
            if script_path.exists():
                if os.access(script_path, os.X_OK):
                    self.log_success(f"Script is executable: {script}")
                else:
                    self.log_warning(f"Script not executable (run 'chmod +x {script}'): {script}")
            else:
                self.log_error(f"Script not found: {script}")
                
    def check_code_quality(self):
        """Basic code quality checks."""
        print("\n6. Checking Code Quality...")
        
        # Check for proper file extensions
        python_files = list(self.project_root.rglob("*.py"))
        r_files = list(self.project_root.rglob("*.R"))
        
        if python_files:
            self.log_success(f"Found {len(python_files)} Python files with proper .py extension")
        else:
            self.log_warning("No Python files found")
            
        if r_files:
            self.log_success(f"Found {len(r_files)} R files with proper .R extension")
        else:
            self.log_warning("No R files found")
            
        # Check for README content
        readme_path = self.project_root / "README.md"
        if readme_path.exists():
            content = readme_path.read_text()
            if len(content) > 1000:  # Reasonable minimum for documentation
                self.log_success("README.md contains substantial documentation")
            else:
                self.log_warning("README.md seems too short")
                
    def check_checklist_compliance(self):
        """Check compliance with specific checklist items."""
        print("\n7. Checking Biometrical Journal Checklist Compliance...")
        
        # Check for master script
        if (self.project_root / "run_reproducibility_pipeline.sh").exists():
            self.log_success("Master script provided for easy reproduction")
        else:
            self.log_error("Missing master script for reproduction")
            
        # Check for synthetic data documentation
        if (self.project_root / "SYNTHETIC_DATA_DOCUMENTATION.md").exists():
            self.log_success("Synthetic data documentation provided")
        else:
            self.log_error("Missing synthetic data documentation")
            
        # Check for requirements specification
        if (self.project_root / "requirements.txt").exists():
            self.log_success("Python requirements specified")
        else:
            self.log_error("Missing requirements.txt file")
            
        # Check for proper file organization
        src_files = list((self.project_root / "src").rglob("*"))
        if len(src_files) >= 5:  # Should have multiple organized files
            self.log_success("Source code is properly organized in src/ directory")
        else:
            self.log_warning("Source code organization could be improved")
            
    def run_quick_test(self):
        """Run a quick test of the synthetic data generation."""
        print("\n8. Running Quick Integration Test...")
        
        try:
            # Test synthetic data generation
            test_script = """
import sys
sys.path.append('src/data_preparation')
from create_toy_dataset import create_toy_copd_dataset

# Generate small test dataset
df = create_toy_copd_dataset(n_patients=10, n_visits_per_patient=2)
print(f'SUCCESS: Generated test dataset with {len(df)} rows and {len(df.columns)} columns')
"""
            
            with open('temp_test_script.py', 'w') as f:
                f.write(test_script)
                
            result = subprocess.run(['python3', 'temp_test_script.py'],
                                  capture_output=True, text=True, timeout=60)
            
            os.remove('temp_test_script.py')
            
            if result.returncode == 0 and 'SUCCESS:' in result.stdout:
                self.log_success("Synthetic data generation test passed")
            else:
                self.log_error(f"Synthetic data generation test failed: {result.stderr}")
                
        except Exception as e:
            self.log_error(f"Quick integration test failed: {str(e)}")
            if os.path.exists('temp_test_script.py'):
                os.remove('temp_test_script.py')
                
    def generate_report(self):
        """Generate a summary report."""
        print("\n" + "="*60)
        print("REPRODUCIBILITY VALIDATION REPORT")
        print("="*60)
        
        print(f"\n✅ Passed Checks: {len(self.passed_checks)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"❌ Errors: {len(self.errors)}")
        
        if self.errors:
            print(f"\n{cross_mark()} ERRORS THAT MUST BE FIXED:")
            for error in self.errors:
                print(f"   - {error}")
                
        if self.warnings:
            print(f"\n{warning_mark()} WARNINGS (recommended to fix):")
            for warning in self.warnings:
                print(f"   - {warning}")
                
        print(f"\n{'='*60}")
        
        if len(self.errors) == 0:
            print(f"{check_mark()} VALIDATION PASSED - Ready for reproducibility testing!")
            print("\nNext steps:")
            print("1. Run quick test: ./run_reproducibility_pipeline.sh quick")
            print("2. Run full analysis: ./run_reproducibility_pipeline.sh full")
            return True
        else:
            print(f"{cross_mark()} VALIDATION FAILED - Please fix the errors above before proceeding.")
            return False
            
    def run_all_checks(self):
        """Run all validation checks."""
        print("AEnabledLoReg Reproducibility Validation")
        print("="*50)
        
        self.check_directory_structure()
        self.check_required_files()
        self.check_python_dependencies()
        self.check_r_availability()
        self.check_script_permissions()
        self.check_code_quality()
        self.check_checklist_compliance()
        self.run_quick_test()
        
        return self.generate_report()

if __name__ == "__main__":
    validator = ReproducibilityValidator()
    success = validator.run_all_checks()
    sys.exit(0 if success else 1)
