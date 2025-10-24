"""
Diagnose CI test failures by simulating CI environment
"""
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and report results"""
    print(f"\n{'='*80}")
    print(f"🔍 {description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}")
    print()
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✅ SUCCESS (exit code {result.returncode})")
        if result.stdout:
            print(result.stdout[:500])  # First 500 chars
    else:
        print(f"❌ FAILED (exit code {result.returncode})")
        if result.stderr:
            print("STDERR:")
            print(result.stderr[:1000])
        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout[:1000])
    
    return result.returncode == 0

def main():
    print("🚀 CI Failure Diagnostic Tool")
    print("="*80)

    # Check Python version
    print(f"\n📌 Python: {sys.version}")
    print(f"📌 Platform: {sys.platform}")

    _extracted_from_main_10("TEST 1: Import all autoprepml modules")
    modules = [
        'autoprepml',
        'autoprepml.core',
        'autoprepml.detection',
        'autoprepml.cleaning',
        'autoprepml.autoeda',
        'autoprepml.feature_engine',
        'autoprepml.graph',
        'autoprepml.image',
        'autoprepml.text',
        'autoprepml.timeseries',
    ]

    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")

    # Test 2: Run pytest collection only (no execution)
    run_command(
        'pytest tests/ --collect-only -q',
        "TEST 2: Collect all tests (no execution)"
    )

    # Test 3: Run first test file only
    run_command(
        'pytest tests/test_core.py -v',
        "TEST 3: Run test_core.py"
    )

    # Test 4: Run tests with --lf (last failed)
    run_command(
        'pytest tests/ --lf -v --tb=short',
        "TEST 4: Run last failed tests with short traceback"
    )

    _extracted_from_main_10("TEST 5: Check for import-time execution issues")
    test_files = [
        'tests/test_dashboard.py',
        'tests/test_gemini.py',
        'tests/test_dynamic_config.py',
        'tests/test_all_functionality.py',
    ]

    for test_file in test_files:
        if os.path.exists(test_file):
            result = subprocess.run(
                f'python -c "import sys; sys.path.insert(0, \'tests\'); import {os.path.basename(test_file)[:-3]}"',
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"✅ {test_file}: imports cleanly")
            else:
                print(f"❌ {test_file}: FAILS ON IMPORT")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}")

    _extracted_from_main_10("📊 DIAGNOSTIC COMPLETE")


# TODO Rename this here and in `main`
def _extracted_from_main_10(arg0):
    # Test 1: Import all modules
    print("\n\n" + "="*80)
    print(arg0)
    print("="*80)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Diagnostic interrupted by user")
    except Exception as e:
        print(f"\n\n💥 Diagnostic tool crashed: {e}")
        import traceback
        traceback.print_exc()
