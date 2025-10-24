"""
Check for and list CI artifacts (test logs)
"""
import requests
import sys

def check_artifacts(run_id='18789808790'):
    """Check artifacts for a GitHub Actions run"""
    url = f'https://api.github.com/repos/mdshoaibuddinchanda/autoprepml/actions/runs/{run_id}/artifacts'
    
    print(f"🔍 Checking artifacts for run #{run_id}...")
    print("="*80)
    
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"❌ Failed to fetch artifacts: {response.status_code}")
        return
    
    data = response.json()
    artifacts = data.get('artifacts', [])
    
    if not artifacts:
        print("ℹ️  No artifacts found (tests may have all passed)")
        return
    
    print(f"\n📦 Found {len(artifacts)} artifact(s):\n")
    
    for artifact in artifacts:
        name = artifact['name']
        size_mb = artifact['size_in_bytes'] / (1024 * 1024)
        created = artifact['created_at']
        expired = artifact['expired']
        
        print(f"  📄 {name}")
        print(f"     Size: {size_mb:.2f} MB")
        print(f"     Created: {created}")
        print(f"     Expired: {expired}")
        print(f"     Download: {artifact['archive_download_url']}")
        print()
        
        # Show which test failed
        if 'failed-test-logs' in name:
            parts = name.split('-')
            if len(parts) >= 5:
                os_name = parts[3]
                py_version = parts[4] if len(parts) > 4 else 'unknown'
                print(f"     🔴 Failed on: {os_name}, Python {py_version}")
                print()
    
    print("="*80)
    print("\n💡 To download artifacts:")
    print("   1. Go to: https://github.com/mdshoaibuddinchanda/autoprepml/actions")
    print(f"   2. Click on run #{run_id}")
    print("   3. Scroll to bottom and download the artifact")
    print("   4. Extract and view pytest.log for full error details")

if __name__ == '__main__':
    run_id = sys.argv[1] if len(sys.argv) > 1 else '18789808790'
    check_artifacts(run_id)
