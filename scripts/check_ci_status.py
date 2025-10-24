#!/usr/bin/env python3
"""Check GitHub Actions CI/CD status for autoprepml."""

import requests
import sys
from datetime import datetime

# Repository details
OWNER = "mdshoaibuddinchanda"
REPO = "autoprepml"
API_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/runs"

def get_latest_run():
    """Fetch the latest CI/CD run."""
    try:
        response = requests.get(API_URL, params={"per_page": 1})
        response.raise_for_status()
        data = response.json()
        
        if not data.get("workflow_runs"):
            print("❌ No CI/CD runs found")
            return None
        
        return data["workflow_runs"][0]
    except Exception as e:
        print(f"❌ Error fetching CI status: {e}")
        return None

def format_duration(seconds):
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes}m {secs}s"

def print_status(run):
    """Print formatted CI/CD status."""
    print("\n" + "="*60)
    print("🔍 CI/CD Status for autoprepml")
    print("="*60)
    
    status = run["status"]
    conclusion = run.get("conclusion", "N/A")
    
    # Status emoji
    if conclusion == "success":
        status_emoji = "✅"
        status_color = "SUCCESS"
    elif conclusion == "failure":
        status_emoji = "❌"
        status_color = "FAILED"
    elif conclusion == "cancelled":
        status_emoji = "⚠️"
        status_color = "CANCELLED"
    elif status == "in_progress":
        status_emoji = "🔄"
        status_color = "RUNNING"
    else:
        status_emoji = "⏳"
        status_color = status.upper()
    
    print(f"\n{status_emoji} Run #{run['run_number']}: {status_color}")
    print(f"   Commit: {run['head_sha'][:7]}")
    print(f"   Branch: {run['head_branch']}")
    print(f"   Event: {run['event']}")
    
    # Time info
    created_at = datetime.fromisoformat(run['created_at'].replace('Z', '+00:00'))
    updated_at = datetime.fromisoformat(run['updated_at'].replace('Z', '+00:00'))
    
    if run.get('run_started_at'):
        started_at = datetime.fromisoformat(run['run_started_at'].replace('Z', '+00:00'))
        now = datetime.now(started_at.tzinfo)
        
        if status == "completed":
            duration = (updated_at - started_at).total_seconds()
            print(f"   Duration: {format_duration(int(duration))}")
        else:
            elapsed = (now - started_at).total_seconds()
            print(f"   Running for: {format_duration(int(elapsed))}")
    
    print(f"\n🔗 View run: {run['html_url']}")
    print("="*60 + "\n")
    
    return conclusion

def get_job_details(run_id):
    """Fetch detailed job information."""
    try:
        jobs_url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/runs/{run_id}/jobs"
        response = requests.get(jobs_url)
        response.raise_for_status()
        data = response.json()
        
        print("📋 Job Details:")
        print("-" * 60)
        
        for job in data.get("jobs", []):
            name = job["name"]
            status = job["status"]
            conclusion = job.get("conclusion", "pending")
            
            if conclusion == "success":
                emoji = "✅"
            elif conclusion == "failure":
                emoji = "❌"
            elif conclusion == "skipped":
                emoji = "⏭️"
            elif status == "in_progress":
                emoji = "🔄"
            else:
                emoji = "⏳"
            
            duration = ""
            if job.get("started_at") and job.get("completed_at"):
                start = datetime.fromisoformat(job["started_at"].replace('Z', '+00:00'))
                end = datetime.fromisoformat(job["completed_at"].replace('Z', '+00:00'))
                dur = (end - start).total_seconds()
                duration = f" ({format_duration(int(dur))})"
            
            print(f"  {emoji} {name}: {status}{duration}")
            
            if conclusion == "failure":
                print(f"     🔗 {job['html_url']}")
        
        print("-" * 60 + "\n")
        
    except Exception as e:
        print(f"⚠️  Could not fetch job details: {e}\n")

def main():
    """Main function."""
    print("Checking CI/CD status...")
    
    run = get_latest_run()
    if not run:
        sys.exit(1)
    
    conclusion = print_status(run)
    get_job_details(run["id"])
    
    # Exit codes
    if conclusion == "success":
        print("🎉 All tests passed!\n")
        sys.exit(0)
    elif conclusion == "failure":
        print("❌ Tests failed. Check the logs above.\n")
        sys.exit(1)
    elif run["status"] == "in_progress":
        print("⏳ Tests are still running...\n")
        sys.exit(2)
    else:
        print(f"⚠️  Workflow status: {conclusion or run['status']}\n")
        sys.exit(3)

if __name__ == "__main__":
    main()
