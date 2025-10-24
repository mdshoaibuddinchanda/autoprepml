import requests

# Get job details
r = requests.get('https://api.github.com/repos/mdshoaibuddinchanda/autoprepml/actions/runs/18776310520/jobs')
jobs = r.json()['jobs']

# Find the first failed job
for job in jobs:
    if job['conclusion'] == 'failure':
        print(f"\n❌ Failed Job: {job['name']}")
        print(f"🔗 URL: {job['html_url']}")
        
        # Get the log
        log_url = job['url'] + '/logs'
        headers = {'Accept': 'application/vnd.github.v3+json'}
        
        log_response = requests.get(log_url, headers=headers)
        if log_response.status_code == 200:
            logs = log_response.text
            
            # Find the error lines
            lines = logs.split('\n')
            for i, line in enumerate(lines):
                if 'FAILED' in line or 'ERROR' in line or 'Error:' in line:
                    # Print context
                    start = max(0, i-3)
                    end = min(len(lines), i+5)
                    print("\n--- Error Context ---")
                    for j in range(start, end):
                        print(lines[j])
                    break
        break
