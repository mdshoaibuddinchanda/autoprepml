# Security policy

## Supported versions

Security fixes are applied to the latest release and the active development
branch. Older versions should be upgraded before reporting a new issue.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use the private
security advisory form on the GitHub repository or contact the maintainer at
mdshoaibuddin.chanda@gmail.com with the subject `AutoPrepML security report`.

Include the affected version, a concise reproduction, impact, and any
suggested mitigation. Do not include personal data, API keys, downloaded
datasets, or other secrets in the report.

The maintainer will acknowledge a report within seven calendar days and will
coordinate a fix, disclosure timeline, and credit with the reporter.

## Security expectations

AutoPrepML can process sensitive data. Applications should use least
privilege, redact logs, keep credentials in environment variables or an
approved secret manager, and avoid sending raw records to optional LLM
providers. `DataPlan.load()` uses Python pickle internally; only load artifacts
from a trusted source after verifying their provenance.
