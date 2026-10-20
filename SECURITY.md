# Security and local configuration

## Credentials

The default dashboard, offline test suite, and current public data-download
endpoints do not need API credentials. If an integration requires authentication,
supply credentials through the tool's environment or OS credential store.

- Keep environment files, Streamlit secrets, Kaggle/Hugging Face credentials,
  private keys, and machine-specific YAML overrides local.
- `.gitignore` covers these files, virtual environments, datasets, model weights,
  experiment manifests, logs, caches, and temporary artifacts.
- Use `configs/*.local.yaml` for private paths or local overrides. Select the file
  in the dashboard or pass it through the relevant CLI's configuration argument.
- Use `git diff --cached` before publishing changes. Ignore rules do not remove
  credentials that were previously committed.

## Data and application boundary

Use de-identified MRI data and comply with the original publisher's usage terms.
Hashed scan labels are pseudonyms, not an anonymization guarantee. Local dataset
and experiment metadata may contain filesystem paths and are excluded from Git.

Streamlit binds to `127.0.0.1` by default. The application is a local research
tool; authentication and public hosting are outside the current implementation.
Tensor loads use restricted PyTorch deserialization and NumPy loads disable
pickled objects. Artifact paths and content hashes are checked at pipeline boundaries.

## Reporting a vulnerability

Use [GitHub's private vulnerability reporting form](https://github.com/reetamdan/3d-mri/security/advisories/new)
for security issues. Include the affected version, reproduction steps, and impact.
Do not include patient data, live credentials, or private datasets in a report.
