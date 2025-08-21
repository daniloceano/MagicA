# GitHub Actions Workflows

This directory contains automated workflows for the MagicA project.

## Workflows

### `docs.yml` - Documentation Build and Deploy
**Triggers**: Push to `main`, Pull Requests to `main`

**Features**:
- ✅ Builds Sphinx documentation
- ✅ Deploys to GitHub Pages on push to main
- ✅ Creates documentation artifacts for PRs
- ✅ Adds preview comments to PRs
- ✅ Caches dependencies for faster builds

**Outputs**:
- **GitHub Pages**: https://daniloceano.github.io/MagicA
- **Artifacts**: Documentation HTML files (30-day retention)

### `docs-quality.yml` - Documentation Quality Check
**Triggers**: Pull Requests affecting documentation or Python files

**Features**:
- ✅ Validates RST syntax
- ✅ Checks for broken links
- ✅ Measures docstring coverage
- ✅ Generates coverage badges
- ✅ Ensures documentation builds without errors

**Outputs**:
- **Quality Report**: Build logs and validation results
- **Coverage Badge**: Docstring coverage visualization

## Setup Requirements

### GitHub Pages
1. Go to repository **Settings** → **Pages**
2. Set source to **GitHub Actions**
3. Documentation will be available at: `https://daniloceano.github.io/MagicA`

### Workflow Permissions
The workflows use `GITHUB_TOKEN` which is automatically provided by GitHub.
No additional setup required.

## Local Testing

Test documentation builds locally before pushing:

```bash
cd docs
pip install -r requirements.txt
make html
make linkcheck
```
