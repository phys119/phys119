# PHYS119 Lab Deployment Guide

## Server Setup (First Time)

**Prerequisites:** Python 3.7+, git, GitHub credentials (SSH key or personal access token with repo write access)

```bash
git clone https://github.com/phys119/phys119.git
cd phys119
python phys119_deploy.py status
```

The status command confirms everything is working and shows what is currently deployed.

---

## Quick Start: Deploy Lab00 and Lab01

To release Lab00 and Lab01 to students at the start of the semester, run:

```bash
python phys119_deploy.py initialize 1
```

This will:
1. Create/reset the `main` branch with only Lab00–Lab01
2. Generate nbgitpuller links pointing to `main`
3. Return you to the `dev` branch for continued work

---

## Common Commands

### 1. Deploy Initial Labs (Lab00 and Lab01)
```bash
python phys119_deploy.py initialize 1
```

### 2. Check Current Status
```bash
python phys119_deploy.py status
```

### 3. Add Another Lab (e.g., Lab02)
```bash
python phys119_deploy.py update 2
```

### 4. Generate Student Links

**Link to all current labs:**
```bash
python phys119_deploy.py link
```

**Link to specific lab:**
```bash
python phys119_deploy.py link 0    # Lab00
python phys119_deploy.py link 1    # Lab01
python phys119_deploy.py link 2    # Lab02
```

---

## How It Works

### The Two-Branch Strategy

This system uses two branches:

- **dev branch** = All labs (Lab00–Lab10) — instructor working branch
- **main branch** = Only released labs — students access this (production)

This follows standard Git conventions where:
- `dev` is where all work happens
- `main` is the stable branch that users see

### Week-by-Week Workflow

**Week 1 (Before course starts):**
```bash
python phys119_deploy.py initialize 1
# Script generates nbgitpuller links — share these with students
```

**Week 2 (Release Lab02):**
```bash
python phys119_deploy.py update 2
# Script generates Lab02 links — share with students
```

**Week 3 (Release Lab03):**
```bash
python phys119_deploy.py update 3
```

And so on through Lab10.

---

## Understanding nbgitpuller Links

The generated links point to the `main` branch:
```
https://phys119.phas.ubc.ca/hub/user-redirect/git-pull
  ?repo=https://github.com/phys119/phys119
  &branch=main
  &urlpath=lab/tree/phys119/Lab01/Lab01.ipynb
```

**What happens when students click:**
1. Opens JupyterHub
2. Clones/updates from the `main` branch
3. Opens the specified lab notebook
4. Students only see released labs

---

## Making Updates to Lab Content

### Updating a future lab (not yet deployed):

```bash
git checkout dev
# Edit Lab05 files
git add Lab05/
git commit -m "Update Lab05 instructions"
git push origin dev
# When ready to deploy:
python phys119_deploy.py update 5
```

### Fixing an already-deployed lab:

```bash
git checkout dev
# Fix typo in Lab01
git add Lab01/
git commit -m "Fix typo in Lab01"
git push origin dev
# Re-deploy using current max lab number:
python phys119_deploy.py update 2
```

Students will receive the fix next time they click the nbgitpuller link.

---

## Important Notes

### Branch Structure
- **dev**: Your working branch with all content (Lab00–Lab10)
- **main**: Production branch with only deployed content
- After running any deploy command, you are returned to `dev` automatically

### First Time Only
- Use `initialize` for the very first deployment of a new semester
- This creates/resets `main` with only the specified labs

### Weekly Updates
- Use `update` to add new labs or push fixes to already-deployed labs
- Always specify the **highest lab number** to include
- Example: `update 3` means "include Labs 0, 1, 2, AND 3"

### Safety
- The script checks you are in the correct repository
- Your `dev` branch is never modified by the script
- All deployment happens on the `main` branch
- You are always returned to `dev` after operations

---

## Configuration File

`phys119_config.json` contains the repo URL, hub URL, branch names, and lab names. Edit it if these change:

```json
{
  "total_labs": 11,
  "repo_url": "https://github.com/phys119/phys119",
  "hub_url": "https://phys119.phas.ubc.ca",
  "current_branch": "main",
  "dev_branch": "dev",
  "lab_names": {
    "0": "Introduction to Jupyter Notebooks",
    "1": "Hooke's Law I",
    ...
  }
}
```

---

## Troubleshooting

### "Not in a git repository"
Make sure you are inside the `phys119` directory:
```bash
cd phys119
```

### "Branch doesn't exist" when running update
You need to initialize first:
```bash
python phys119_deploy.py initialize 1
```

### Check what's currently deployed
```bash
python phys119_deploy.py status
```

### Start over from scratch
```bash
# initialize recreates main from scratch
python phys119_deploy.py initialize 1
```

---

## GitHub Default Branch

**Important:** Confirm `main` is set as the default branch on GitHub:

1. Go to: https://github.com/phys119/phys119/settings/branches
2. Set "Default branch" to `main`

This ensures:
- Anyone cloning the repo gets production content by default
- GitHub shows production content when browsing

---

## Script Help

```bash
python phys119_deploy.py          # Show all commands and examples
python phys119_deploy.py status   # Check what is deployed
```
