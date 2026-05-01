#!/usr/bin/env python3
"""
PHYS119 Lab Deployment Script for Production
Manages a single 'current-labs' branch that gets updated with new labs each week
Compatible with Windows, macOS, and Linux
"""

import subprocess
import sys
import os
from datetime import datetime
import json

class PHYS119Deployer:
    def __init__(self, config_file='phys119_config.json'):
        """Initialize with PHYS119 configuration"""
        self.config = self.load_config(config_file)
        self.total_labs = self.config.get('total_labs', 11)  # Labs 0-10
        self.repo_url = self.config.get('repo_url', 'https://github.com/phys119/phys119')
        self.hub_url = self.config.get('hub_url', 'https://phys119.phas.ubc.ca')
        self.production_branch = self.config.get('current_branch', 'main')  # Branch for deployed labs (production)
        self.dev_branch = self.config.get('dev_branch', 'dev')  # Branch with all labs (development)
        self.lab_names = self.config.get('lab_names', {})

        # Extract repository name from URL (e.g., "phys119" from "https://github.com/phys119/phys119")
        self.repo_name = self.repo_url.rstrip('/').split('/')[-1]

    def get_lab_file_differences(self, max_lab):
        """
        Check for differences between local files and git repository for labs 0 through max_lab.
        Returns a dict with categorized file differences.
        """
        differences = {
            'modified': [],    # Files that exist in both but differ
            'untracked': [],   # Files that exist locally but not in git
            'deleted': [],     # Files that exist in git but not locally
        }

        # Get list of lab directories to check
        lab_dirs = [f"Lab{i:02d}" for i in range(max_lab + 1)]

        # Check for modified and deleted files (tracked files with changes)
        for lab_dir in lab_dirs:
            if not os.path.exists(lab_dir):
                continue

            # Get modified files in this lab directory
            result = self.run_command(f'git diff --name-only HEAD -- "{lab_dir}"', suppress_errors=True)
            if result.returncode == 0 and result.stdout.strip():
                for file_path in result.stdout.strip().split('\n'):
                    if file_path:
                        differences['modified'].append(file_path)

            # Get staged but uncommitted changes
            result = self.run_command(f'git diff --cached --name-only HEAD -- "{lab_dir}"', suppress_errors=True)
            if result.returncode == 0 and result.stdout.strip():
                for file_path in result.stdout.strip().split('\n'):
                    if file_path and file_path not in differences['modified']:
                        differences['modified'].append(file_path)

        # Check for untracked files in lab directories
        for lab_dir in lab_dirs:
            if not os.path.exists(lab_dir):
                continue

            result = self.run_command(f'git ls-files --others --exclude-standard "{lab_dir}"', suppress_errors=True)
            if result.returncode == 0 and result.stdout.strip():
                for file_path in result.stdout.strip().split('\n'):
                    if file_path:
                        differences['untracked'].append(file_path)

        # Check for deleted files (files in git index but not in working directory)
        for lab_dir in lab_dirs:
            result = self.run_command(f'git diff --name-only --diff-filter=D HEAD -- "{lab_dir}"', suppress_errors=True)
            if result.returncode == 0 and result.stdout.strip():
                for file_path in result.stdout.strip().split('\n'):
                    if file_path:
                        differences['deleted'].append(file_path)

        return differences

    def prompt_sync_decisions(self, differences):
        """
        Prompt user for sync direction for each differing file.
        Returns a dict with user decisions for each file.
        """
        decisions = {
            'commit_local': [],      # Files to commit (local -> git)
            'restore_git': [],       # Files to restore from git
            'add_new': [],           # New files to add to git
            'keep_deleted': [],      # Confirm deletion in git
            'restore_deleted': [],   # Restore deleted files from git
            'skip': []               # Files to skip/ignore
        }

        total_diffs = len(differences['modified']) + len(differences['untracked']) + len(differences['deleted'])

        if total_diffs == 0:
            print("\n  No differences found between local files and git repository.")
            return decisions

        print(f"\n{'='*60}")
        print("FILE DIFFERENCES DETECTED")
        print(f"{'='*60}")
        print(f"Found {total_diffs} file(s) with differences.\n")

        # Handle modified files
        if differences['modified']:
            print(f"\n--- MODIFIED FILES ({len(differences['modified'])}) ---")
            print("These files have local changes not yet committed.\n")

            for file_path in differences['modified']:
                print(f"\nFile: {file_path}")

                # Show a brief diff preview
                diff_result = self.run_command(f'git diff --stat HEAD -- "{file_path}"', suppress_errors=True)
                if diff_result.stdout.strip():
                    print(f"  Changes: {diff_result.stdout.strip().split(chr(10))[-1].strip()}")

                print("  Options:")
                print("    [L] Keep LOCAL version (commit to git)")
                print("    [G] Restore GIT version (discard local changes)")
                print("    [S] Skip (leave as-is, don't sync)")

                while True:
                    choice = input("  Your choice (L/G/S): ").strip().upper()
                    if choice == 'L':
                        decisions['commit_local'].append(file_path)
                        print(f"    -> Will commit local version")
                        break
                    elif choice == 'G':
                        decisions['restore_git'].append(file_path)
                        print(f"    -> Will restore from git")
                        break
                    elif choice == 'S':
                        decisions['skip'].append(file_path)
                        print(f"    -> Skipping")
                        break
                    else:
                        print("    Invalid choice. Please enter L, G, or S.")

        # Handle untracked files
        if differences['untracked']:
            print(f"\n--- NEW LOCAL FILES ({len(differences['untracked'])}) ---")
            print("These files exist locally but are not in git.\n")

            for file_path in differences['untracked']:
                print(f"\nFile: {file_path}")
                print("  Options:")
                print("    [A] ADD to git (include in deployment)")
                print("    [S] Skip (don't add to git)")

                while True:
                    choice = input("  Your choice (A/S): ").strip().upper()
                    if choice == 'A':
                        decisions['add_new'].append(file_path)
                        print(f"    -> Will add to git")
                        break
                    elif choice == 'S':
                        decisions['skip'].append(file_path)
                        print(f"    -> Skipping")
                        break
                    else:
                        print("    Invalid choice. Please enter A or S.")

        # Handle deleted files
        if differences['deleted']:
            print(f"\n--- DELETED FILES ({len(differences['deleted'])}) ---")
            print("These files exist in git but were deleted locally.\n")

            for file_path in differences['deleted']:
                print(f"\nFile: {file_path}")
                print("  Options:")
                print("    [D] Confirm DELETION (remove from git)")
                print("    [R] RESTORE from git (bring file back)")
                print("    [S] Skip (leave in inconsistent state)")

                while True:
                    choice = input("  Your choice (D/R/S): ").strip().upper()
                    if choice == 'D':
                        decisions['keep_deleted'].append(file_path)
                        print(f"    -> Will delete from git")
                        break
                    elif choice == 'R':
                        decisions['restore_deleted'].append(file_path)
                        print(f"    -> Will restore from git")
                        break
                    elif choice == 'S':
                        decisions['skip'].append(file_path)
                        print(f"    -> Skipping")
                        break
                    else:
                        print("    Invalid choice. Please enter D, R, or S.")

        return decisions

    def apply_sync_decisions(self, decisions):
        """Apply the user's sync decisions before deployment."""
        changes_made = False

        # Restore files from git (discard local changes)
        for file_path in decisions['restore_git']:
            print(f"  Restoring from git: {file_path}")
            self.run_command(f'git checkout HEAD -- "{file_path}"')
            changes_made = True

        # Restore deleted files from git
        for file_path in decisions['restore_deleted']:
            print(f"  Restoring deleted file: {file_path}")
            self.run_command(f'git checkout HEAD -- "{file_path}"')
            changes_made = True

        # Stage files to commit (local changes)
        for file_path in decisions['commit_local']:
            print(f"  Staging local changes: {file_path}")
            self.run_command(f'git add "{file_path}"')
            changes_made = True

        # Stage new files to add
        for file_path in decisions['add_new']:
            print(f"  Adding new file: {file_path}")
            self.run_command(f'git add "{file_path}"')
            changes_made = True

        # Stage deletions
        for file_path in decisions['keep_deleted']:
            print(f"  Staging deletion: {file_path}")
            self.run_command(f'git rm "{file_path}"')
            changes_made = True

        # Commit if we made changes
        if changes_made:
            staged_result = self.run_command('git diff --cached --name-only', suppress_errors=True)
            if staged_result.stdout.strip():
                print("\n  Committing sync changes...")
                self.run_command('git commit -m "Sync local changes before deployment"')
                return True

        return False

    def load_config(self, config_file):
        """Load configuration from JSON file"""
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def run_command(self, cmd, suppress_errors=False):
        """Run shell command and return output"""
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if result.returncode != 0 and not suppress_errors:
            print(f"Error running command: {cmd}")
            print(f"Error message: {result.stderr}")
        return result

    def check_repo_status(self):
        """Check if we're in a git repository"""
        result = self.run_command("git rev-parse --git-dir", suppress_errors=True)
        if result.returncode != 0:
            print("ERROR: Not in a git repository!")
            return False

        result = self.run_command("git remote -v")
        if self.repo_url not in result.stdout:
            print(f"WARNING: This doesn't appear to be the PHYS119 production repo!")
            print(f"Expected: {self.repo_url}")
            print(f"Found: {result.stdout}")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                return False

        return True

    def initialize_deployment(self, max_lab):
        """Initialize the production branch with labs 0 through max_lab from dev"""
        if not self.check_repo_status():
            return False

        print(f"\n{'='*60}")
        print(f"INITIALIZING DEPLOYMENT")
        print(f"{'='*60}")
        print(f"Production Branch: {self.production_branch}")
        print(f"Source Branch: {self.dev_branch}")
        print(f"Content: Lab00 through Lab{max_lab:02d}")
        print(f"{'='*60}\n")

        # Ensure we're on dev and up to date
        print(f"Step 1: Updating {self.dev_branch} branch...")
        self.run_command(f"git checkout {self.dev_branch}")
        self.run_command(f"git pull origin {self.dev_branch}")

        # Check if production branch exists remotely
        print(f"\nStep 2: Preparing {self.production_branch} branch...")
        remote_check = self.run_command(f"git ls-remote --heads origin {self.production_branch}", suppress_errors=True)

        if remote_check.stdout.strip():
            print(f"  Branch {self.production_branch} exists remotely. Deleting to start fresh...")
            # Delete local branch if it exists
            self.run_command(f"git branch -D {self.production_branch}", suppress_errors=True)
            # Delete remote branch
            self.run_command(f"git push origin --delete {self.production_branch}", suppress_errors=True)

        # Create fresh production branch
        print(f"\nStep 3: Creating {self.production_branch} branch...")
        self.run_command(f"git checkout -b {self.production_branch}")

        # Remove future labs
        print(f"\nStep 4: Removing labs after Lab{max_lab:02d}...")
        labs_to_remove = []
        for future_lab in range(max_lab + 1, self.total_labs):
            lab_dir = f"Lab{future_lab:02d}"
            if os.path.exists(lab_dir):
                labs_to_remove.append(lab_dir)
                print(f"  Removing {lab_dir}")

        if labs_to_remove:
            for lab_dir in labs_to_remove:
                self.run_command(f"git rm -rf {lab_dir}")

            # Commit changes
            commit_msg = f"Initialize deployment: Labs 0-{max_lab}"
            self.run_command(f'git commit -m "{commit_msg}"')
        else:
            print("  No labs to remove")

        # Push to remote
        print(f"\nStep 5: Pushing to remote...")
        self.run_command(f"git push -u origin {self.production_branch}")

        # Return to dev
        self.run_command(f"git checkout {self.dev_branch}")

        print(f"\n{'='*60}")
        print(f"SUCCESS! Deployment initialized with Labs 0-{max_lab}")
        print(f"{'='*60}\n")

        # Generate and display links
        self.show_deployed_links(max_lab)

        return True

    def update_deployment(self, new_max_lab):
        """Update the production branch to include up to new_max_lab from dev"""
        if not self.check_repo_status():
            return False

        print(f"\n{'='*60}")
        print(f"UPDATING DEPLOYMENT")
        print(f"{'='*60}")
        print(f"Production Branch: {self.production_branch}")
        print(f"Source Branch: {self.dev_branch}")
        print(f"Adding: Up to Lab{new_max_lab:02d}")
        print(f"{'='*60}\n")

        # Ensure we're on dev and up to date
        print(f"Step 1: Updating {self.dev_branch} branch...")
        self.run_command(f"git checkout {self.dev_branch}")
        self.run_command(f"git pull origin {self.dev_branch}")

        # Check for file differences before deployment
        print(f"\nStep 2: Checking for local changes in Labs 0-{new_max_lab}...")
        differences = self.get_lab_file_differences(new_max_lab)

        total_diffs = len(differences['modified']) + len(differences['untracked']) + len(differences['deleted'])
        if total_diffs > 0:
            decisions = self.prompt_sync_decisions(differences)

            # Check if user wants to proceed
            print(f"\n{'='*60}")
            print("SYNC SUMMARY")
            print(f"{'='*60}")
            print(f"  Files to commit (local -> git): {len(decisions['commit_local'])}")
            print(f"  Files to restore (git -> local): {len(decisions['restore_git'])}")
            print(f"  New files to add: {len(decisions['add_new'])}")
            print(f"  Deletions to confirm: {len(decisions['keep_deleted'])}")
            print(f"  Deleted files to restore: {len(decisions['restore_deleted'])}")
            print(f"  Files skipped: {len(decisions['skip'])}")

            proceed = input("\nProceed with these changes and continue deployment? (y/N): ").strip().lower()
            if proceed != 'y':
                print("\nDeployment cancelled.")
                return False

            # Apply sync decisions
            print(f"\nStep 3: Applying sync decisions...")
            self.apply_sync_decisions(decisions)
        else:
            print("  No differences found. Local files match git repository.")

        # Switch to production branch
        print(f"\nStep 4: Switching to {self.production_branch} branch...")
        result = self.run_command(f"git checkout {self.production_branch}")
        if result.returncode != 0:
            print(f"\nERROR: Branch {self.production_branch} doesn't exist!")
            print("Run 'initialize' command first.")
            return False

        # Sync with remote before making changes
        print(f"  Syncing {self.production_branch} with remote...")
        pull_result = self.run_command(f"git pull origin {self.production_branch}")
        if pull_result.returncode != 0:
            print(f"\nERROR: Failed to pull from remote. Resolve conflicts manually and try again.")
            self.run_command(f"git checkout {self.dev_branch}", suppress_errors=True)
            return False

        # Add labs from dev that we want to include (without rewriting history)
        print(f"\nStep 5: Adding labs 0-{new_max_lab} from {self.dev_branch}...")
        labs_to_add = []
        for lab_to_add in range(0, new_max_lab + 1):
            lab_dir = f"Lab{lab_to_add:02d}"
            if not os.path.exists(lab_dir):
                # This lab was previously removed, restore it from dev
                labs_to_add.append(lab_dir)
                print(f"  Restoring {lab_dir} from {self.dev_branch}")
                self.run_command(f"git checkout {self.dev_branch} -- {lab_dir}")

        if labs_to_add:
            # Stage the restored labs
            for lab_dir in labs_to_add:
                self.run_command(f"git add {lab_dir}")

        # Remove future labs
        print(f"\nStep 6: Removing labs after Lab{new_max_lab:02d}...")
        labs_to_remove = []
        for future_lab in range(new_max_lab + 1, self.total_labs):
            lab_dir = f"Lab{future_lab:02d}"
            if os.path.exists(lab_dir):
                labs_to_remove.append(lab_dir)
                print(f"  Removing {lab_dir}")

        if labs_to_remove:
            for lab_dir in labs_to_remove:
                self.run_command(f"git rm -rf {lab_dir}")

        # Commit if there are any changes
        if labs_to_add or labs_to_remove:
            commit_msg = f"Updated to include Labs 0-{new_max_lab}"
            self.run_command(f'git commit -m "{commit_msg}"')
        else:
            print("  No changes needed")

        # Push to remote (normal push, no force)
        print(f"\nStep 7: Pushing to remote...")
        self.run_command(f"git push origin {self.production_branch}")

        # Return to dev
        self.run_command(f"git checkout {self.dev_branch}")

        print(f"\n{'='*60}")
        print(f"SUCCESS! Now deployed: Labs 0-{new_max_lab}")
        print(f"{'='*60}\n")

        # Generate and display links
        self.show_deployed_links(new_max_lab)

        return True

    def generate_link(self, lab_num=None, file_path=None, prelab=False):
        """Generate nbgitpuller link for the current deployment"""
        base_link = (
            f"{self.hub_url}/hub/user-redirect/git-pull"
            f"?repo={self.repo_url}"
            f"&branch={self.production_branch}"
        )

        if file_path:
            # Specific file link (include repo name in path)
            link = f"{base_link}&urlpath=lab/tree/{self.repo_name}/{file_path}"
            print(f"\nLink to {file_path}:")
        elif lab_num is not None:
            # Link to specific lab or prelab (include repo name in path)
            lab_dir = f"Lab{lab_num:02d}"
            lab_name = self.lab_names.get(str(lab_num), f"Lab {lab_num}")

            if prelab:
                # Generate prelab link
                prelab_file = f"Prelab{lab_num:02d}.ipynb"
                link = f"{base_link}&urlpath=lab/tree/{self.repo_name}/{lab_dir}/{prelab_file}"
                print(f"\nLink to Prelab for {lab_name} (Lab{lab_num:02d}):")
            else:
                # Generate lab link
                link = f"{base_link}&urlpath=lab/tree/{self.repo_name}/{lab_dir}/Lab{lab_num:02d}.ipynb"
                print(f"\nLink to {lab_name} (Lab{lab_num:02d}):")
        else:
            # Link to repository root (include repo name in path)
            link = f"{base_link}&urlpath=lab/tree/{self.repo_name}"
            print(f"\nLink to all current labs:")

        print(link)
        print()
        return link

    def show_deployed_links(self, max_lab):
        """Display all nbgitpuller links for deployed labs"""
        print(f"\n{'='*60}")
        print(f"NBGITPULLER LINKS")
        print(f"{'='*60}\n")

        # Link to all labs
        base_link = (
            f"{self.hub_url}/hub/user-redirect/git-pull"
            f"?repo={self.repo_url}"
            f"&branch={self.production_branch}"
        )
        all_labs_link = f"{base_link}&urlpath=lab/tree/{self.repo_name}"
        print(f"All current labs:")
        print(f"{all_labs_link}\n")

        # Individual lab links
        for lab_num in range(max_lab + 1):
            lab_dir = f"Lab{lab_num:02d}"

            # Check if prelab files exist
            prelab_files = []
            if os.path.exists(lab_dir):
                # Check for standard prelab
                standard_prelab = f"Prelab{lab_num:02d}.ipynb"
                if os.path.exists(os.path.join(lab_dir, standard_prelab)):
                    prelab_files.append(standard_prelab)

                # Check for V2 prelab
                v2_prelab = f"Prelab{lab_num:02d}-V2.ipynb"
                if os.path.exists(os.path.join(lab_dir, v2_prelab)):
                    prelab_files.append(v2_prelab)

            # Lab link
            lab_link = f"{base_link}&urlpath=lab/tree/{self.repo_name}/{lab_dir}/Lab{lab_num:02d}.ipynb"
            print(f"Lab{lab_num:02d}:")
            print(lab_link)

            # Prelab links (if any)
            for prelab_file in prelab_files:
                prelab_link = f"{base_link}&urlpath=lab/tree/{self.repo_name}/{lab_dir}/{prelab_file}"
                print(prelab_link)

            print()

        print(f"{'='*60}\n")

    def show_status(self):
        """Show current deployment status"""
        if not self.check_repo_status():
            return

        print(f"\n{'='*60}")
        print(f"DEPLOYMENT STATUS")
        print(f"{'='*60}")

        # Check if production branch exists
        remote_check = self.run_command(f"git ls-remote --heads origin {self.production_branch}", suppress_errors=True)

        if not remote_check.stdout.strip():
            print(f"Status: NOT INITIALIZED")
            print(f"Branch '{self.production_branch}' does not exist")
            print(f"\nRun: python phys119_deploy.py initialize <max_lab>")
            print(f"{'='*60}\n")
        else:
            # Fetch and checkout to see what's deployed
            self.run_command("git fetch origin")
            self.run_command(f"git checkout {self.production_branch}", suppress_errors=True)

            # Count labs
            deployed_labs = []
            max_deployed_lab = -1
            for i in range(self.total_labs):
                lab_dir = f"Lab{i:02d}"
                if os.path.exists(lab_dir):
                    deployed_labs.append(f"  Lab{i:02d}")
                    max_deployed_lab = i

            print(f"Status: ACTIVE")
            print(f"Production Branch: {self.production_branch}")
            print(f"Deployed Labs: {len(deployed_labs)}")
            print("\n" + "\n".join(deployed_labs))

            print(f"{'='*60}\n")

            # Show nbgitpuller links for all deployed labs
            if max_deployed_lab >= 0:
                self.show_deployed_links(max_deployed_lab)

            self.run_command(f"git checkout {self.dev_branch}", suppress_errors=True)

def print_usage():
    """Print usage instructions"""
    print("""
PHYS119 Lab Deployment Tool
===========================

Usage: python phys119_deploy.py <command> [options]

Commands:
  initialize <max_lab>    Initialize deployment with Labs 0 through <max_lab>
                         Example: python phys119_deploy.py initialize 1
                         (Deploys Lab00 and Lab01)

  update <max_lab>        Update deployment to include Labs 0 through <max_lab>
                         Example: python phys119_deploy.py update 2
                         (Adds Lab02 to existing deployment)

  link [lab_num] [--prelab]  Generate nbgitpuller link
                         Example: python phys119_deploy.py link 1
                         Example: python phys119_deploy.py link 2 --prelab
                         (No lab_num = link to all labs)

  status                  Show current deployment status
                         Example: python phys119_deploy.py status

Examples:
  # First time setup - deploy Lab00 and Lab01
  python phys119_deploy.py initialize 1

  # Later, add Lab02
  python phys119_deploy.py update 2

  # Get link for Lab01
  python phys119_deploy.py link 1

  # Get link for Prelab02
  python phys119_deploy.py link 2 --prelab

  # Check what's deployed
  python phys119_deploy.py status
""")

def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1].lower()
    deployer = PHYS119Deployer()

    if command == 'initialize':
        if len(sys.argv) < 3:
            print("ERROR: Please specify max lab number")
            print("Example: python phys119_deploy.py initialize 1")
            sys.exit(1)
        max_lab = int(sys.argv[2])
        deployer.initialize_deployment(max_lab)

    elif command == 'update':
        if len(sys.argv) < 3:
            print("ERROR: Please specify max lab number")
            print("Example: python phys119_deploy.py update 2")
            sys.exit(1)
        max_lab = int(sys.argv[2])
        deployer.update_deployment(max_lab)

    elif command == 'link':
        if len(sys.argv) >= 3:
            lab_num = int(sys.argv[2])
            # Check if --prelab flag is present
            prelab = '--prelab' in sys.argv
            deployer.generate_link(lab_num=lab_num, prelab=prelab)
        else:
            deployer.generate_link()

    elif command == 'status':
        deployer.show_status()

    else:
        print(f"ERROR: Unknown command '{command}'")
        print_usage()
        sys.exit(1)

if __name__ == "__main__":
    main()
