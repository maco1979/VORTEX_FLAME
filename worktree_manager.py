"""
WorktreeManager — Concept Interface
=====================================
Git worktree lifecycle management for Subagent isolation.
Core implementation is proprietary.

Workflow:
1. create(stage_name, branch_suffix) -> {status, worktree_path, branch}
2. Subagent works in isolated worktree
3. merge(branch, target_branch) -> {status}
4. cleanup(worktree_path, branch) -> {status}
5. cleanup_all() -> {status, count}

Branch naming: vf/{stage_name}/{sub_id}/{hash8}
Worktree base: .worktrees/
Merge strategy: --no-ff (conflicts abort and report, never auto-resolve)
"""

import os
import subprocess
from pathlib import Path


class WorktreeManager:
    def __init__(self, project_path: str = None):
        self.project_path = Path(project_path or os.getcwd())
        self.worktree_base = self.project_path / ".worktrees"
        self._active: dict = {}

    def _is_git_repo(self) -> bool:
        git_dir = self.project_path / ".git"
        return git_dir.exists()

    def _run_git(self, *args, cwd=None) -> subprocess.CompletedProcess:
        work_dir = str(cwd or self.project_path)
        return subprocess.run(
            ["git"] + list(args),
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def create(self, stage_name: str, branch_suffix: str = "") -> dict:
        if not self._is_git_repo():
            return {"status": "error", "message": "Not a git repository"}

        self.worktree_base.mkdir(parents=True, exist_ok=True)

        import hashlib
        short_hash = hashlib.md5(f"{stage_name}{branch_suffix}".encode()).hexdigest()[:8]
        branch = f"vf/{stage_name}/{branch_suffix or 'default'}/{short_hash}"
        worktree_path = self.worktree_base / f"{stage_name}_{short_hash}"

        result = self._run_git("worktree", "add", str(worktree_path), "-b", branch)
        if result.returncode != 0:
            result = self._run_git("worktree", "add", str(worktree_path), "-b", branch, "HEAD")
            if result.returncode != 0:
                return {"status": "error", "message": result.stderr.strip(), "branch": branch}

        self._active[branch] = {
            "path": str(worktree_path),
            "stage": stage_name,
            "branch": branch,
        }

        return {
            "status": "created",
            "worktree_path": str(worktree_path),
            "branch": branch,
        }

    def merge(self, branch: str, target_branch: str = "main", delete_after: bool = True) -> dict:
        if not self._is_git_repo():
            return {"status": "error", "message": "Not a git repository"}

        result = self._run_git("merge", "--no-ff", branch, "-m", f"Merge {branch} into {target_branch}")
        if result.returncode != 0:
            self._run_git("merge", "--abort")
            return {"status": "conflict", "message": "Merge conflict, aborted", "branch": branch}

        if delete_after:
            self._run_git("branch", "-d", branch)

        self._active.pop(branch, None)

        return {"status": "merged", "branch": branch, "target": target_branch}

    def cleanup(self, worktree_path: str, branch: str = None) -> dict:
        result = self._run_git("worktree", "remove", worktree_path, "--force")
        if branch:
            self._run_git("branch", "-D", branch)
            self._active.pop(branch, None)

        return {
            "status": "cleaned",
            "worktree_path": worktree_path,
            "branch": branch,
        }

    def cleanup_all(self) -> dict:
        count = 0
        for branch, info in list(self._active.items()):
            self.cleanup(info["path"], branch)
            count += 1

        self._run_git("worktree", "prune")
        self._active.clear()

        return {"status": "all_cleaned", "count": count}

    def list_active(self) -> list:
        result = self._run_git("worktree", "list", "--porcelain")
        worktrees = []
        if result.returncode == 0:
            current = {}
            for line in result.stdout.strip().split("\n"):
                if line.startswith("worktree "):
                    current = {"path": line.split(" ", 1)[1]}
                elif line.startswith("branch "):
                    current["branch"] = line.split(" ", 1)[1]
                    if current.get("path") and "worktrees" in current["path"]:
                        worktrees.append(current.copy())
                    current = {}
        return worktrees
