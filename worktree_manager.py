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


class WorktreeManager:
    def create(self, stage_name: str, branch_suffix: str = "") -> dict:
        raise NotImplementedError("Core worktree manager is proprietary")

    def merge(self, branch: str, target_branch: str = "main", delete_after: bool = True) -> dict:
        raise NotImplementedError("Core worktree manager is proprietary")

    def cleanup(self, worktree_path: str, branch: str = None) -> dict:
        raise NotImplementedError("Core worktree manager is proprietary")

    def cleanup_all(self) -> dict:
        raise NotImplementedError("Core worktree manager is proprietary")
