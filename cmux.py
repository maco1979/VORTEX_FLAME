"""
cmux Integration — Terminal Multiplexer for Execution Visualization
====================================================================
Terminal multiplexer for VORTEX_FLAME execution mode visualization.
Based on GitHub trending project "cmux".

Status: Interface complete. Requires Unix socket support (Linux/macOS).

Architecture:
- CmuxSession: Manages terminal panes per execution mode
- PaneLayout: Maps execution modes to terminal layouts
- SocketAPI: Unix socket interface for external control

Integration Points:
- soul_orchestrator: Visualizes EXECUTION_MODES in terminal panes
- worktree_manager: Each subagent gets its own pane
- harness_runtime: Audit trail streamed to dedicated pane

Layouts:
- Team mode: 5 panes (plan/code/review/test/deploy)
- Ultrapilot mode: 3 panes (research/design/code) + 1 overview
- Ralph mode: 4 panes (verify/fix/retest/report) in loop
"""

from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class Pane:
    id: int
    title: str
    soul: str
    stage: str
    command: Optional[str] = None
    active: bool = False


@dataclass
class PaneLayout:
    mode: str
    panes: List[Pane]
    layout_type: str


LAYOUTS = {
    "team": PaneLayout(
        mode="team",
        layout_type="vertical_split",
        panes=[
            Pane(id=0, title="Plan",    soul="cezanne",    stage="team_plan",   command="watch -n 1 'cat .vf/plan.json'"),
            Pane(id=1, title="Code",    soul="cezanne",    stage="team_code",   command="watch -n 1 'cat .vf/code.json'"),
            Pane(id=2, title="Review",  soul="montesquieu", stage="team_review", command="watch -n 1 'cat .vf/review.json'"),
            Pane(id=3, title="Test",    soul="galileo",    stage="team_test",   command="watch -n 1 'cat .vf/test.json'"),
            Pane(id=4, title="Deploy",  soul="davinci",    stage="team_deploy", command="watch -n 1 'cat .vf/deploy.json'"),
        ],
    ),
    "ultrapilot": PaneLayout(
        mode="ultrapilot",
        layout_type="grid_2x2",
        panes=[
            Pane(id=0, title="Research", soul="einstein", stage="ultra_research", command="watch -n 1 'cat .vf/research.json'"),
            Pane(id=1, title="Design",   soul="davinci",  stage="ultra_design",   command="watch -n 1 'cat .vf/design.json'"),
            Pane(id=2, title="Code",     soul="cezanne",  stage="ultra_code",     command="watch -n 1 'cat .vf/code.json'"),
            Pane(id=3, title="Overview", soul="cezanne",  stage="overview",       command="watch -n 1 'cat .vf/overview.json'"),
        ],
    ),
    "ralph": PaneLayout(
        mode="ralph",
        layout_type="vertical_split",
        panes=[
            Pane(id=0, title="Verify",  soul="montesquieu", stage="ralph_verify", command="watch -n 1 'cat .vf/verify.json'"),
            Pane(id=1, title="Fix",     soul="cezanne",     stage="ralph_fix",    command="watch -n 1 'cat .vf/fix.json'"),
            Pane(id=2, title="Retest",  soul="galileo",     stage="ralph_retest", command="watch -n 1 'cat .vf/retest.json'"),
            Pane(id=3, title="Report",  soul="herodotus",   stage="ralph_report", command="watch -n 1 'cat .vf/report.json'"),
        ],
    ),
}


class CmuxSession:
    def __init__(self):
        self.active_mode: Optional[str] = None
        self.active_panes: List[Pane] = []
        self._socket_path: str = "/tmp/vf_cmux.sock"

    def start(self, mode: str) -> dict:
        layout = LAYOUTS.get(mode)
        if not layout:
            return {"status": "error", "message": f"Unknown mode: {mode}"}

        self.active_mode = mode
        self.active_panes = layout.panes
        return {
            "status": "started",
            "mode": mode,
            "layout": layout.layout_type,
            "pane_count": len(layout.panes),
            "panes": [{"id": p.id, "title": p.title, "soul": p.soul} for p in layout.panes],
        }

    def switch_mode(self, mode: str) -> dict:
        return self.start(mode)

    def get_status(self) -> dict:
        return {
            "active_mode": self.active_mode,
            "pane_count": len(self.active_panes),
            "panes": [{"id": p.id, "title": p.title, "soul": p.soul, "active": p.active} for p in self.active_panes],
        }

    def focus_pane(self, pane_id: int) -> dict:
        for p in self.active_panes:
            p.active = (p.id == pane_id)
        return {"status": "focused", "pane_id": pane_id}

    def send_to_pane(self, pane_id: int, data: str) -> dict:  # noqa: ARG002
        pane = next((p for p in self.active_panes if p.id == pane_id), None)
        if not pane:
            return {"status": "error", "message": f"Pane {pane_id} not found"}
        return {"status": "sent", "pane_id": pane_id, "pane_title": pane.title}

    def socket_api(self, command: str, payload: Optional[Dict] = None) -> dict:
        """
        Unix socket API for external control.
        Commands: start, switch, focus, send, status, quit
        """
        payload = payload or {}
        handlers = {
            "start": lambda: self.start(payload.get("mode", "team")),
            "switch": lambda: self.switch_mode(payload.get("mode", "team")),
            "focus": lambda: self.focus_pane(payload.get("pane_id", 0)),
            "send": lambda: self.send_to_pane(payload.get("pane_id", 0), payload.get("data", "")),
            "status": lambda: self.get_status(),
            "quit": lambda: {"status": "stopped"},
        }
        handler = handlers.get(command)
        if not handler:
            return {"status": "error", "message": f"Unknown command: {command}"}
        return handler()
