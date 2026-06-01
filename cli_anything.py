"""
CLI-Anything Integration — CLI Tool Wrapper
=============================================
CLI tool wrapper that exposes any CLI as an MCP tool for VORTEX_FLAME souls.
Based on GitHub trending project "cli-anything".

Status: Interface complete. Runtime requires harness_runtime integration.

Architecture:
- CLIAdapter: Wraps CLI commands as MCP-callable tools
- ToolRegistry: Maps CLI commands to soul tool whitelists
- SafetyGuard: Validates commands against guardian/harness whitelists

Integration Points:
- soul_orchestrator: CLI tools registered as soul capabilities
- harness_runtime: CLI commands go through action_guard_check
- guardian: CLI process execution monitored via process whitelist
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class CLITool:
    name: str
    command: str
    description: str
    souls: List[str] = field(default_factory=list)
    requires_network: bool = False
    requires_filesystem: bool = False
    dangerous: bool = False


CLI_TOOL_CATALOG = {
    "git_status": CLITool(
        name="git_status",
        command="git status --porcelain",
        description="Get git working tree status",
        souls=["cezanne", "davinci", "galileo"],
        requires_filesystem=True,
    ),
    "git_log": CLITool(
        name="git_log",
        command="git log --oneline -20",
        description="Get recent commit history",
        souls=["cezanne", "herodotus", "galileo"],
        requires_filesystem=True,
    ),
    "git_diff": CLITool(
        name="git_diff",
        command="git diff --stat",
        description="Get diff statistics",
        souls=["cezanne", "montesquieu"],
        requires_filesystem=True,
    ),
    "npm_test": CLITool(
        name="npm_test",
        command="npm test",
        description="Run npm test suite",
        souls=["cezanne", "galileo"],
        requires_filesystem=True,
        requires_network=True,
    ),
    "pip_install": CLITool(
        name="pip_install",
        command="pip install {package}",
        description="Install Python package",
        souls=["cezanne"],
        requires_network=True,
        dangerous=True,
    ),
    "curl_api": CLITool(
        name="curl_api",
        command="curl -s {url}",
        description="Make HTTP request",
        souls=["einstein", "humboldt"],
        requires_network=True,
    ),
    "docker_ps": CLITool(
        name="docker_ps",
        command="docker ps",
        description="List running containers",
        souls=["cezanne", "davinci"],
    ),
    "python_run": CLITool(
        name="python_run",
        command="python {script}",
        description="Execute Python script",
        souls=["cezanne", "einstein", "darwin", "galileo"],
        requires_filesystem=True,
    ),
    "mano_p_status": CLITool(
        name="mano_p_status",
        command="python -c \"from mano_p_adapter import get_adapter; print(get_adapter().status())\"",
        description="Check Mano-P adapter status (mode, hardware, server)",
        souls=["cezanne", "davinci"],
    ),
    "mano_p_screenshot": CLITool(
        name="mano_p_screenshot",
        command="python -c \"from mano_p_adapter import get_adapter; print(get_adapter().take_screenshot())\"",
        description="Capture screenshot via Mano-P GUI agent",
        souls=["cezanne", "davinci"],
        requires_filesystem=True,
    ),
    "mano_p_run": CLITool(
        name="mano_p_run",
        command="python -c \"from mano_p_adapter import get_adapter; print(get_adapter().execute_task('cezanne', '{task}'))\"",
        description="Execute a GUI task via Mano-P (natural language)",
        souls=["cezanne", "davinci"],
        requires_filesystem=True,
    ),
}


class CLIAdapter:
    def __init__(self, catalog: Optional[Dict] = None):
        self.catalog = catalog or CLI_TOOL_CATALOG

    def get_tools_for_soul(self, soul_name: str) -> List[CLITool]:
        return [t for t in self.catalog.values() if soul_name in t.souls]

    def get_mcp_tools(self) -> List[dict]:
        return [
            {
                "name": f"cli_{tool.name}",
                "description": tool.description,
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "args": {"type": "string", "description": "Command arguments"},
                    },
                },
            }
            for tool in self.catalog.values()
            if not tool.dangerous
        ]

    def validate_command(self, tool_name: str, soul_name: str) -> dict:
        tool = self.catalog.get(tool_name)
        if not tool:
            return {"allowed": False, "reason": "Tool not found"}

        if soul_name not in tool.souls:
            return {"allowed": False, "reason": f"Soul {soul_name} not authorized for {tool_name}"}

        if tool.dangerous:
            return {"allowed": False, "reason": "Dangerous command requires explicit approval"}

        return {"allowed": True, "tool": tool_name, "soul": soul_name}

    def register_tool(self, tool: CLITool) -> dict:
        self.catalog[tool.name] = tool
        return {"status": "registered", "tool": tool.name}
