"""
Harness Runtime — Model Governance Interface
==============================================
Model governance runtime with 7-layer security.

Status: Interface complete. Core implementation pending.

Governance Layers:
- Training safety: T001-T007 red lines
- Code quality: Hooks system (4 stages)
- Secret governance: Hardcoded key/PII/injection detection
- Supply chain: npm backdoor/ghost dependency detection
- Audit trail: TrainingTraceLogger (JSONL)
- Vulnerability scan: OWASP Top 10 / dependency / key / compliance
- Runtime guard: Guardian integration

Agent Security 4 Layers:
- ActionGuard: Action whitelist per soul
- PromptInjectionDetector: 16 injection patterns
- NetworkWhitelist: Domain whitelist/blacklist
- AgentAuditTrail: Full operation logging

Integration Whitelists (P0):
- NetworkWhitelist additions:
  - localhost:9432 (CodeGraph MCP server)
  - localhost:9433 (Understand-Anything MCP server)
- ActionGuard additions:
  - ci_callers, ci_callees, ci_affected, ci_search (CodeGraph tools)
  - ci_domain, ci_impact, ci_context, ci_ask (UA tools)
  - kwp_* actions (knowledge-work-plugins)
"""

NETWORK_WHITELIST = [
    "localhost:9432",
    "localhost:9433",
    "127.0.0.1:9432",
    "127.0.0.1:9433",
]

ACTION_WHITELIST_PER_SOUL = {
    "cezanne": [
        "ci_callers", "ci_callees", "ci_affected", "ci_search",
        "ci_context", "ci_domain", "ci_impact", "ci_ask",
        "kwp_engineering", "kwp_operations",
    ],
    "einstein": [
        "ci_search", "ci_context", "ci_impact",
        "kwp_finance", "kwp_research", "kwp_data",
    ],
    "davinci": [
        "ci_context", "ci_impact", "ci_domain",
        "kwp_engineering", "kwp_design", "kwp_product",
    ],
    "strategy": [
        "ci_impact", "ci_domain",
        "kwp_sales", "kwp_finance", "kwp_product",
    ],
    "montesquieu": [
        "ci_impact", "ci_domain",
        "kwp_legal", "kwp_sales", "kwp_hr", "kwp_marketing",
    ],
    "guizhu": [
        "kwp_hr", "kwp_support", "kwp_healthcare", "kwp_education",
    ],
    "herodotus": [
        "ci_search", "ci_ask",
        "kwp_writing", "kwp_support", "kwp_education",
    ],
    "humboldt": [
        "ci_search", "ci_context", "ci_domain",
        "kwp_data", "kwp_operations",
    ],
    "galileo": ["ci_search", "ci_context", "kwp_research"],
    "darwin": ["ci_search", "ci_context", "kwp_research", "kwp_healthcare"],
    "monet": ["ci_context", "kwp_design", "kwp_marketing", "kwp_writing"],
    "vangogh": ["ci_context", "kwp_design"],
    "yuanlongping": [],
    "beethoven": [],
}


class HarnessRuntime:
    def action_guard_check(self, soul: str, action: str, **kwargs) -> dict:
        allowed = ACTION_WHITELIST_PER_SOUL.get(soul, [])
        if action in allowed:
            return {"allowed": True, "soul": soul, "action": action}
        return {"allowed": False, "soul": soul, "action": action, "reason": "Not in soul action whitelist"}

    def injection_detect(self, prompt: str) -> dict:
        raise NotImplementedError("Core harness is proprietary")

    def network_check(self, url: str) -> dict:
        for allowed in NETWORK_WHITELIST:
            if url.startswith(allowed) or f"//{allowed}" in url:
                return {"allowed": True, "url": url}
        return {"allowed": False, "url": url, "reason": "Not in network whitelist"}

    def agent_security_check(self, soul: str, action: str, **kwargs) -> dict:
        raise NotImplementedError("Core harness is proprietary")

    def register_action(self, soul: str, action: str) -> dict:
        raise NotImplementedError("Core harness is proprietary")

    def register_network_endpoint(self, endpoint: str) -> dict:
        raise NotImplementedError("Core harness is proprietary")
