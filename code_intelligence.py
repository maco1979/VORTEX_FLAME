"""
Code Intelligence — Unified Code Understanding Interface
=========================================================
Integrates CodeGraph (syntax-level) and Understand-Anything (semantic-level)
into a single interface for soul memory and JEPA world model.

Status: Interface complete. MCP server deployment pending (P1 todo).

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                   CodeIntelligenceManager                    │
│                                                              │
│  ┌─────────────────┐    ┌──────────────────────────────┐   │
│  │  SyntaxLayer     │    │  SemanticLayer                │   │
│  │  (CodeGraph)     │    │  (Understand-Anything)        │   │
│  │                  │    │                                │   │
│  │  - call_graph    │    │  - domain_graph               │   │
│  │  - dependencies  │    │  - business_flow              │   │
│  │  - symbol_refs   │    │  - impact_analysis            │   │
│  │  - affected      │    │  - concept_map                │   │
│  └────────┬─────────┘    └──────────────┬───────────────┘   │
│           │                              │                    │
│           └──────────┬───────────────────┘                   │
│                      ▼                                       │
│           ┌─────────────────────┐                            │
│           │  UnifiedQuery       │                            │
│           │  - search(query)    │                            │
│           │  - context(symbol)  │                            │
│           │  - impact(change)   │                            │
│           │  - to_embedding()   │                            │
│           └─────────────────────┘                            │
└─────────────────────────────────────────────────────────────┘

Integration Points:
- soul_memory: code_memory + domain_memory categories
- JEPA: enhanced hidden_states for PHYS-JEPA / DESIGN-JEPA
- soul_orchestrator: tool registration for Cezanne/Einstein souls
- guardian: file_monitor whitelist for .codegraph/ directory
- harness_runtime: network_whitelist for MCP server ports

Source Projects:
- CodeGraph (github.com/anthropics/codegraph)
  - tree-sitter + SQLite FTS5 pre-built index
  - 8 MCP tools: codegraph_context, codegraph_callers, etc.
  - 35% token savings, 70% fewer tool calls (benchmark)

- Understand-Anything (github.com/Lum1104/Understand-Anything)
  - 5-agent pipeline: Parser → Analyzer → GraphBuilder → QA → Visualizer
  - React Flow interactive knowledge graph
  - Business domain view + dependency analysis
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class CodeIntelligenceLayer(Enum):
    SYNTAX = "syntax"
    SEMANTIC = "semantic"


@dataclass
class SymbolInfo:
    name: str
    kind: str
    file_path: str
    line_start: int
    line_end: int
    callers: List[str] = field(default_factory=list)
    callees: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)


@dataclass
class DomainConcept:
    name: str
    description: str
    related_concepts: List[str] = field(default_factory=list)
    affected_symbols: List[str] = field(default_factory=list)
    business_flow: Optional[str] = None


@dataclass
class ImpactResult:
    changed_symbol: str
    direct_impact: List[str] = field(default_factory=list)
    indirect_impact: List[str] = field(default_factory=list)
    affected_domains: List[str] = field(default_factory=list)
    risk_level: str = "low"


@dataclass
class CodeContext:
    symbols: List[SymbolInfo] = field(default_factory=list)
    domains: List[DomainConcept] = field(default_factory=list)
    impact: Optional[ImpactResult] = None
    raw_context: str = ""


class SyntaxProvider(ABC):
    """
    Syntax-level code intelligence provider.
    Default implementation: CodeGraph (tree-sitter + SQLite FTS5).
    """

    @abstractmethod
    def build_index(self, project_path: str) -> dict:
        pass

    @abstractmethod
    def get_callers(self, symbol: str) -> List[SymbolInfo]:
        pass

    @abstractmethod
    def get_callees(self, symbol: str) -> List[SymbolInfo]:
        pass

    @abstractmethod
    def get_references(self, symbol: str) -> List[SymbolInfo]:
        pass

    @abstractmethod
    def get_affected(self, file_path: str) -> List[SymbolInfo]:
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[SymbolInfo]:
        pass


class SemanticProvider(ABC):
    """
    Semantic-level code intelligence provider.
    Default implementation: Understand-Anything (5-agent pipeline).
    """

    @abstractmethod
    def analyze_domain(self, project_path: str) -> List[DomainConcept]:
        pass

    @abstractmethod
    def get_business_flow(self, domain: str) -> Optional[str]:
        pass

    @abstractmethod
    def analyze_impact(self, change_description: str) -> ImpactResult:
        pass

    @abstractmethod
    def get_concept_map(self, symbol: str) -> List[DomainConcept]:
        pass

    @abstractmethod
    def ask(self, question: str) -> str:
        pass


class CodeGraphProvider(SyntaxProvider):
    """
    CodeGraph integration via MCP protocol.
    Communicates with codegraph MCP server over stdio (JSON-RPC).

    MCP Tools exposed:
    - codegraph_context: Get context for a symbol
    - codegraph_callers: Get callers of a function
    - codegraph_callees: Get callees of a function
    - codegraph_references: Get all references to a symbol
    - codegraph_affected: Get symbols affected by a file change
    - codegraph_search: Full-text search across codebase
    - codegraph_outline: Get file outline/structure
    - codegraph_recent: Get recently changed symbols
    """

    def __init__(self, project_path: str, mcp_port: Optional[int] = None):
        self.project_path = project_path
        self.mcp_port = mcp_port
        self._index_built = False

    def build_index(self, project_path: str) -> dict:
        raise NotImplementedError("CodeGraph MCP integration is proprietary")

    def get_callers(self, symbol: str) -> List[SymbolInfo]:
        raise NotImplementedError("CodeGraph MCP integration is proprietary")

    def get_callees(self, symbol: str) -> List[SymbolInfo]:
        raise NotImplementedError("CodeGraph MCP integration is proprietary")

    def get_references(self, symbol: str) -> List[SymbolInfo]:
        raise NotImplementedError("CodeGraph MCP integration is proprietary")

    def get_affected(self, file_path: str) -> List[SymbolInfo]:
        raise NotImplementedError("CodeGraph MCP integration is proprietary")

    def search(self, query: str, top_k: int = 10) -> List[SymbolInfo]:
        raise NotImplementedError("CodeGraph MCP integration is proprietary")


class UnderstandAnythingProvider(SemanticProvider):
    """
    Understand-Anything integration via MCP protocol.
    5-agent pipeline: Parser → Analyzer → GraphBuilder → QA → Visualizer.

    Output formats:
    - knowledge-graph.json: Domain concepts and relationships
    - dependency-graph.json: Module dependencies
    - React Flow visualization: Interactive exploration
    """

    def __init__(self, project_path: str, mcp_port: Optional[int] = None):
        self.project_path = project_path
        self.mcp_port = mcp_port
        self._domain_cache: Dict[str, List[DomainConcept]] = {}

    def analyze_domain(self, project_path: str) -> List[DomainConcept]:
        raise NotImplementedError("Understand-Anything integration is proprietary")

    def get_business_flow(self, domain: str) -> Optional[str]:
        raise NotImplementedError("Understand-Anything integration is proprietary")

    def analyze_impact(self, change_description: str) -> ImpactResult:
        raise NotImplementedError("Understand-Anything integration is proprietary")

    def get_concept_map(self, symbol: str) -> List[DomainConcept]:
        raise NotImplementedError("Understand-Anything integration is proprietary")

    def ask(self, question: str) -> str:
        raise NotImplementedError("Understand-Anything integration is proprietary")


class CodeIntelligenceManager:
    """
    Unified code intelligence interface combining syntax and semantic layers.

    Usage:
        ci = CodeIntelligenceManager(project_path="/path/to/project")
        ci.initialize()

        # Syntax-level query (CodeGraph)
        callers = ci.get_callers("UserService.authenticate")
        affected = ci.get_affected("src/auth/service.py")

        # Semantic-level query (Understand-Anything)
        domains = ci.analyze_domain()
        impact = ci.analyze_impact("Remove the legacy auth module")

        # Unified query
        context = ci.get_full_context("UserService.authenticate")
        # context.symbols  → syntax info
        # context.domains  → semantic info
        # context.impact   → change impact

        # For JEPA: convert to embedding
        embedding = ci.to_embedding(context)

    Integration with soul_memory:
        ci → code_memory (syntax) + domain_memory (semantic)

    Integration with JEPA:
        ci.to_embedding(context) → augmented hidden_states for
        PHYS-JEPA (code dynamics) and DESIGN-JEPA (design logic)
    """

    def __init__(
        self,
        project_path: str,
        syntax_provider: Optional[SyntaxProvider] = None,
        semantic_provider: Optional[SemanticProvider] = None,
    ):
        self.project_path = project_path
        self.syntax = syntax_provider or CodeGraphProvider(project_path)
        self.semantic = semantic_provider or UnderstandAnythingProvider(project_path)
        self._initialized = False

    def initialize(self) -> dict:
        raise NotImplementedError("Core initialization is proprietary")

    def get_callers(self, symbol: str) -> List[SymbolInfo]:
        return self.syntax.get_callers(symbol)

    def get_callees(self, symbol: str) -> List[SymbolInfo]:
        return self.syntax.get_callees(symbol)

    def get_references(self, symbol: str) -> List[SymbolInfo]:
        return self.syntax.get_references(symbol)

    def get_affected(self, file_path: str) -> List[SymbolInfo]:
        return self.syntax.get_affected(file_path)

    def search(self, query: str, top_k: int = 10) -> List[SymbolInfo]:
        return self.syntax.search(query, top_k)

    def analyze_domain(self) -> List[DomainConcept]:
        return self.semantic.analyze_domain(self.project_path)

    def get_business_flow(self, domain: str) -> Optional[str]:
        return self.semantic.get_business_flow(domain)

    def analyze_impact(self, change_description: str) -> ImpactResult:
        return self.semantic.analyze_impact(change_description)

    def get_concept_map(self, symbol: str) -> List[DomainConcept]:
        return self.semantic.get_concept_map(symbol)

    def ask(self, question: str) -> str:
        return self.semantic.ask(question)

    def get_full_context(self, symbol: str) -> CodeContext:
        """
        Unified query: combine syntax and semantic information for a symbol.
        This is the primary interface for souls and JEPA.
        """
        symbols = self.syntax.get_references(symbol)
        domains = self.semantic.get_concept_map(symbol)
        return CodeContext(
            symbols=symbols,
            domains=domains,
            raw_context=self._format_context(symbols, domains),
        )

    def to_embedding(self, context: CodeContext) -> List[float]:
        """
        Convert code context to embedding vector for JEPA input.
        Syntax features + Semantic features → unified vector.
        """
        import hashlib
        dim = 768
        result = [0.0] * dim
        for i, s in enumerate(context.symbols[:20]):
            h = int(hashlib.md5(f"{s.name}:{s.kind}".encode()).hexdigest(), 16)
            idx = (h + i) % dim
            result[idx] += 1.0
        for i, d in enumerate(context.domains[:20]):
            h = int(hashlib.md5(f"{d.name}:{d.description}".encode()).hexdigest(), 16)
            idx = (h + i + 384) % dim
            result[idx] += 0.5
        max_val = max(abs(v) for v in result) or 1.0
        result = [v / max_val for v in result]
        return result

    def _format_context(
        self, symbols: List[SymbolInfo], domains: List[DomainConcept]
    ) -> str:
        parts = []
        for s in symbols[:20]:
            parts.append(
                f"{s.kind} {s.name} ({s.file_path}:{s.line_start}-{s.line_end})"
            )
        for d in domains[:10]:
            parts.append(f"[{d.name}] {d.description}")
        return "\n".join(parts)

    def get_mcp_tools(self) -> List[dict]:
        """
        Return MCP tool definitions for soul registration.
        Each tool maps to a CodeIntelligenceManager method.
        """
        return [
            {
                "name": "ci_callers",
                "description": "Get callers of a symbol (syntax layer)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Symbol name"}
                    },
                    "required": ["symbol"],
                },
            },
            {
                "name": "ci_callees",
                "description": "Get callees of a symbol (syntax layer)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Symbol name"}
                    },
                    "required": ["symbol"],
                },
            },
            {
                "name": "ci_affected",
                "description": "Get symbols affected by a file change (syntax layer)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to changed file",
                        }
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": "ci_search",
                "description": "Full-text search across codebase (syntax layer)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {
                            "type": "integer",
                            "description": "Max results",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "ci_domain",
                "description": "Analyze business domain of project (semantic layer)",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "ci_impact",
                "description": "Analyze impact of a proposed change (semantic layer)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "change": {
                            "type": "string",
                            "description": "Description of proposed change",
                        }
                    },
                    "required": ["change"],
                },
            },
            {
                "name": "ci_context",
                "description": "Get full context (syntax + semantic) for a symbol",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Symbol name"}
                    },
                    "required": ["symbol"],
                },
            },
            {
                "name": "ci_ask",
                "description": "Ask a question about the codebase (semantic layer)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Question about the codebase",
                        }
                    },
                    "required": ["question"],
                },
            },
        ]
