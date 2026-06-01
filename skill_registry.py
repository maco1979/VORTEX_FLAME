"""
Skill Registry — Unified Skill Registration Interface
======================================================
Integrates Anthropic's knowledge-work-plugins with VORTEX_FLAME's
skill_evolver and soul_config systems.

Architecture:
┌──────────────────────────────────────────────────────────────┐
│                     SkillRegistry                             │
│                                                               │
│  ┌──────────────────────┐   ┌────────────────────────────┐  │
│  │  External Skills      │   │  Evolved Skills             │  │
│  │  (knowledge-work-     │   │  (skill_evolver output)     │  │
│  │   plugins format)     │   │                              │  │
│  │                       │   │                              │  │
│  │  - Markdown + JSON    │   │  - core_rules (stable)      │  │
│  │  - 15 job plugins     │   │  - errata (dynamic)         │  │
│  │  - Anthropic official │   │  - 5-dimension audited      │  │
│  └──────────┬────────────┘   └──────────────┬─────────────┘  │
│             │                                │                 │
│             └──────────┬─────────────────────┘                 │
│                        ▼                                       │
│             ┌──────────────────────┐                           │
│             │  Unified Skill       │                           │
│             │  - name              │                           │
│             │  - soul_mapping      │                           │
│             │  - source            │                           │
│             │  - commands          │                           │
│             │  - connectors        │                           │
│             └──────────────────────┘                           │
└──────────────────────────────────────────────────────────────┘

Integration Flow:
  1. knowledge-work-plugins → SkillRegistry.register_external()
  2. SkillRegistry maps each plugin to appropriate soul(s)
  3. soul_config/*.yaml gains `skills` field referencing registered skills
  4. skill_evolver treats external skills as Stage 0 seeds
  5. Evolved skills replace external skills after audit passes

Source Project:
  anthropics/knowledge-work-plugins (GitHub, 9,700+ stars)
  - 15 job plugins: Sales, Finance, Legal, Engineering, Data, Design,
    Marketing, HR, Product, Research, Support, Operations, Writing,
    Education, Healthcare
  - Format: skills/{job}/commands/*.md + skills/{job}/connectors/*.json
  - Architecture: Pure Markdown + JSON, no runtime dependencies
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SkillSource(Enum):
    EXTERNAL = "external"
    EVOLVED = "evolved"
    CUSTOM = "custom"


class SkillStatus(Enum):
    SEED = "seed"
    AUDITING = "auditing"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


@dataclass
class SkillCommand:
    name: str
    description: str
    template: str
    parameters: List[str] = field(default_factory=list)


@dataclass
class SkillConnector:
    name: str
    connector_type: str
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegisteredSkill:
    skill_id: str
    name: str
    description: str
    source: SkillSource
    status: SkillStatus
    soul_mapping: List[str] = field(default_factory=list)
    commands: List[SkillCommand] = field(default_factory=list)
    connectors: List[SkillConnector] = field(default_factory=list)
    core_rules: str = ""
    errata: List[str] = field(default_factory=list)
    audit_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


KNOWLEDGE_WORK_PLUGIN_MAPPING = {
    "sales": {
        "souls": ["montesquieu", "strategy"],
        "domain": "Sales strategy, CRM, pipeline management",
    },
    "finance": {
        "souls": ["einstein", "strategy"],
        "domain": "Financial analysis, forecasting, risk assessment",
    },
    "legal": {
        "souls": ["montesquieu"],
        "domain": "Legal compliance, contract review, regulatory analysis",
    },
    "engineering": {
        "souls": ["cezanne", "davinci"],
        "domain": "Software engineering, architecture, code review",
    },
    "data": {
        "souls": ["humboldt", "einstein"],
        "domain": "Data analysis, visualization, statistical modeling",
    },
    "design": {
        "souls": ["davinci", "monet"],
        "domain": "UI/UX design, prototyping, design systems",
    },
    "marketing": {
        "souls": ["monet", "montesquieu"],
        "domain": "Marketing strategy, content creation, brand management",
    },
    "hr": {
        "souls": ["guizhu", "montesquieu"],
        "domain": "HR management, recruitment, employee development",
    },
    "product": {
        "souls": ["davinci", "strategy"],
        "domain": "Product management, roadmap, user research",
    },
    "research": {
        "souls": ["einstein", "galileo", "darwin"],
        "domain": "Research methodology, literature review, hypothesis testing",
    },
    "support": {
        "souls": ["guizhu", "herodotus"],
        "domain": "Customer support, troubleshooting, knowledge base",
    },
    "operations": {
        "souls": ["humboldt", "cezanne"],
        "domain": "Operations management, process optimization, logistics",
    },
    "writing": {
        "souls": ["herodotus", "monet"],
        "domain": "Technical writing, documentation, content strategy",
    },
    "education": {
        "souls": ["herodotus", "guizhu"],
        "domain": "Education design, curriculum development, assessment",
    },
    "healthcare": {
        "souls": ["darwin", "guizhu"],
        "domain": "Healthcare analytics, clinical decision support, patient care",
    },
    "mano_p_gui": {
        "souls": ["cezanne", "davinci"],
        "domain": "GUI perception and operation, computer control, screen interaction, visual automation",
    },
}


class SkillRegistry:
    """
    Unified skill registry for VORTEX_FLAME.

    Manages skills from multiple sources:
    - External: knowledge-work-plugins (Anthropic official)
    - Evolved: skill_evolver output (VORTEX_FLAME proprietary)
    - Custom: User-defined skills

    Lifecycle:
      External Skill → Seed → Auditing → Active → (optional) Evolved
                                         ↘ Deprecated

    Integration Points:
    - soul_config/*.yaml: `skills` field references registered skill IDs
    - skill_evolver.py: External skills serve as Stage 0 seeds
    - soul_orchestrator.py: Skills activate based on task matching
    - soul_memory.py: Skill usage tracked in `skill` memory category
    """

    def __init__(self):
        self._skills: Dict[str, RegisteredSkill] = {}

    def register_external(
        self,
        plugin_name: str,
        commands: List[Dict[str, Any]],
        connectors: List[Dict[str, Any]],
        description: str = "",
    ) -> RegisteredSkill:
        """
        Register a knowledge-work-plugin as an external skill.
        Maps to soul(s) based on KNOWLEDGE_WORK_PLUGIN_MAPPING.
        """
        mapping = KNOWLEDGE_WORK_PLUGIN_MAPPING.get(plugin_name)
        if mapping is None:
            mapping = {"souls": ["cezanne"], "domain": description or plugin_name}

        skill_commands = [
            SkillCommand(
                name=c.get("name", ""),
                description=c.get("description", ""),
                template=c.get("template", ""),
                parameters=c.get("parameters", []),
            )
            for c in commands
        ]

        skill_connectors = [
            SkillConnector(
                name=cn.get("name", ""),
                connector_type=cn.get("type", ""),
                config=cn.get("config", {}),
            )
            for cn in connectors
        ]

        skill = RegisteredSkill(
            skill_id=f"kwp_{plugin_name}",
            name=plugin_name,
            description=mapping["domain"],
            source=SkillSource.EXTERNAL,
            status=SkillStatus.SEED,
            soul_mapping=mapping["souls"],
            commands=skill_commands,
            connectors=skill_connectors,
        )

        self._skills[skill.skill_id] = skill
        return skill

    def register_evolved(
        self,
        skill_name: str,
        core_rules: str,
        errata: List[str],
        audit_scores: Dict[str, float],
        soul_mapping: List[str],
    ) -> RegisteredSkill:
        """
        Register a skill that has been evolved by skill_evolver.
        Replaces the corresponding external skill if it exists.
        """
        skill = RegisteredSkill(
            skill_id=f"evolved_{skill_name}",
            name=skill_name,
            description="",
            source=SkillSource.EVOLVED,
            status=SkillStatus.ACTIVE,
            soul_mapping=soul_mapping,
            core_rules=core_rules,
            errata=errata,
            audit_scores=audit_scores,
        )

        seed_id = f"kwp_{skill_name}"
        if seed_id in self._skills:
            self._skills[seed_id].status = SkillStatus.DEPRECATED

        self._skills[skill.skill_id] = skill
        return skill

    def get_skills_for_soul(self, soul_name: str) -> List[RegisteredSkill]:
        """
        Get all active skills mapped to a specific soul.
        Used by soul_orchestrator to activate skills during task dispatch.
        """
        return [
            s
            for s in self._skills.values()
            if soul_name in s.soul_mapping and s.status in (SkillStatus.SEED, SkillStatus.ACTIVE)
        ]

    def get_skill(self, skill_id: str) -> Optional[RegisteredSkill]:
        return self._skills.get(skill_id)

    def list_skills(
        self,
        source: Optional[SkillSource] = None,
        status: Optional[SkillStatus] = None,
        soul: Optional[str] = None,
    ) -> List[RegisteredSkill]:
        skills = list(self._skills.values())
        if source:
            skills = [s for s in skills if s.source == source]
        if status:
            skills = [s for s in skills if s.status == status]
        if soul:
            skills = [s for s in skills if soul in s.soul_mapping]
        return skills

    def get_magic_keywords(self) -> Dict[str, str]:
        """
        Generate magic keyword mappings from skill commands.
        Each command name becomes a trigger for the corresponding soul + skill.
        """
        keywords = {}
        for skill in self._skills.values():
            if skill.status not in (SkillStatus.SEED, SkillStatus.ACTIVE):
                continue
            for cmd in skill.commands:
                if cmd.name:
                    keywords[cmd.name] = skill.soul_mapping[0] if skill.soul_mapping else "cezanne"
        return keywords

    def to_soul_config_format(self, soul_name: str) -> List[Dict[str, Any]]:
        """
        Export skills for a soul in the format expected by soul_config YAML.
        Used to populate the `skills` field in soul YAML files.
        """
        skills = self.get_skills_for_soul(soul_name)
        return [
            {
                "id": s.skill_id,
                "name": s.name,
                "source": s.source.value,
                "status": s.status.value,
                "commands": [c.name for c in s.commands],
            }
            for s in skills
        ]
