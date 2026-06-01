"""
Soul Knowledge Alignment — RAG Knowledge Base Auto-Indexer
===========================================================
Automatically downloads and indexes殿堂级 open knowledge bases
into soul_memory for each of the 14 VORTEX_FLAME souls.

Verified Data Sources (2026-05-28):
- librarian-bots/arxiv-metadata-snapshot — 2.5M arXiv papers, fields: id/title/categories/abstract
- ashish-chouhan/arxiv_cs_papers          — CS-only arXiv subset, fields: title/abstract/arxiv_id
- wikimedia/wikipedia (20231101.en)       — 6.4M Wikipedia articles, fields: id/url/title/text

RAG Pipeline: Load → Clean → Chunk → Embed → Store → Retrieve
- All sources converted to clean text before indexing
- BM25 + semantic hybrid search via soul_memory
- No GPU required, no external API keys
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_CHECKPOINT_DIR = Path(os.environ.get("VORTEX_FLAME_MEMORY_DIR", ".vf_memory")) / "checkpoints"
_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

ARXIV_FULL = "librarian-bots/arxiv-metadata-snapshot"
ARXIV_CS = "ashish-chouhan/arxiv_cs_papers"
WIKI_EN = "wikimedia/wikipedia"
WIKI_SUBSET = "20231101.en"

ARXIV_CATEGORY_PREFIXES = {
    "cezanne": ["cs."],
    "einstein": ["hep-", "quant-ph", "cond-mat", "astro-ph", "gr-qc", "math-ph", "physics.", "q-fin.", "math."],
    "galileo": ["astro-ph", "gr-qc", "physics.space-ph", "physics.hist-ph"],
    "darwin": ["q-bio.", "physics.bio-ph", "stat.AP"],
    "davinci": ["eess.", "cs.", "physics.app-ph"],
    "strategy": ["cs.GT", "cs.AI", "econ.", "math.OC", "stat.ML", "q-fin."],
    "montesquieu": ["cs.CY", "cs.AI", "cs.LG"],
    "humboldt": ["physics.ao-ph", "physics.geo-ph", "physics.atm-clus", "astro-ph.EP"],
    "yuanlongping": ["q-bio.", "stat.AP"],
    "guizhu": [],
    "herodotus": [],
    "monet": [],
    "vangogh": ["cs.CV", "cs.GR", "eess.IV"],
    "beethoven": ["eess.AS", "cs.SD", "cs.MM", "physics.acc-ph"],
}

SOUL_KNOWLEDGE_MAP = {
    "cezanne": {
        "full_name": "Paul Cézanne",
        "domains": ["Code", "Logic", "Algorithm", "Systems"],
        "datasets": [
            {
                "name": "arxiv_cs",
                "hf_id": ARXIV_CS,
                "hf_split": "train",
                "content_fields": ["title", "abstract"],
                "category": "knowledge",
                "max_entries": 50000,
                "priority": "P0",
            },
            {
                "name": "wikipedia_cs",
                "hf_id": WIKI_EN,
                "hf_subset": WIKI_SUBSET,
                "hf_split": "train",
                "hf_filter": {"title_keywords": ["algorithm", "programming", "software", "data structure", "operating system", "compiler", "database", "distributed computing", "concurrent", "computer science", "machine learning", "artificial intelligence", "deep learning", "neural network", "binary tree", "hash table", "sorting", "recursion", "complexity", "API", "framework", "debugging"]},
                "content_fields": ["text"],
                "category": "knowledge",
                "max_entries": 10000,
                "priority": "P0",
                "use_cache": True,
            },
        ],
    },
    "einstein": {
        "full_name": "Albert Einstein",
        "domains": ["Physics", "Quantum Mechanics", "Quantitative Finance", "Innovation"],
        "datasets": [
            {
                "name": "arxiv_physics",
                "hf_id": ARXIV_FULL,
                "hf_split": "train",
                "hf_filter": {"category_prefixes": ["hep-", "quant-ph", "cond-mat", "astro-ph", "gr-qc", "math-ph", "physics.", "q-fin.", "math."]},
                "content_fields": ["title", "abstract"],
                "category": "knowledge",
                "max_entries": 50000,
                "priority": "P0",
            },
            {
                "name": "wikipedia_physics",
                "hf_id": WIKI_EN,
                "hf_subset": WIKI_SUBSET,
                "hf_split": "train",
                "hf_filter": {"title_keywords": ["physics", "quantum", "relativity", "thermodynamics", "electromagnetism", "particle physics", "mechanics", "field theory", "wave", "Schrödinger", "Einstein", "Newton", "Maxwell", "entropy", "Hamiltonian", "Lagrangian", "boson", "fermion", "quark", "string theory", "dark matter", "dark energy"]},
                "content_fields": ["text"],
                "category": "knowledge",
                "max_entries": 8000,
                "priority": "P0",
                "use_cache": True,
            },
        ],
    },
    "galileo": {
        "full_name": "Galileo Galilei",
        "domains": ["Astronomy", "Astrophysics", "Orbital Mechanics"],
        "datasets": [
            {
                "name": "arxiv_astro",
                "hf_id": ARXIV_FULL,
                "hf_split": "train",
                "hf_filter": {"category_prefixes": ["astro-ph", "gr-qc", "physics.space-ph", "physics.hist-ph"]},
                "content_fields": ["title", "abstract"],
                "category": "knowledge",
                "max_entries": 30000,
                "priority": "P0",
            },
            {
                "name": "wikipedia_astronomy",
                "hf_id": WIKI_EN,
                "hf_subset": WIKI_SUBSET,
                "hf_split": "train",
                "hf_filter": {"title_keywords": ["astronomy", "astrophysics", "galaxy", "star", "planet", "orbit", "telescope", "cosmology", "black hole", "nebula", "solar system", "constellation", "supernova", "pulsar", "quasar", "exoplanet", "asteroid", "comet", "red giant", "white dwarf", "neutron star"]},
                "content_fields": ["text"],
                "category": "knowledge",
                "max_entries": 5000,
                "priority": "P0",
                "use_cache": True,
            },
        ],
    },
    "darwin": {
        "full_name": "Charles Darwin",
        "domains": ["Biology", "Genetics", "Evolution", "Healthcare"],
        "datasets": [
            {
                "name": "arxiv_biology",
                "hf_id": ARXIV_FULL,
                "hf_split": "train",
                "hf_filter": {"category_prefixes": ["q-bio.", "physics.bio-ph", "stat.AP"]},
                "content_fields": ["title", "abstract"],
                "category": "knowledge",
                "max_entries": 15000,
                "priority": "P0",
            },
            {
                "name": "wikipedia_biology",
                "hf_id": WIKI_EN,
                "hf_subset": WIKI_SUBSET,
                "hf_split": "train",
                "hf_filter": {"title_keywords": ["biology", "evolution", "genetics", "DNA", "cell biology", "protein", "species", "ecology", "organism", "mutation", "genome", "natural selection", "Darwin", "Mendel", "CRISPR", "RNA", "chromosome", "mitosis", "meiosis", "photosynthesis"]},
                "content_fields": ["text"],
                "category": "knowledge",
                "max_entries": 8000,
                "priority": "P0",
                "use_cache": True,
            },
        ],
    },
    "davinci": {
        "full_name": "Leonardo da Vinci",
        "domains": ["Engineering", "Architecture", "Design"],
        "datasets": [
            {
                "name": "arxiv_engineering",
                "hf_id": ARXIV_FULL,
                "hf_split": "train",
                "hf_filter": {"category_prefixes": ["eess.", "cs.", "physics.app-ph"]},
                "content_fields": ["title", "abstract"],
                "category": "knowledge",
                "max_entries": 20000,
                "priority": "P0",
            },
            {
                "name": "wikipedia_engineering",
                "hf_id": WIKI_EN,
                "hf_subset": WIKI_SUBSET,
                "hf_split": "train",
                "hf_filter": {"title_keywords": ["engineering", "architecture", "design", "mechanical engineering", "electrical engineering", "civil engineering", "structural", "CAD", "robotics", "automation", "manufacturing", "bridge", "dam", "skyscraper", "turbine", "engine", "circuit", "sensor", "actuator"]},
                "content_fields": ["text"],
                "category": "knowledge",
                "max_entries": 6000,
                "priority": "P0",
                "use_cache": True,
            },
        ],
    },
    "strategy": {
        "full_name": "John Nash",
        "domains": ["Game Theory", "Strategy", "Decision Making", "Finance"],
        "datasets": [
            {
                "name": "arxiv_game_theory",
                "hf_id": ARXIV_FULL,
                "hf_split": "train",
                "hf_filter": {"category_prefixes": ["cs.GT", "cs.AI", "econ.", "math.OC", "stat.ML", "q-fin."]},
                "content_fields": ["title", "abstract"],
                "category": "knowledge",
                "max_entries": 15000,
                "priority": "P0",
            },
            {
                "name": "wikipedia_strategy",
                "hf_id": WIKI_EN,
                "hf_subset": WIKI_SUBSET,
                "hf_split": "train",
                "hf_filter": {"title_keywords": ["game theory", "Nash equilibrium", "decision theory", "auction", "mechanism design", "optimization", "finance", "investment", "portfolio", "prisoner dilemma", "Pareto", "oligopoly", "bargaining", "rational choice", "expected utility", "Bayesian game"]},
                "content_fields": ["text"],
                "category": "knowledge",
                "max_entries": 5000,
                "priority": "P0",
                "use_cache": True,
            },
        ],
    },
    "montesquieu": {
        "full_name": "Montesquieu",
        "domains": ["Law", "Political Science", "Logic", "Compliance"],
        "datasets": [
            {
                "name": "wikipedia_law",
                "hf_id": WIKI_EN,
                "hf_subset": WIKI_SUBSET,
                "hf_split": "train",
                "hf_filter": {"title_keywords": ["law", "legal", "constitution", "court", "legislation", "jurisprudence", "political", "governance", "regulation", "compliance", "rights", "justice", "separation of powers", "judicial", "statute", "common law", "civil law", "criminal law", "administrative law", "international law"]},
                "content_fields": ["text"],
                "category": "knowledge",
                "max_entries": 8000,
                "priority": "P0",
                "use_cache": True,
            },
            {
                "name": "arxiv_law",
                "hf_id": ARXIV_CS,
                "hf_split": "train",
                "content_fields": ["title", "abstract"],
                "category": "knowledge",
                "max_entries": 5000,
                "priority": "P1",
            },
        ],
    },
    "humboldt": {
        "full_name": "Alexander von Humboldt",
        "domains": ["Geography", "Ecology", "Earth Science", "Data Analysis"],
        "datasets": [
            {
                "name": "arxiv_earth",
                "hf_id": ARXIV_FULL,
                "hf_split": "train",
                "hf_filter": {"category_prefixes": ["physics.ao-ph", "physics.geo-ph", "physics.atm-clus", "astro-ph.EP"]},
                "content_fields": ["title", "abstract"],
                "category": "knowledge",
                "max_entries": 10000,
                "priority": "P0",
            },
            {
                "name": "wikipedia_earth",
                "hf_id": WIKI_EN,
                "hf_subset": WIKI_SUBSET,
                "hf_split": "train",
                "hf_filter": {"title_keywords": ["geography", "ecology", "climate", "ocean", "geology", "atmosphere", "environment", "ecosystem", "biodiversity", "earth science", "volcano", "earthquake", "tectonic", "glacier", "coral reef", "rainforest", "desert", "tundra", "watershed"]},
                "content_fields": ["text"],
                "category": "knowledge",
                "max_entries": 8000,
                "priority": "P0",
                "use_cache": True,
            },
        ],
    },
    "yuanlongping": {
        "full_name": "Yuan Longping",
        "domains": ["Agriculture", "Genetics", "Food Science"],
        "datasets": [
            {
                "name": "wikipedia_agriculture",
                "hf_id": WIKI_EN,
                "hf_subset": WIKI_SUBSET,
                "hf_split": "train",
                "hf_filter": {"title_keywords": ["agriculture", "crop", "rice", "wheat", "fertilizer", "irrigation", "soil", "harvest", "hybrid", "cultivation", "food security", "grain", "pest", "farm", "horticulture", "agronomy", "plant breeding", "genetically modified", "drought", "pesticide"]},
                "content_fields": ["text"],
                "category": "knowledge",
                "max_entries": 5000,
                "priority": "P0",
                "use_cache": True,
            },
            {
                "name": "arxiv_agriculture",
                "hf_id": ARXIV_FULL,
                "hf_split": "train",
                "hf_filter": {"category_prefixes": ["q-bio.", "stat.AP"]},
                "content_fields": ["title", "abstract"],
                "category": "knowledge",
                "max_entries": 5000,
                "priority": "P1",
            },
        ],
    },
    "guizhu": {
        "full_name": "Guizhu Jushi",
        "domains": ["Philosophy", "Logic", "Dialogue", "Psychology"],
        "datasets": [
            {
                "name": "wikipedia_philosophy",
                "hf_id": WIKI_EN,
                "hf_subset": WIKI_SUBSET,
                "hf_split": "train",
                "hf_filter": {"title_keywords": ["philosophy", "logic", "ethics", "metaphysics", "epistemology", "consciousness", "philosophy of mind", "dialogue", "Socratic", "Buddhism", "Taoism", "stoicism", "existentialism", "phenomenology", "pragmatism", "Kant", "Hegel", "Aristotle", "Plato", "Confucius", "Nietzsche", "Wittgenstein"]},
                "content_fields": ["text"],
                "category": "knowledge",
                "max_entries": 8000,
                "priority": "P0",
                "use_cache": True,
            },
            {
                "name": "wikipedia_psychology",
                "hf_id": WIKI_EN,
                "hf_subset": WIKI_SUBSET,
                "hf_split": "train",
                "hf_filter": {"title_keywords": ["psychology", "cognitive", "behavior", "therapy", "emotion", "personality", "mental health", "perception", "motivation", "learning", "Freud", "Jung", "cognitive behavioral", "neuroscience", "developmental", "social psychology"]},
                "content_fields": ["text"],
                "category": "knowledge",
                "max_entries": 5000,
                "priority": "P1",
                "use_cache": True,
            },
        ],
    },
    "herodotus": {
        "full_name": "Herodotus",
        "domains": ["History", "Causality", "Civilization", "Documentation"],
        "datasets": [
            {
                "name": "wikipedia_history",
                "hf_id": WIKI_EN,
                "hf_subset": WIKI_SUBSET,
                "hf_split": "train",
                "hf_filter": {"title_keywords": ["history", "ancient", "medieval", "civilization", "empire", "war", "revolution", "dynasty", "archaeology", "chronology", "historian", "colonial", "Renaissance", "Industrial Revolution", "World War", "Roman Empire", "Greek", "Egyptian", "Mesopotamia", "Byzantine"]},
                "content_fields": ["text"],
                "category": "knowledge",
                "max_entries": 10000,
                "priority": "P0",
                "use_cache": True,
            },
        ],
    },
    "monet": {
        "full_name": "Claude Monet",
        "domains": ["Aesthetics", "Creative Writing", "Art Therapy", "Marketing"],
        "datasets": [
            {
                "name": "wikipedia_art",
                "hf_id": WIKI_EN,
                "hf_subset": WIKI_SUBSET,
                "hf_split": "train",
                "hf_filter": {"title_keywords": ["art", "painting", "sculpture", "aesthetics", "impressionism", "renaissance", "baroque", "modern art", "gallery", "museum", "composition", "color theory", "visual", "abstract art", "cubism", "surrealism", "expressionism", "watercolor", "oil painting", "portrait"]},
                "content_fields": ["text"],
                "category": "knowledge",
                "max_entries": 6000,
                "priority": "P0",
                "use_cache": True,
            },
            {
                "name": "wikipedia_marketing",
                "hf_id": WIKI_EN,
                "hf_subset": WIKI_SUBSET,
                "hf_split": "train",
                "hf_filter": {"title_keywords": ["marketing", "brand", "advertising", "consumer", "campaign", "social media", "content strategy", "copywriting", "public relations", "SEO", "digital marketing"]},
                "content_fields": ["text"],
                "category": "knowledge",
                "max_entries": 3000,
                "priority": "P1",
                "use_cache": True,
            },
        ],
    },
    "vangogh": {
        "full_name": "Vincent van Gogh",
        "domains": ["Emotion", "Visual Art", "Color Science"],
        "datasets": [
            {
                "name": "wikipedia_visual_art",
                "hf_id": WIKI_EN,
                "hf_subset": WIKI_SUBSET,
                "hf_split": "train",
                "hf_filter": {"title_keywords": ["art", "painting", "color", "pigment", "light", "vision", "optical", "perception", "canvas", "brushwork", "expressionism", "post-impressionism", "Van Gogh", "Monet", "Cézanne", "Gauguin", "Seurat", "pointillism", "fauvism"]},
                "content_fields": ["text"],
                "category": "knowledge",
                "max_entries": 5000,
                "priority": "P0",
                "use_cache": True,
            },
            {
                "name": "arxiv_color_science",
                "hf_id": ARXIV_CS,
                "hf_split": "train",
                "content_fields": ["title", "abstract"],
                "category": "knowledge",
                "max_entries": 5000,
                "priority": "P1",
            },
        ],
    },
    "beethoven": {
        "full_name": "Ludwig van Beethoven",
        "domains": ["Music", "Acoustics", "Language Composition"],
        "datasets": [
            {
                "name": "wikipedia_music",
                "hf_id": WIKI_EN,
                "hf_subset": WIKI_SUBSET,
                "hf_split": "train",
                "hf_filter": {"title_keywords": ["music", "symphony", "composition", "harmony", "melody", "rhythm", "acoustics", "sound", "orchestra", "instrument", "chord", "scale", "tempo", "sonata", "concerto", "Beethoven", "Mozart", "Bach", "Chopin", "opera", "jazz", "blues"]},
                "content_fields": ["text"],
                "category": "knowledge",
                "max_entries": 8000,
                "priority": "P0",
                "use_cache": True,
            },
            {
                "name": "arxiv_audio",
                "hf_id": ARXIV_FULL,
                "hf_split": "train",
                "hf_filter": {"category_prefixes": ["eess.AS", "cs.SD", "cs.MM", "physics.acc-ph"]},
                "content_fields": ["title", "abstract"],
                "category": "knowledge",
                "max_entries": 5000,
                "priority": "P1",
            },
        ],
    },
}


def _matches_filter(record: dict, filter_config: dict) -> bool:
    if not filter_config:
        return True

    if "category_prefixes" in filter_config:
        cats = record.get("categories", "")
        if not cats:
            return False
        cats_lower = cats.lower()
        if not any(cats_lower.startswith(p) or f" {p}" in cats_lower for p in filter_config["category_prefixes"]):
            return False

    if "title_keywords" in filter_config:
        title = (record.get("title", "") or "").lower()
        if not any(kw.lower() in title for kw in filter_config["title_keywords"]):
            return False

    return True


def _extract_content(record: dict, content_fields: List[str]) -> str:
    parts = []
    for field in content_fields:
        val = record.get(field)
        if val is None:
            continue
        if isinstance(val, list):
            parts.extend(str(v) for v in val)
        else:
            parts.append(str(val))
    return "\n".join(parts)


def _stream_hf_dataset(hf_id: str, split: str = "train",
                       subset: Optional[str] = None,
                       filter_config: Optional[dict] = None,
                       content_fields: Optional[List[str]] = None,
                       max_entries: int = 10000,
                       use_cache: bool = False) -> List[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("pip install datasets required")
        return []

    logger.info(f"Loading {hf_id} (subset={subset}, cache={use_cache})...")

    try:
        if use_cache:
            if subset:
                ds = load_dataset(hf_id, subset, split=split)
            else:
                ds = load_dataset(hf_id, split=split)
        else:
            if subset:
                ds = load_dataset(hf_id, subset, split=split, streaming=True)
            else:
                ds = load_dataset(hf_id, split=split, streaming=True)
    except Exception as e:
        logger.warning(f"Failed to load {hf_id}: {e}")
        return []

    entries = []
    scanned = 0
    last_log = time.time()

    for record in ds:
        scanned += 1
        now = time.time()
        if now - last_log > 30:
            logger.info(f"  Scanned {scanned}, matched {len(entries)}/{max_entries}")
            last_log = now

        if filter_config and not _matches_filter(record, filter_config):  # type: ignore[reportArgumentType]
            continue

        content = _extract_content(record, content_fields or ["text"])  # type: ignore[reportArgumentType]
        if not content or len(content.strip()) < 50:
            continue

        title = record.get("title", record.get("name", "")) or ""  # type: ignore[reportAttributeAccessIssue]
        entry = {
            "topic": title[:200] if title else "untitled",
            "detail": content[:5000],
            "source": hf_id,
        }
        entries.append(entry)

        if len(entries) >= max_entries:
            break

    logger.info(f"  Collected {len(entries)} entries (scanned {scanned})")
    return entries


def _load_checkpoint(soul: str) -> dict:
    cp_file = _CHECKPOINT_DIR / f"{soul}_alignment.json"
    if cp_file.exists():
        try:
            return json.loads(cp_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_checkpoint(soul: str, state: dict):
    cp_file = _CHECKPOINT_DIR / f"{soul}_alignment.json"
    cp_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

def align_soul(soul: str, max_per_dataset: Optional[int] = None,
               resume: bool = True) -> dict:
    from soul_memory import write, search

    config = SOUL_KNOWLEDGE_MAP.get(soul)
    if not config:
        return {"soul": soul, "status": "unknown_soul"}

    checkpoint = _load_checkpoint(soul) if resume else {}
    total_indexed = checkpoint.get("total_indexed", 0)
    dataset_stats = checkpoint.get("dataset_stats", [])
    completed_datasets = {ds["dataset"] for ds in dataset_stats if ds.get("completed")}

    for ds_config in config["datasets"]:
        ds_name = ds_config["name"]
        if ds_name in completed_datasets:
            logger.info(f"[{soul}] Skipping completed dataset: {ds_name}")
            continue

        max_entries = max_per_dataset or ds_config.get("max_entries", 10000)
        ds_offset = checkpoint.get("offsets", {}).get(ds_name, 0)

        logger.info(f"[{soul}] Indexing {ds_name} (max={max_entries}, offset={ds_offset})...")

        entries = _stream_hf_dataset(
            hf_id=ds_config["hf_id"],
            split=ds_config.get("hf_split", "train"),
            subset=ds_config.get("hf_subset"),
            filter_config=ds_config.get("hf_filter"),
            content_fields=ds_config.get("content_fields"),
            max_entries=max_entries,
            use_cache=ds_config.get("use_cache", False),
        )

        indexed = 0
        processed = 0
        for entry in entries:
            processed += 1
            if processed <= ds_offset:
                continue
            try:
                existing = search(soul, ds_config["category"], entry["topic"], top_k=1)
                if existing and any(e.get("content", {}).get("topic") == entry["topic"] for e in existing):
                    continue

                write(soul, ds_config["category"], entry,
                      importance=0.7 if ds_config.get("priority") == "P0" else 0.5,
                      tags=[ds_name, ds_config.get("priority", "P1")])
                indexed += 1
            except Exception as e:
                logger.debug(f"Skip entry: {e}")

            if processed % 500 == 0:
                checkpoint.setdefault("offsets", {})[ds_name] = processed
                checkpoint["total_indexed"] = total_indexed + indexed
                _save_checkpoint(soul, checkpoint)

        total_indexed += indexed
        ds_stat = {
            "dataset": ds_name,
            "collected": len(entries),
            "indexed": indexed,
            "priority": ds_config.get("priority", "P1"),
            "completed": True,
        }
        dataset_stats.append(ds_stat)
        checkpoint.setdefault("offsets", {})[ds_name] = processed
        checkpoint["total_indexed"] = total_indexed
        checkpoint["dataset_stats"] = dataset_stats
        _save_checkpoint(soul, checkpoint)
        logger.info(f"  {ds_name}: {indexed}/{len(entries)} indexed")

    checkpoint["status"] = "aligned"
    _save_checkpoint(soul, checkpoint)

    return {
        "soul": soul,
        "full_name": config["full_name"],
        "domains": config["domains"],
        "total_indexed": total_indexed,
        "datasets": dataset_stats,
        "status": "aligned",
    }


def align_all(souls: Optional[List[str]] = None,
              max_per_dataset: Optional[int] = None) -> dict:
    target_souls = souls or list(SOUL_KNOWLEDGE_MAP.keys())
    results = {}

    for soul in target_souls:
        logger.info(f"\n{'='*60}")
        logger.info(f"Aligning knowledge for {soul}...")
        logger.info(f"{'='*60}")
        result = align_soul(soul, max_per_dataset=max_per_dataset)
        results[soul] = result

    total = sum(r["total_indexed"] for r in results.values())
    summary = {
        "total_souls": len(results),
        "total_entries_indexed": total,
        "souls": results,
        "status": "complete",
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"ALIGNMENT COMPLETE: {total} entries indexed across {len(results)} souls")
    logger.info(f"{'='*60}")

    return summary


def get_alignment_status() -> dict:
    from soul_memory import _engine as eng

    status = {}
    for soul, config in SOUL_KNOWLEDGE_MAP.items():
        try:
            conn = eng._get_db(soul)
            count = conn.execute("SELECT COUNT(*) as cnt FROM memories WHERE category = 'knowledge'").fetchone()["cnt"]
            status[soul] = {
                "full_name": config["full_name"],
                "domains": config["domains"],
                "knowledge_entries": count,
                "datasets_configured": len(config["datasets"]),
                "status": "aligned" if count > 100 else "needs_alignment",
            }
        except Exception:
            status[soul] = {
                "full_name": config["full_name"],
                "domains": config["domains"],
                "knowledge_entries": 0,
                "datasets_configured": len(config["datasets"]),
                "status": "empty",
            }

    return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    import argparse
    parser = argparse.ArgumentParser(description="Soul Knowledge Alignment")
    parser.add_argument("--soul", type=str, default=None, help="Align specific soul")
    parser.add_argument("--all", action="store_true", help="Align all souls")
    parser.add_argument("--status", action="store_true", help="Show alignment status")
    parser.add_argument("--max", type=int, default=None, help="Max entries per dataset")
    args = parser.parse_args()

    if args.status:
        status = get_alignment_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))
    elif args.soul:
        result = align_soul(args.soul, max_per_dataset=args.max)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.all:
        result = align_all(max_per_dataset=args.max)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        parser.print_help()
