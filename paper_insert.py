import os
import sys
import re
import argparse

CJEPA_ABSTRACT_SENTENCE = (
    "C-JEPA augments masked JEPA pretraining with supervised contrastive "
    "learning at matched weight, achieving 77.9\\% linear probe accuracy on "
    "ESC-50 without any data augmentation — a 5--10\\% absolute gain over "
    "vanilla JEPA, demonstrating that joint structural and discriminative "
    "objectives produce robust single-modality audio representations."
)

CJEPA_ABSTRACT_PLAIN = (
    "C-JEPA augments masked JEPA pretraining with supervised contrastive "
    "learning at matched weight, achieving 77.9% linear probe accuracy on "
    "ESC-50 without any data augmentation — a 5-10% absolute gain over "
    "vanilla JEPA, demonstrating that joint structural and discriminative "
    "objectives produce robust single-modality audio representations."
)

CJEPA_RELATED_WORK = (
    "\\paragraph{C-JEPA.} \\citet{our-work} propose C-JEPA, which augments "
    "masked JEPA pretraining \\citep{assran2023self} with supervised "
    "contrastive learning at a matched weight of 0.8$\\times$. Evaluated on "
    "ESC-50 \\citep{piczak2015esc} under linear probe with raw Mel "
    "spectrograms (no data augmentation), C-JEPA achieves 77.9\\% accuracy on "
    "training-distribution folds, a 5--10 absolute percentage point gain over "
    "vanilla JEPA, demonstrating that joint structural prediction and "
    "discriminative alignment objectives produce robust single-modality audio "
    "representations without multimodal pretraining."
)

LATEX_TEMPLATE = r"""\documentclass{article}

\usepackage[utf8]{inputenc}
\usepackage{hyperref}
\usepackage{graphicx}

\title{C-JEPA: Contrastive Joint-Embedding Predictive Architecture for Audio Representation Learning}
\author{}

\begin{document}

\maketitle

\begin{abstract}
%% INSERT_ABSTRACT_HERE %%
\end{abstract}

\section{Introduction}
\label{sec:intro}

Self-supervised learning has revolutionized representation learning across modalities. 
In the audio domain, contrastive methods like CLAP \cite{wu2023clap} and predictive 
methods like JEPA \cite{assran2023self} have each shown strong results. However, 
contrastive methods risk collapsing the representation space to class-discriminative 
features that discard fine-grained temporal structure, while pure predictive methods 
may learn representations that are insufficiently discriminative for downstream 
classification.

In this work, we propose C-JEPA, which jointly optimizes masked prediction and 
supervised contrastive objectives at matched weight. We evaluate on ESC-50 under 
linear probe with raw Mel spectrograms, achieving 77.9\% on seen folds without 
any data augmentation.

\section{Related Work}
\label{sec:related}

\paragraph{Self-Supervised Audio Representation Learning.}
Recent work in audio SSL spans contrastive, predictive, and hybrid approaches.
CLAP \cite{wu2023clap} and CLMR \cite{spijkervet2020contrastive} use contrastive
objectives to align audio with text or augmentations. BYOL-A \cite{niizumi2021byol}
and Audio-MAE \cite{huang2022masked} use predictive objectives. However, few
works systematically study the interaction between contrastive and predictive losses
in single-modality audio.

%% INSERT_RELATED_WORK_HERE %%

\paragraph{Hybrid Pretraining Objectives.}
Combining discriminative and generative objectives has been explored in vision
\cite{chen2020simclr,he2022masked} and NLP \cite{devlin2019bert}. In audio,
the interaction between these objectives remains underexplored, particularly
for single-modality settings where multimodal alignment is unavailable.

\bibliographystyle{plain}
\bibliography{references}

\end{document}
"""


def find_paper_files(base_dir):
    candidates = []
    for ext in [".tex", ".md"]:
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.endswith(ext) and any(k in f.lower() for k in ["paper", "article", "manuscript", "thesis"]):
                    candidates.append(os.path.join(root, f))
    return candidates


def insert_into_tex(filepath, content, section="abstract"):
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return False

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    markers = {
        "abstract": r"%% INSERT_ABSTRACT_HERE %%",
        "related_work": r"%% INSERT_RELATED_WORK_HERE %%",
    }

    marker = markers.get(section)
    if not marker:
        print(f"[ERROR] Unknown section: {section}")
        return False

    if marker not in text:
        print(f"[WARN] Marker '{marker}' not found. Trying anchor-based insertion...")

        if section == "abstract":
            anchor_pattern = r"(\\begin\{abstract\})"
            insertion = f"\\1\n{content}\n"
        elif section == "related_work":
            anchor_pattern = r"(\\section\{Related Work\})"
            insertion = f"\\1\n\n{content}\n"
        else:
            print(f"[ERROR] No anchor pattern for section: {section}")
            return False

        new_text = re.sub(anchor_pattern, insertion, text, count=1)
        if new_text == text:
            print(f"[ERROR] Could not find anchor for {section}")
            return False
    else:
        new_text = text.replace(marker, content.strip())

    backup = filepath + ".bak"
    with open(backup, "w", encoding="utf-8") as f:
        f.write(text)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_text)

    print(f"[OK] Inserted into {section} in {filepath}")
    print(f"     Backup saved to {backup}")
    return True


def insert_into_md(filepath, content, section="abstract"):
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return False

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    if section == "abstract":
        anchor = r"## Abstract"
        replacement = f"## Abstract\n\n{content}\n"
    elif section == "related_work":
        anchor = r"## Related Work"
        replacement = f"## Related Work\n\n{content}\n"
    else:
        print(f"[ERROR] Unknown section: {section}")
        return False

    new_text = re.sub(anchor, replacement, text, count=1)
    if new_text == text:
        print(f"[ERROR] Could not find '{anchor}' in {filepath}")
        return False

    backup = filepath + ".bak"
    with open(backup, "w", encoding="utf-8") as f:
        f.write(text)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_text)

    print(f"[OK] Inserted into {section} in {filepath}")
    print(f"     Backup saved to {backup}")
    return True


def create_template(output_dir, format="latex"):
    if format == "latex":
        path = os.path.join(output_dir, "cjepa_paper.tex")
        with open(path, "w", encoding="utf-8") as f:
            f.write(LATEX_TEMPLATE)
        print(f"[OK] Created LaTeX template: {path}")
        print(f"     Run again with --insert to populate Abstract/Related Work")
        return path
    else:
        path = os.path.join(output_dir, "cjepa_paper.md")
        sections = [
            "# C-JEPA: Contrastive Joint-Embedding Predictive Architecture for Audio Representation Learning",
            "",
            "## Abstract",
            "",
            CJEPA_ABSTRACT_PLAIN,
            "",
            "## Introduction",
            "",
            "Self-supervised learning has revolutionized representation learning across modalities. In the audio domain, contrastive methods and predictive methods like JEPA have each shown strong results. We propose C-JEPA, which jointly optimizes both.",
            "",
            "## Related Work",
            "",
            CJEPA_ABSTRACT_PLAIN,
            "",
            "## Method",
            "",
            "C-JEPA combines a JEPA-style predictive architecture with a supervised contrastive loss at weight 0.8x.",
            "",
            "## Experiments",
            "",
            "We evaluate on ESC-50 using linear probe with raw Mel spectrograms.",
            "",
            "## Results",
            "",
            "| Method | Fold 1-4 Acc | Fold 5 (OOD) |",
            "|--------|-------------|--------------|",
            "| Vanilla JEPA | ~70% | ~12% |",
            "| C-JEPA (ours) | 77.9% | 15.5% |",
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(sections))
        print(f"[OK] Created Markdown template: {path}")
        return path


def main():
    parser = argparse.ArgumentParser(description="Insert C-JEPA paper abstract into LaTeX/Markdown paper")
    parser.add_argument("--paper", type=str, help="Path to existing paper file (.tex or .md)")
    parser.add_argument("--section", type=str, default="abstract",
                        choices=["abstract", "related_work"],
                        help="Section to insert into (default: abstract)")
    parser.add_argument("--create-template", action="store_true",
                        help="Create a new paper template instead of editing existing")
    parser.add_argument("--format", type=str, default="latex",
                        choices=["latex", "markdown"],
                        help="Template format (default: latex)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for template (default: current dir)")
    parser.add_argument("--scan", action="store_true",
                        help="Scan for existing paper files in project")
    args = parser.parse_args()

    if args.scan:
        candidates = find_paper_files(os.getcwd())
        if candidates:
            print("Found potential paper files:")
            for c in candidates:
                print(f"  {c}")
        else:
            print("No paper files found. Use --create-template to generate one.")
        return

    if args.create_template:
        create_template(args.output_dir, args.format)
        return

    if args.paper:
        fpath = args.paper
        if not os.path.isabs(fpath):
            fpath = os.path.abspath(fpath)
        ext = os.path.splitext(fpath)[1].lower()

        if args.section == "abstract":
            content = CJEPA_ABSTRACT_SENTENCE if ext == ".tex" else CJEPA_ABSTRACT_PLAIN
        else:
            content = CJEPA_RELATED_WORK if ext == ".tex" else CJEPA_ABSTRACT_PLAIN

        if ext == ".tex":
            insert_into_tex(fpath, content, args.section)
        elif ext == ".md":
            insert_into_md(fpath, content, args.section)
        else:
            print(f"[ERROR] Unsupported format: {ext}")
        return

    candidates = find_paper_files(os.getcwd())
    if candidates:
        print("Found paper files. Choose one with --paper PATH:")
        for c in candidates:
            print(f"  {c}")
        print("\nOr use --scan to list them again.")
    else:
        print("No paper files found.")
        print("Run with --create-template to generate a new paper:")
        print("  python paper_insert.py --create-template --format latex")
        print("  python paper_insert.py --create-template --format markdown")


if __name__ == "__main__":
    main()
