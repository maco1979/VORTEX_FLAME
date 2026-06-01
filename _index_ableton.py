import os
import sys
import json
import re

sys.path.insert(0, r"D:\VORTEX_FLAME")
from soul_memory import write

ABLETON_ROOT = r"C:\ProgramData\Ableton\Live 12 Suite"

def extract_ableton_knowledge():
    knowledge_items = []

    knowledge_items.append({
        "topic": "Ableton Live 12 Suite - OSC Control Protocol",
        "source": "ableton_adapter",
        "type": "software_control",
        "software": "Ableton Live 12 Suite",
        "control_method": "OSC",
        "protocol": {
            "default_port": 11000,
            "message_format": "/live/{command}",
            "commands": [
                "/live/play", "/live/stop", "/live/tempo",
                "/live/track/create", "/live/track/volume",
                "/live/clip/fire", "/live/clip/stop",
                "/live/device/parameter", "/live/scene/fire",
            ],
        },
        "key_shortcuts": {
            "play": "Space",
            "record": "F9",
            "new_midi_clip": "Ctrl+Shift+M",
            "new_audio_clip": "Ctrl+Shift+A",
            "quantize": "Ctrl+U",
            "duplicate": "Ctrl+D",
            "split": "Ctrl+E",
            "consolidate": "Ctrl+J",
            "automation": "A",
            "draw_mode": "B",
        },
    })

    knowledge_items.append({
        "topic": "Ableton Live 12 Suite - Core Workflow",
        "source": "ableton_adapter",
        "type": "software_workflow",
        "software": "Ableton Live 12 Suite",
        "workflows": [
            "Session View: clip-based live performance, scene triggering",
            "Arrangement View: timeline-based production, linear editing",
            "MIDI: record, draw, quantize, velocity editing, note expressions",
            "Audio: warp, slice, comping, crossfade editing",
            "Mixing: volume, pan, sends, device chains, groups",
            "Effects: audio effects rack, MIDI effects, max for live",
            "Export: render to audio, freeze, flatten",
        ],
        "file_formats": {
            "project": ".als",
            "preset": ".adg / .adv",
            "clip": ".alc",
            "sample": ".wav / .aiff / .flac",
        },
    })

    knowledge_items.append({
        "topic": "Ableton Live 12 Suite - Mano-P GUI Operations",
        "source": "mano_p_adapter",
        "type": "gui_automation",
        "software": "Ableton Live 12 Suite",
        "gui_operations": [
            "Screenshot → identify track lanes, mixer, browser",
            "Click track header → select track",
            "Double-click clip slot → create/edit MIDI clip",
            "Drag device from browser → add to track chain",
            "Click automation lane → toggle automation editing",
            "Right-click context menu → quantize, consolidate, etc.",
        ],
        "screen_regions": {
            "browser": "left panel",
            "session": "center (clip grid)",
            "detail": "bottom (clip/device editor)",
            "mixer": "right panel (session view)",
            "transport": "top bar (play/stop/tempo)",
        },
    })

    return knowledge_items

SKIP_SUBDIRS = {"Max", "cache", "Crash", "Log", "tmp", "__pycache__"}

def scan_ableton_docs():
    doc_items = []
    if not os.path.exists(ABLETON_ROOT):
        print(f"  Ableton root not found: {ABLETON_ROOT}")
        return doc_items

    for root, dirs, files in os.walk(ABLETON_ROOT):
        rel_root = os.path.relpath(root, ABLETON_ROOT)
        skip = False
        for sd in SKIP_SUBDIRS:
            if sd in rel_root.split(os.sep):
                skip = True
                break
        if skip:
            dirs.clear()
            continue
        for f in files:
            if f.endswith(('.txt', '.md', '.html', '.xml', '.json', '.py')):
                if f.endswith('.json') and os.path.getsize(os.path.join(root, f)) < 500:
                    continue
                fp = os.path.join(root, f)
                try:
                    with open(fp, 'r', encoding='utf-8', errors='ignore') as fh:
                        content = fh.read(2000)
                    if len(content) > 100:
                        rel = os.path.relpath(fp, ABLETON_ROOT)
                        doc_items.append({
                            "topic": f"Ableton Live 12 - {rel}",
                            "source": "local_install",
                            "type": "software_documentation",
                            "software": "Ableton Live 12 Suite",
                            "content_preview": content[:1500],
                            "file_path": fp,
                        })
                except:
                    pass

    return doc_items

def main():
    print("=" * 60)
    print("Indexing Ableton Live 12 → beethoven knowledge base")
    print("=" * 60)

    knowledge_items = extract_ableton_knowledge()
    doc_items = scan_ableton_docs()
    all_items = knowledge_items + doc_items

    print(f"\n  Built-in knowledge: {len(knowledge_items)} items")
    print(f"  Local docs scanned: {len(doc_items)} items")
    print(f"  Total to index: {len(all_items)} items")

    indexed = 0
    errors = 0
    for item in all_items:
        try:
            write(
                soul="beethoven",
                category="domain_memory",
                content=item,
                importance=0.8,
                tags=["ableton", "music_production", "daw", item.get("type", "general")],
            )
            indexed += 1
            print(f"  ✓ {item['topic'][:60]}")
        except Exception as e:
            errors += 1
            print(f"  ✗ {item['topic'][:60]}: {e}")

    print(f"\n  Result: {indexed} indexed, {errors} errors")

if __name__ == "__main__":
    main()
