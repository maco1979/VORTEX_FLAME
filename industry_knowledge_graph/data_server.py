#!/usr/bin/env python3
import json, os, glob, time, sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

BASE = r"D:\VORTEX_FLAME"
SOULS = ["cezanne","einstein","galileo","darwin","davinci","strategy","humboldt","yuanlongping","herodotus","guizhu","montesquieu","beethoven","monet","vangogh"]

def scan_dir(path, depth=0, max_depth=2):
    if depth > max_depth:
        return None
    try:
        items = []
        for entry in os.scandir(path):
            if entry.name.startswith('.') or entry.name == 'node_modules' or entry.name == '__pycache__':
                continue
            if entry.is_dir():
                children = scan_dir(entry.path, depth + 1, max_depth)
                count = 0
                if children:
                    count = len(children)
                items.append({"name": entry.name, "type": "dir", "path": entry.path, "children": children, "count": count})
            else:
                try:
                    size = entry.stat().st_size
                except:
                    size = 0
                items.append({"name": entry.name, "type": "file", "path": entry.path, "size": size})
        return items
    except PermissionError:
        return None

def get_training_status():
    status = {}
    for soul in SOULS:
        soul_dir = os.path.join(BASE, "soul_lora_v2", soul)
        if not os.path.exists(soul_dir):
            status[soul] = {"exists": False}
            continue
        iters = sorted(glob.glob(os.path.join(soul_dir, "selfplay_iter*")))
        stages = sorted(glob.glob(os.path.join(soul_dir, "stage*")))
        final_weights = glob.glob(os.path.join(soul_dir, "**/adapter_model.safetensors"), recursive=True)
        total_size = sum(os.path.getsize(f) for f in final_weights if os.path.exists(f))
        status[soul] = {
            "exists": True,
            "iters": len(iters),
            "stages": len(stages),
            "weights": len(final_weights),
            "total_size_mb": round(total_size / 1024 / 1024, 1),
            "latest": os.path.basename(iters[-1]) if iters else None
        }
    return status

def get_knowledge_stats():
    stats = {}
    for soul in SOULS:
        kb_dir = os.path.join(BASE, "long-memory", soul, "knowledge")
        if not os.path.exists(kb_dir):
            stats[soul] = {"exists": False}
            continue
        data_file = os.path.join(kb_dir, "data.jsonl")
        count = 0
        if os.path.exists(data_file):
            count = sum(1 for _ in open(data_file, 'r', encoding='utf-8'))
        faiss_file = os.path.join(kb_dir, "faiss_index", "index.faiss")
        faiss_size = os.path.getsize(faiss_file) / 1024 / 1024 if os.path.exists(faiss_file) else 0
        stats[soul] = {
            "exists": True,
            "entries": count,
            "faiss_size_mb": round(faiss_size, 1)
        }
    return stats

def get_project_files():
    top_dirs = []
    try:
        for entry in os.scandir(BASE):
            if entry.name.startswith('.') or entry.name == 'node_modules':
                continue
            if entry.is_dir():
                children = scan_dir(entry.path, 0, 1)
                count = len(children) if children else 0
                top_dirs.append({"name": entry.name, "type": "dir", "path": entry.path, "count": count})
            else:
                try:
                    size = entry.stat().st_size
                except:
                    size = 0
                top_dirs.append({"name": entry.name, "type": "file", "path": entry.path, "size": size})
    except:
        pass
    return top_dirs

def get_disk_usage():
    result = {}
    for label, path in [("D_VORTEX_FLAME", BASE), ("C_Desktop", r"C:\Users\42235\Desktop")]:
        try:
            total = 0
            for dirpath, dirnames, filenames in os.walk(path):
                dirnames[:] = [d for d in dirnames if d not in ('.git', 'node_modules', '__pycache__', '.gitnexus')]
                for f in filenames:
                    try:
                        total += os.path.getsize(os.path.join(dirpath, f))
                    except:
                        pass
            result[label] = {"path": path, "size_gb": round(total / 1024 / 1024 / 1024, 2)}
        except:
            result[label] = {"path": path, "size_gb": 0}
    return result

def get_gitnexus_data():
    try:
        import urllib.request
        r = urllib.request.urlopen("http://172.20.43.89:4748/api/repos", timeout=3)
        repos = json.loads(r.read())
        if repos:
            repo = repos[0]["name"]
            r2 = urllib.request.urlopen(f"http://172.20.43.89:4748/api/graph?repo={repo}", timeout=10)
            return json.loads(r2.read())
    except:
        pass
    return None

ROUTES = {}

def route(path):
    def decorator(func):
        ROUTES[path] = func
        return func
    return decorator

@route("/api/status")
def api_status(params):
    return {"status": "ok", "time": time.strftime("%Y-%m-%d %H:%M:%S"), "base": BASE}

@route("/api/training")
def api_training(params):
    return get_training_status()

@route("/api/knowledge")
def api_knowledge(params):
    return get_knowledge_stats()

@route("/api/project")
def api_project(params):
    return get_project_files()

@route("/api/disk")
def api_disk(params):
    return get_disk_usage()

@route("/api/gitnexus")
def api_gitnexus(params):
    data = get_gitnexus_data()
    if data:
        return data
    return {"error": "GitNexus not available", "hint": "Start gitnexus serve in WSL2 on port 4748"}

@route("/api/all")
def api_all(params):
    return {
        "training": get_training_status(),
        "knowledge": get_knowledge_stats(),
        "project": get_project_files(),
        "disk": get_disk_usage()
    }

SERVE_DIR = os.path.join(BASE, "industry_knowledge_graph")
MIME_TYPES = {'.html':'text/html;charset=utf-8','.css':'text/css','.js':'application/javascript','.json':'application/json','.png':'image/png','.jpg':'image/jpeg','.svg':'image/svg+xml','.ico':'image/x-icon'}

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)
        if path in ROUTES:
            try:
                data = ROUTES[path](params)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            if path == "/":
                path = "/gitnexus_viewer.html"
            file_path = os.path.normpath(os.path.join(SERVE_DIR, path.lstrip("/")))
            if not file_path.startswith(SERVE_DIR):
                self.send_error(403)
                return
            if os.path.isfile(file_path):
                ext = os.path.splitext(file_path)[1]
                mime = MIME_TYPES.get(ext, "application/octet-stream")
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)

    def log_message(self, format, *args):
        pass

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"VORTEX_FLAME Data Server running on http://localhost:{port}")
    print(f"API endpoints: /api/status /api/training /api/knowledge /api/project /api/disk /api/gitnexus /api/all")
    print(f"Web UI: http://localhost:{port}/gitnexus_viewer.html")
    print(f"Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
