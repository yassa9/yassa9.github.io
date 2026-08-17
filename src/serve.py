#!/usr/bin/env python3
import os, re, shutil, http.server, socketserver, sys
import markdown

SITE, PORT = "docs", 4000
BUILD_ONLY = "--build" in sys.argv[1:]
for a in sys.argv[1:]:
    if a != "--build":
        PORT = int(a)
SRC = "src"
CONTENT = "content"

def strip_fm(t):
    return t.split("---", 2)[2].strip() if t.startswith("---") else t

def parse_fm(t):
    fm = {}
    if t.startswith("---"):
        for l in t.split("---", 2)[1].strip().split("\n"):
            if ":" in l:
                k, v = l.split(":", 1)
                fm[k.strip()] = v.strip().strip('"').strip("'")
    return fm

def protect_math(t):
    stash = []
    def grab(m):
        stash.append(m.group(0))
        return f"@@MATH{len(stash)-1}@@"
    t = re.sub(r"\$\$.*?\$\$", grab, t, flags=re.DOTALL)
    t = re.sub(r"\$[^$\n]+?\$", grab, t)
    return t, stash

def restore_math(h, stash):
    for i, m in enumerate(stash):
        h = h.replace(f"@@MATH{i}@@", m)
    return h

def resolve(tpl, vars):
    tpl = re.sub(r"{% if (.+?) %}(.*?){% endif %}",
                 lambda m: (lambda p: p[0] if vars.get(m.group(1).split()[0], "") else (p[1] if len(p) > 1 else ""))
                 (re.split(r"{% else %}", m.group(2), 1)), tpl, flags=re.DOTALL)
    tpl = re.sub(r"{% for [^%]+ %}(.*?){% endfor %}", r"\1", tpl, flags=re.DOTALL)
    for k, v in vars.items():
        tpl = tpl.replace("{{ " + k + " }}", str(v))
    return tpl

def build():
    if not BUILD_ONLY and os.path.exists(SITE):
        shutil.rmtree(SITE)
    os.makedirs(SITE, exist_ok=True)

    if os.path.isdir(f"{SRC}/assets"):
        shutil.copytree(f"{SRC}/assets", f"{SITE}/assets")
    if os.path.isdir(f"{CONTENT}/images"):
        shutil.copytree(f"{CONTENT}/images", f"{SITE}/images")

    dt = strip_fm(open(f"{SRC}/layouts/default.html").read())
    sc = {"site.title": "yassa9", "site.description": "just a blog"}

    for d in ("main", "pages"):
        pdir = f"{CONTENT}/{d}"
        if not os.path.isdir(pdir):
            continue
        for fname in os.listdir(pdir):
            assets_dir = None
            if fname.endswith(".md"):
                mdpath = f"{pdir}/{fname}"
            elif os.path.isdir(f"{pdir}/{fname}") and os.path.isfile(f"{pdir}/{fname}/index.md"):
                mdpath = f"{pdir}/{fname}/index.md"
                assets_dir = f"{pdir}/{fname}"
            else:
                continue
            md = open(mdpath).read()
            fm = parse_fm(md)
            if "permalink" not in fm:
                continue
            title = fm.get("title", "")
            body, mstash = protect_math(strip_fm(md))
            h = markdown.markdown(body, extensions=['extra', 'codehilite'], extension_configs={'codehilite': {'guess_lang': False}})
            h = restore_math(h, mstash)
            h = re.sub(r'<a\s', '<a target="_blank" rel="noopener" ', h)
            pg = resolve(dt, {**sc, "content": h, "page.title": title, "page.description": ""})
            p = fm["permalink"].strip("/")
            out = f"{SITE}/{p}" if p else SITE
            os.makedirs(out, exist_ok=True)
            open(f"{out}/index.html", "w").write(pg)
            if assets_dir:
                for afname in os.listdir(assets_dir):
                    if afname != "index.md":
                        shutil.copy(f"{assets_dir}/{afname}", f"{out}/{afname}")

    print(f"✓ Built → {SITE}/")

build()
if BUILD_ONLY:
    sys.exit(0)
os.chdir(SITE)
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
    print(f"  http://localhost:{PORT}")
    httpd.serve_forever()
