"""Compile paper/main.tex to paper/main.pdf on Modal (CPU only, TeX Live from Debian).

Usage: modal run engine/paper/build_pdf.py
"""
from pathlib import Path

import modal

HERE = Path(__file__).parent
image = (
    modal.Image.debian_slim()
    .apt_install("texlive-latex-base", "texlive-latex-recommended", "texlive-latex-extra", "texlive-fonts-recommended",
                 "poppler-utils")
    .add_local_file(HERE / "main.tex", "/paper/main.tex")
)
app = modal.App("paper-pdf")


@app.function(image=image, cpu=2, timeout=600)
def build() -> tuple[bytes, str, list]:
    import subprocess

    log = ""
    for _ in range(2):  # twice, so references resolve
        p = subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd="/paper", capture_output=True, text=True)
        log = p.stdout[-3000:]
    pdf = Path("/paper/main.pdf")
    pages = []
    if pdf.exists():
        # Page previews, to check layout without a local PDF renderer.
        subprocess.run(["pdftoppm", "-r", "60", "-png", "main.pdf", "page"], cwd="/paper")
        pages = [(f.name, f.read_bytes()) for f in sorted(Path("/paper").glob("page*.png"))]
    return (pdf.read_bytes() if pdf.exists() else b""), log, pages


@app.local_entrypoint()
def main():
    pdf, log, pages = build.remote()
    if not pdf:
        print(log)
        raise SystemExit("no PDF produced")
    (HERE / "main.pdf").write_bytes(pdf)
    preview = HERE / "preview"
    preview.mkdir(exist_ok=True)
    for name, data in pages:
        (preview / name).write_bytes(data)
    warnings = [l for l in log.splitlines() if "Warning" in l or "Overfull" in l]
    print(f"wrote {HERE / 'main.pdf'} ({len(pdf)} bytes)")
    print("\n".join(warnings[:15]))
