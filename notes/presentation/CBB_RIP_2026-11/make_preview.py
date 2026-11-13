"""Render CBB_RIP_talk_v1.pptx as an HTML preview (no LibreOffice on this node): each slide is a 16:9 board with
its shapes placed at their pptx positions (text, pictures, tables, grouped shapes), speaker notes alongside.
Writes preview/index.html + preview/img/*.  Usage: python make_preview.py"""
import os, json, html, hashlib
from pptx import Presentation
from pptx.util import Pt
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
DECK = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "CBB_RIP_talk_v2.pptx")
OUT = os.path.join(HERE, "preview"); IMG = os.path.join(OUT, "img"); os.makedirs(IMG, exist_ok=True)

prs = Presentation(DECK)
SW, SH = prs.slide_width, prs.slide_height
PT_PER_W = 100.0 / (SW / 12700.0)      # % of board width per point (SW in EMU; 12700 EMU = 1 pt)

def pct(v, total): return f"{100.0 * v / total:.3f}%"

def save_blob(blob, ext):
    name = hashlib.md5(blob).hexdigest()[:12] + "." + ext
    p = os.path.join(IMG, name)
    if not os.path.exists(p):
        with open(p, "wb") as f: f.write(blob)
    return "img/" + name

def run_size(run, para, shape):
    if run.font.size: return run.font.size.pt
    try:
        if para.font.size: return para.font.size.pt
    except Exception: pass
    # placeholder inherits from layout
    try:
        if shape.is_placeholder:
            lp = shape.placeholder_format.idx
            for ph in shape.part.slide_layout.placeholders if hasattr(shape.part, "slide_layout") else []:
                pass
    except Exception: pass
    return 18.0 if not (shape.is_placeholder and shape.placeholder_format.idx == 0) else 28.0

def text_html(shape):
    tf = shape.text_frame
    out = []
    for para in tf.paragraphs:
        lvl = para.level or 0
        align = {PP_ALIGN.CENTER: "center", PP_ALIGN.RIGHT: "right"}.get(para.alignment, "left")
        runs = []
        for run in para.runs:
            t = html.escape(run.text)
            if not t: continue
            sz = run_size(run, para, shape)
            st = f"font-size:calc({sz:.1f} * var(--pt))"
            if run.font.bold: st += ";font-weight:700"
            if run.font.italic: st += ";font-style:italic"
            try:
                if run.font.color and run.font.color.type is not None and run.font.color.rgb is not None:
                    st += f";color:#{run.font.color.rgb}"
            except Exception: pass
            runs.append(f"<span style=\"{st}\">{t}</span>")
        bullet = ""
        try:
            pPr = para._p.pPr
            if pPr is not None and pPr.find("{http://schemas.openxmlformats.org/drawingml/2006/main}buChar") is not None:
                bullet = "• "
        except Exception: pass
        if runs or bullet:
            out.append(f"<p style=\"text-align:{align};padding-left:{lvl * 1.2}em\">{bullet}{''.join(runs)}</p>")
        else:
            out.append("<p>&nbsp;</p>")
    return "".join(out)

def fill_css(shape):
    try:
        f = shape.fill
        if f.type == 1:  # solid
            rgb = f.fore_color.rgb
            return f"background:#{rgb};"
    except Exception: pass
    return ""

def emit(shape, off=(0, 0), scale=(1.0, 1.0), parts=None):
    """Append HTML for shape (absolute coords in EMU after group transform)."""
    l = off[0] + (shape.left or 0) * scale[0]; t = off[1] + (shape.top or 0) * scale[1]
    w = (shape.width or 0) * scale[0]; h = (shape.height or 0) * scale[1]
    pos = f"left:{pct(l, SW)};top:{pct(t, SH)};width:{pct(w, SW)};height:{pct(h, SH)};"
    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        xfrm = shape._element.xpath("./p:grpSpPr/a:xfrm")
        if xfrm:
            x = xfrm[0]
            ch_off = x.find("{http://schemas.openxmlformats.org/drawingml/2006/main}chOff")
            ch_ext = x.find("{http://schemas.openxmlformats.org/drawingml/2006/main}chExt")
            cox, coy = int(ch_off.get("x")), int(ch_off.get("y"))
            cex, cey = int(ch_ext.get("cx")), int(ch_ext.get("cy"))
            sx = w / cex if cex else 1.0; sy = h / cey if cey else 1.0
            for sub in shape.shapes:
                emit(sub, off=(l - cox * sx, t - coy * sy), scale=(sx, sy), parts=parts)
        return
    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
        try:
            src = save_blob(shape.image.blob, shape.image.ext)
            parts.append(f"<img class=\"pic\" style=\"{pos}\" src=\"{src}\" alt=\"{html.escape(shape.name)}\">")
        except Exception as e:
            parts.append(f"<div class=\"box\" style=\"{pos}\">[image: {html.escape(shape.name)}]</div>")
        return
    if shape.has_table if hasattr(shape, "has_table") else False:
        rows = []
        for r_i, row in enumerate(shape.table.rows):
            cells = "".join(f"<{'th' if r_i == 0 else 'td'}>{html.escape(c.text)}</{'th' if r_i == 0 else 'td'}>" for c in row.cells)
            rows.append(f"<tr>{cells}</tr>")
        parts.append(f"<div class=\"tbl\" style=\"{pos}\"><table>{''.join(rows)}</table></div>")
        return
    if shape.has_text_frame:
        inner = text_html(shape)
        fc = fill_css(shape)
        if inner.strip() or fc:
            cls = "txt" + (" title" if shape.is_placeholder and shape.placeholder_format.idx == 0 else "")
            parts.append(f"<div class=\"{cls}\" style=\"{pos}{fc}\">{inner}</div>")
        return
    fc = fill_css(shape)
    if fc:
        parts.append(f"<div class=\"box\" style=\"{pos}{fc}\"></div>")

slides_html = []
for i, slide in enumerate(prs.slides, 1):
    parts = []
    for sh in slide.shapes:
        emit(sh, parts=parts)
    title = ""
    for sh in slide.shapes:
        if sh.is_placeholder and sh.placeholder_format.idx == 0 and sh.has_text_frame:
            title = sh.text_frame.text.strip(); break
    if not title:
        for sh in slide.shapes:
            if sh.has_text_frame and sh.text_frame.text.strip():
                title = sh.text_frame.text.strip().split("\n")[0]; break
    notes = slide.notes_slide.notes_text_frame.text if slide.has_notes_slide else ""
    tag = ""
    for sh in slide.shapes:
        if sh.is_placeholder and sh.placeholder_format.idx == 13 and sh.has_text_frame:
            tag = sh.text_frame.text.strip()
    kind = "new" if tag.startswith("NEW") else ("reused" if tag.startswith("reused") else "")
    chip = f"<span class=\"chip {kind}\">{html.escape(tag)}</span>" if tag else ""
    notes_html = "".join(f"<p>{html.escape(p)}</p>" for p in notes.split("\n") if p.strip())
    slides_html.append(f"""
<section class="slide" id="s{i}">
  <header><span class="num">{i:02d}</span><h2>{html.escape(title)}</h2>{chip}</header>
  <div class="row">
    <div class="board">{''.join(parts)}</div>
    <aside class="notes"><h3>Speaker notes</h3>{notes_html or '<p class="muted">none</p>'}</aside>
  </div>
</section>""")

nav = "".join(f"<a href=\"#s{i}\">{i}</a>" for i in range(1, len(prs.slides) + 1))
page = f"""<title>CBB RIP Deck Preview</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap">
<style>
:root{{--bg:#eef0ed;--board:#ffffff;--ink:#1b1d21;--muted:#6b7075;--line:#d6d9d3;--accent:#2f6f73;--new:#b4541a;--reuse:#2f6f73;--panel:#f7f8f6;--pt:0.1042cqw}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{color-scheme:dark;--bg:#15171a;--board:#ffffff;--ink:#e6e8e3;--muted:#9a9ea3;--line:#2c3035;--panel:#1e2125}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#15171a;--board:#ffffff;--ink:#e6e8e3;--muted:#9a9ea3;--line:#2c3035;--panel:#1e2125}}
body{{background:var(--bg);color:var(--ink);font-family:"Source Sans 3",system-ui,Arial,sans-serif;padding-inline:16px;padding-block:0 48px;margin:0}}
.wrap{{max-width:1480px;margin:0 auto}}
.top{{position:sticky;top:env(safe-area-inset-top,0px);background:var(--bg);padding-block:14px 10px;display:flex;flex-wrap:wrap;gap:8px 16px;align-items:baseline;border-bottom:1px solid var(--line);z-index:2}}
.top h1{{font-size:1.15rem;margin:0;font-weight:700}} .top .sub{{color:var(--muted);font-size:.9rem}}
.top nav{{display:flex;flex-wrap:wrap;gap:4px;margin-left:auto}} .top nav a{{color:var(--accent);text-decoration:none;font-size:.8rem;padding:2px 6px;border:1px solid var(--line);border-radius:4px}}
.top nav a:focus-visible{{outline:2px solid var(--accent)}}
.slide{{padding-block:28px 8px;border-bottom:1px solid var(--line)}}
.slide header{{display:flex;flex-wrap:wrap;gap:6px 12px;align-items:baseline;margin-bottom:10px}}
.slide .num{{font-variant-numeric:tabular-nums;color:var(--muted);font-weight:600}} .slide h2{{font-size:1.05rem;margin:0;font-weight:600;text-wrap:balance}}
.chip{{font-size:.72rem;letter-spacing:.04em;text-transform:uppercase;padding:2px 8px;border-radius:999px;border:1px solid var(--line);color:var(--muted)}}
.chip.new{{border-color:var(--new);color:var(--new)}} .chip.reused{{border-color:var(--reuse);color:var(--reuse)}}
.row{{display:grid;grid-template-columns:minmax(0,2fr) minmax(260px,1fr);gap:18px;align-items:start}}
@media (max-width:900px){{.row{{grid-template-columns:1fr}}}}
.board{{position:relative;aspect-ratio:16/9;max-width:100%;background:var(--board);color:#1b1d21;overflow:hidden;container-type:inline-size;box-shadow:0 1px 3px rgba(0,0,0,.18);font-family:Calibri,Arial,Helvetica,sans-serif;line-height:1.2}}
.board .txt,.board .box,.board .pic,.board .tbl{{position:absolute;box-sizing:border-box}}
.board .txt{{padding:0.3cqw 0.6cqw;overflow:hidden}} .board .txt p{{margin:0}} .board .title{{font-weight:700}}
.board .pic{{object-fit:contain}} .board .box{{font-size:calc(12 * var(--pt));color:#777}}
.board .tbl table{{border-collapse:collapse;width:100%;height:100%;font-size:calc(13 * var(--pt))}} .board .tbl th,.board .tbl td{{border:1px solid #c9ccc6;padding:.3cqw .6cqw;text-align:left}} .board .tbl th{{background:#e8ebe6;font-weight:700}}
.notes{{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px 14px;font-size:.92rem;line-height:1.45}}
.notes h3{{margin:0 0 6px;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600}} .notes p{{margin:0 0 8px}} .muted{{color:var(--muted)}}
</style>
<div class="wrap">
<div class="top"><h1>CBB research-in-progress talk, draft v1</h1><span class="sub">{len(prs.slides)} slides · reconstructed from CBB_RIP_talk_v1.pptx (text, images and positions; theme colors approximate)</span><nav>{nav}</nav></div>
{''.join(slides_html)}
</div>
"""
with open(os.path.join(OUT, "index.html"), "w") as f: f.write(page)
files = sorted(os.listdir(IMG))
print(json.dumps({"index": "index.html", "images": ["img/" + x for x in files]}))
