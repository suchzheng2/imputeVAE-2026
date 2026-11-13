"""Build the CBB research-in-progress talk deck, v2 (2026-10-28).

Base: the qualifying-exam deck (July 2026; same theme as the follow-up deck). Slides are reused from the qual deck
in place, imported from the follow-up deck (shape tree cloned, pictures re-embedded), or inserted as a pre-qual
PDF page image. New slides are placeholders: main message (title), main visual spec (body), NEW tag, speaker notes.
Backup slides follow the Thank-you slide.

Usage (impute2 env; PyMuPDF under the scratchpad pylib):
    PYTHONPATH=<pylib> python notes/presentation/CBB_RIP_2026-11/build_deck_v2.py
"""
import copy, io, os
from pptx import Presentation
from pptx.util import Inches, Pt

UP = "/home/sz477/.claude/uploads/86a8a919-16cc-5a84-8c48-e3eb3208a069"
QUAL = f"{UP}/418c1c2b-Qualifying_slide_Suchen_progress.pptx"
FOLLOWUP = f"{UP}/bd939522-Qualifying_followup-meeting-slide_Suchen-v1_Autosaved.pptx"
PREQUAL = f"{UP}/b830e303-ImputeVAE_PreQual_Slides.pdf"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(OUT_DIR, "CBB_RIP_talk_v2.pptx")
IMG_DIR = os.path.join(OUT_DIR, "prequal_pages"); os.makedirs(IMG_DIR, exist_ok=True)

import fitz
pdf = fitz.open(PREQUAL)
def page_png(n):
    p = os.path.join(IMG_DIR, f"prequal_p{n:02d}.png")
    if not os.path.exists(p): pdf[n - 1].get_pixmap(dpi=170).save(p)
    return p

NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

# ------------------------------------------------------------------ plan
PLAN = [
 dict(kind="qual", src=1, title="ImputeVAE: what a nucleus can tell you about its cytoplasm",
      notes="Hi everyone. I work on single-nucleus RNA-seq, and the question behind my project is simple: a nucleus only holds part of a cell's RNA. "
            "How much of the rest can we recover, and what does it change? I will show the biology we learned, the model we use as the instrument, "
            "and then one open question I want your opinion on."),
 dict(kind="qual", src=3, tag="reused: qual slide 3 (Jul 2026)",
      notes="Why this problem exists. Most brain tissue is frozen or postmortem; cells fall apart on dissociation and only nuclei survive. "
            "So the large human resources, PsychENCODE, SEA-AD, ROSMAP, are nuclei, and a nucleus is only part of the cell."),
 dict(kind="qual", src=4, tag="reused: qual slide 4 (Jul 2026)",
      notes="A nucleus is not a shrunk-down cell. What is missing is structured in three ways: gene-specific (ribosomal, cytoskeletal, synaptic-translation mRNAs are depleted), "
            "directional and large (nuclear RNA is 20 to 50 percent of the cell's mRNA), and cell-type dependent (neurons ship more RNA out to dendrites than glia)."),
 dict(kind="qual", src=5, tag="reused: qual slide 5 (Jul 2026)",
      notes="And it follows known rules of RNA localization. Some RNAs are deliberately held in the nucleus, Malat1 and Neat1 never leave; "
            "in neurons about 2,500 transcripts are shipped into the dendrites, like Camk2a here. Those are the two directions the model has to learn, "
            "and there are public compartment maps I can check it against."),
 dict(kind="qual", src=15, tag="reused: qual slide 15 (Jul 2026)",
      notes="The instrument. A conditional VAE: cells and nuclei go through a shared encoder into one latent space, and a modality flag tells the decoder which modality to produce. "
            "Batch correction would treat the nucleus-versus-cell difference as noise and remove it; here it is the thing being learned."),
 dict(kind="qual", src=16, tag="reused: qual slide 16 (Jul 2026); caption to update: single-stage training, library-size covariate, no labels at inference",
      notes="At inference we encode a nucleus and flip the flag to decode as a cell. Two changes since this was drawn: training is single-stage, "
            "and the decoder takes an explicit library-size covariate, so the output RNA content is a parameter rather than something the latent has to carry. "
            "No cell-type label is needed at inference; the type is read from the nucleus itself."),
 dict(kind="qual", src=28, retitle="What the model adds back is ribosomal and cytoplasmic-translation machinery", tag="reused: qual slide 28 (Jul 2026)",
      notes="First result, the biology. The genes the model adds back are overwhelmingly ribosomal and cytoplasmic-translation machinery, the same signal in four ontologies. "
            "That is exactly the program a nuclear prep loses."),
 dict(kind="fu", src=10, tag="reused: follow-up slide 10 (Aug 2026)",
      notes="And the magnitude is calibrated. For every gene and cell type I compare the empirical cell-minus-nucleus delta with the predicted one. "
            "On the synaptic genes from SynGO the slope is one and the correlation 0.94."),
 dict(kind="fu", src=12, tag="reused: follow-up slide 12 (Aug 2026)",
      notes="It is specific. Transcription factor genes from JASPAR, a nuclear-associated contrast set, show a much smaller neuron-versus-glia contrast than synaptic genes. "
            "The model is not inflating everything in neurons."),
 dict(kind="qual", src=27, retitle="Genome-wide check against an independent localization database", tag="reused: qual slide 27 (Jul 2026)",
      notes="Same result genome-wide against a database the model never saw: cytoplasm-annotated genes get larger predicted shifts than nucleus-annotated genes, "
            "and the difference is highly significant."),
 dict(kind="new", title="Nuclear-retained RNAs move the other way: Malat1, Meg3 and the 45-gene list",
      visual="MAIN VISUAL: horizontal bar chart of predicted vs empirical delta for ~10 showcase genes: nuclear lncRNAs (Malat1, Meg3, Neat1, Gm*) negative, "
             "ribosomal / cytoplasmic genes (Rpl*, Rps*, Fos, Arc) positive. Inset: 45 data-driven nuclear-retained genes, 100 % correct sign (mean delta -0.36); ribosomal 100 % positive. "
             "Source: nuclear lncRNA metric (job 7754430), compartment_classification/ (AUROC 0.991, scrambled z = 40).",
      tag="NEW slide to build",
      notes="The committee asked for showcase genes, and this is the restraint check. Nuclear-retained RNAs such as Malat1 and Meg3 must not be pushed up by imputation. "
            "On a data-driven list of 45 nuclear-retained genes the predicted shift has the correct, negative sign for every one, while ribosomal genes go up in every type. "
            "A classifier separating nuclear from cytoplasmic genes on the predicted delta alone reaches an AUROC of 0.99, and a scrambled control collapses to chance."),
 dict(kind="new", title="Non-circular validation: fractionation-seq and ribosome-bound RNA agree with the predicted shift",
      visual="MAIN VISUAL: two panels. Left: scatter of predicted delta vs biochemical nucleus/cytoplasm fractionation log-ratio (Buckley mouse brain rho 0.44; Moutinho human FC rho 0.40). "
             "Right: TRAP/RiboTag per-population bar: correlation of pseudobulk with the ribosome-bound profile, raw nuclei 0.43 -> imputed 0.73, ceiling (real cells) 0.74; holds across 19 populations incl. excitatory + glia. "
             "Source: analysis/fractionation_validation.py (job 8558579); validation/trap_ribotag_external_validation/outputs/report/.",
      tag="NEW slide to build",
      notes="These two validations never touch the training data. Biochemical fractionation separates nuclear and cytoplasmic RNA directly, and the predicted shift correlates with that ratio in mouse brain and in human frontal cortex. "
            "TRAP and RiboTag profile ribosome-bound RNA in genetically defined populations, which William suggested at the qual. Imputed nuclei move from 0.43 to 0.73 correlation with the ribosome-bound profile, the ceiling set by real cells, in 19 populations including glia. "
            "One caveat: a per-type mean shift reaches the same numbers. I will come back to that."),
 dict(kind="new", title="Benchmark: best label-free translator; with labels, simple methods reach the ceiling",
      visual="MAIN VISUAL: native table (filled below). Optional: a one-axis ladder plot of the same numbers. Source: run/benchmark/test/results/ (split-aware benchmark, 3 seeds), notes/benchmark_review_2026-10-09.md §5b. "
             "Do NOT reuse the qual-deck baseline tables (slides 19-26): those are all-data numbers and the scVI/scANVI rows (0.386) are superseded by the split-aware 0.86.",
      table=[["Setting", "Gene corr (test)", "Uses label at inference"],
             ["Cell-type mean of real cells (ceiling)", "0.981", "yes"],
             ["scGen", "0.975", "yes"],
             ["ImputeVAE, predicted type (covariate model)", "0.962", "no"],
             ["ImputeVAE, no labels anywhere", "0.921", "no"],
             ["ImputeVAE, no covariate (ablation)", "0.900", "no"],
             ["scVI / scANVI, label-free", "0.86", "no"]],
      tag="NEW slide to build",
      notes="This is the honest ladder on a held-out split. If you have cell-type labels, the cell-type mean is the ceiling and scGen sits just under it. "
            "Without labels at inference, ImputeVAE reaches 0.96 by reading the type from the nucleus, well above label-free scVI and scANVI. "
            "So the method claim is: the best label-free translator. The claim is not that a 22-million-parameter model beats the type mean. It does not, and I would rather say that than have a reviewer say it."),
 dict(kind="pdf", src=28, title="Known limitation in May: within-cell-type variance collapses", tag="reused: pre-qual slide p28 (May 2026)",
      notes="At the qual this was the open problem: within a cell type, imputed profiles had a fraction of the real variance. "
            "I spent the summer on it, and the answer turned out to be a finding rather than a fix."),
 dict(kind="new", title="What a nucleus can and cannot tell you about its cytoplasm: type identity and RNA content",
      visual="MAIN VISUAL: three panels. (a) Latent usage: KL per latent dim, 4-6 of 32 active; one dim = library size carrying 95 % of imputed within-type variance. "
             "(b) Real within-type variance decomposition: 52 % Poisson / 25 % library size / 23 % biological; imputed variance 82 % library axis. "
             "(c) Latent-width control table: 32 / 8 / 4 dims -> active 4 / 2 / 1, gene corr 0.970 / 0.957 / 0.910, classification 0.97 / 0.94 / 0.56. "
             "Source: notes/experiments/variance_zero_nonzero_fullmodel/, libsize review §6.7, notes/experiments/libsize_covariate/latent{8,4}/.",
      tag="NEW slide to build",
      notes="The collapsed variance was library size passing through the latent. Real within-type variance is half Poisson sampling, a quarter RNA content, and a quarter biology; the imputed variance was almost entirely the RNA-content axis. "
            "Once RNA content is an explicit covariate, the latent needs only two to four dimensions to place 15 of 17 types correctly. "
            "So at 10x depth a nucleus determines its type-level cytoplasm plus its RNA content, and individual-nucleus cytoplasm beyond that is below the sampling ceiling. "
            "This is the information ceiling I listed as a limitation at the qual, now measured. The variance-matching losses I proposed in August were dropped: they would be fitting noise."),
 dict(kind="fu", src=22, tag="reused: follow-up slide 22 (Aug 2026)",
      notes="Zero-shot on human motor cortex nuclei, with mouse-trained weights and one-to-one orthologs: neurons get the larger predicted shift, synaptic genes more than transcription factors, "
            "the same direction as mouse, and the per-class shifts correlate 0.72 with mouse across 17 classes. Plausibility only; there is no paired human truth. "
            "The same core, ribosomal up and MALAT1 down, holds in human kidney (Lake 2023), the conserved-transformation test from the qual."),
 dict(kind="new", title="Generalisation: unseen chemistry and an independent matched hold-out, with the ceiling and the baseline",
      visual="MAIN VISUAL: two panels. Left (10x v2, unseen chemistry): ladder of gene corr, old model 0.59 -> covariate model label-free scale 0.84-0.86 -> predicted type (reference classifier, 84 % acc.) 0.90 -> with label 0.93. "
             "Right (Bakken 2018 GSE123454, VISp SMART-seq, 461 nuclei / 445 cells, 11 matched clusters): predicted vs empirical delta scatter, r 0.58 pooled / 0.73 composition-free, sign agreement 0.94 = split-half ceiling; reliability ceiling 0.92; 'predicted type + training mean shift' baseline 0.56-0.59 drawn as a dashed line. "
             "Note for the talk: the qual dataset table (slide 29) listed Ding 2020 as matched sc/sn; it has nuclei only (4 chemistries + bulk). "
             "Source: libsize review §6.4a; notes/experiments/bakken2018_holdout/README.md; run/external_eval/bakken2018*/.",
      tag="NEW slide to build",
      notes="Two generalisation tests. On 10x v2 data the old model fell to 0.59; with the explicit output scale and a label-free type assignment the covariate model recovers 0.90, against 0.93 with labels. "
            "On the only public matched mouse cortex set outside Allen 10x, Bakken 2018 SMART-seq, the direction of the shift is predicted at the split-half ceiling and the magnitude at 0.58 against a reliability ceiling of 0.92. "
            "But the dashed line matters: predicting the type and adding that type's mean training shift scores the same. On external data the model's contribution is the label-free, chemistry-robust type assignment, not extra per-gene accuracy."),
 dict(kind="new", title="Open question: which nuclei-only result would change if we applied this?",
      visual="MAIN VISUAL: left, one sentence in large type: 'Every external test is matched by predicted type + that type's mean nucleus-to-cell shift.' "
             "Right, three candidate applications as cards: (a) cytoplasmic / activity gene recovery in an independent snRNA-seq set; (b) disease DE on imputed profiles (SCORCH human snRNA-seq, 655k nuclei, HIV vs control, PFC / insula / striatum, on the group share); "
             "(c) marker-based annotation of nuclei improves after imputation, on external data. Footer: committee asks status: TRAP/RiboTag done; nuclear lncRNA showcase done; subcellular spatial (MERFISH) not started.",
      tag="NEW slide to build",
      notes="Here is where I want your input. The compartment map is validated, label-free and chemistry-robust, but on every external test a type mean plus a mean shift does as well. "
            "So the contribution has to be an application: a nuclei-only result that changes once you impute. "
            "Three candidates: recovering activity and cytoplasmic genes in an independent nuclei set; re-running a disease DE study on imputed profiles, where we have a 655-thousand-nucleus human set on the group share; or showing annotation of nuclei gets better after imputation on external data. "
            "Which of these would you find convincing, and is there one I am missing?"),
 dict(kind="qual", src=33, notes="Thank you. Questions, and especially opinions on the last slide, are welcome."),
 # ---- backup slides
 dict(kind="qual", src=10, tag="BACKUP (reused: qual slide 10): why batch correction is the wrong tool",
      notes="Backup, if asked why not scVI or Harmony: batch correction assumes the gap is noise and erases it, applies one unstructured shift to ribosomal genes and nuclear lncRNAs alike, and in practice over-corrects."),
 dict(kind="qual", src=29, tag="BACKUP (reused: qual slide 29): dataset inventory. Correction: Ding 2020 cortex has nuclei only (no matched cells); Bakken 2018 is the matched hold-out (done)",
      notes="Backup: the matched and orthogonal datasets. Matched brain data exist only off-v3, and matched v3 data only off-brain. Since the qual: Bakken 2018 done as the matched hold-out; Ding 2020 turned out to have nuclei only and is a four-chemistry nuclei-only test; Scala Patch-seq not started."),
 dict(kind="qual", src=31, tag="BACKUP (reused: qual slide 31): limitations as stated at the qual; the information-ceiling point is now a measured result (slide 15)",
      notes="Backup: the limitations I stated at the qual. The information ceiling is now quantified; the validation is no longer only indirect (fractionation, TRAP/RiboTag, Bakken hold-out)."),
]

# ------------------------------------------------------------------ helpers
prs = Presentation(QUAL)
fu = Presentation(FOLLOWUP)
layout_by_name = {l.name: l for l in prs.slide_layouts}
content_layout = layout_by_name["Single Block (with citation)"]
qual_slides = list(prs.slides)

def set_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text

def fill_placeholder(slide, idx, text, size=None, bold=False):
    for sh in slide.placeholders:
        if sh.placeholder_format.idx == idx:
            tf = sh.text_frame; tf.text = ""
            lines = text if isinstance(text, list) else [text]
            for i, line in enumerate(lines):
                para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                run = para.add_run(); run.text = line
                if size: run.font.size = Pt(size)
                run.font.bold = bold
            return sh
    return None

def import_slide(src):
    """Clone a slide from another deck with the same layout names: copy the shape tree, re-embed pictures."""
    dst = prs.slides.add_slide(layout_by_name[src.slide_layout.name])
    for sh in list(dst.shapes):                      # drop the layout's default empty placeholders
        sh._element.getparent().remove(sh._element)
    src_tree = src.shapes._spTree; dst_tree = dst.shapes._spTree
    for child in src_tree:
        tag = child.tag.split("}")[-1]
        if tag in ("nvGrpSpPr", "grpSpPr"): continue
        dst_tree.append(copy.deepcopy(child))
    # relationships: every r:embed / r:link in the copied tree
    for el in dst_tree.iter():
        for attr in ("embed", "link"):
            key = f"{{{NS_R}}}{attr}"
            rid = el.get(key)
            if not rid: continue
            rel = src.part.rels[rid]
            if rel.is_external:
                new_rid = dst.part.rels.get_or_add_ext_rel(rel.reltype, rel.target_ref)
            else:
                part = rel.target_part
                if "image" in rel.reltype:
                    img_part, new_rid = dst.part.get_or_add_image_part(io.BytesIO(part.blob))
                else:
                    raise RuntimeError(f"unhandled internal rel {rel.reltype} on imported slide")
            el.set(key, new_rid)
    if src.has_notes_slide:
        set_notes(dst, src.notes_slide.notes_text_frame.text)
    return dst

REPLACE = {
    "Qualifying Exam Presentation": "CBB Research in Progress Talk",
    "July 24th, 2024": "November 2026",
    "with a Conditional Variational Autoencoder": "Suchen Zheng, Gerstein Lab",
    "Aim 1.2: Localization-aware biological plausibility": "The nucleus-to-cell shift is cell-type-specific biology",
    "Aim 1.2 Preliminary Result": "SynGO / JASPAR validation (mouse MOp)",
    "Aim 3: Human M1 Zero-Shot Transfer": "Transfer to human motor cortex (zero-shot)",
    "Proposed Method": "The instrument: a conditional VAE with a modality flag",
    "Validation 2： Whether imputation add the right genes": "Genome-wide check against an independent localization database",
    "Validation 1: Whether imputation add the right genes": "What the model adds back is ribosomal and cytoplasmic-translation machinery",
}
def retitle(slide, extra=None):
    table = dict(REPLACE); table.update(extra or {})
    for sh in slide.shapes:
        if not sh.has_text_frame: continue
        for para in sh.text_frame.paragraphs:
            for run in para.runs:
                if run.text in table: run.text = table[run.text]
            pt = para.text.strip()
            if pt in table and para.runs:
                para.runs[0].text = table[pt]
                for run in para.runs[1:]: run.text = ""

# ------------------------------------------------------------------ build
new_order = []
for item in PLAN:
    k = item["kind"]
    if k == "qual":
        s = qual_slides[item["src"] - 1]
    elif k == "fu":
        s = import_slide(list(fu.slides)[item["src"] - 1])
    elif k == "pdf":
        s = prs.slides.add_slide(content_layout)
        fill_placeholder(s, 0, item["title"])
        for sh in list(s.placeholders):
            if sh.placeholder_format.idx in (1, 2): sh._element.getparent().remove(sh._element)
        h = Inches(5.45); w = int(h * 16 / 9)
        s.shapes.add_picture(page_png(item["src"]), int((prs.slide_width - w) / 2), Inches(1.3), width=w, height=h)
    elif k == "new":
        s = prs.slides.add_slide(content_layout)
        fill_placeholder(s, 0, item["title"])
        body = fill_placeholder(s, 1, [item["visual"]], size=14)
        if item.get("table"):
            rows = item["table"]
            body.top = Inches(4.9); body.height = Inches(1.8)
            tbl = s.shapes.add_table(len(rows), len(rows[0]), Inches(1.2), Inches(1.35), Inches(10.9), Inches(3.3)).table
            tbl.columns[0].width = Inches(6.2); tbl.columns[1].width = Inches(2.4); tbl.columns[2].width = Inches(2.3)
            for r, row in enumerate(rows):
                for c, val in enumerate(row):
                    cell = tbl.cell(r, c); cell.text = val
                    for para in cell.text_frame.paragraphs:
                        for run in para.runs: run.font.size = Pt(15); run.font.bold = (r == 0)
    if item.get("title") and k == "qual":
        fill_placeholder(s, 0, item["title"])
    if item.get("tag"):
        fill_placeholder(s, 13, item["tag"])
    retitle(s, {item["retitle_from"]: item["retitle"]} if item.get("retitle_from") else None)
    if item.get("retitle"):          # title placeholder text on qual validation slides
        fill_placeholder(s, 0, item["retitle"])
    set_notes(s, item["notes"])
    new_order.append(s)

# reorder sldIdLst to PLAN order; drop the qual slides not used
sldIdLst = prs.slides._sldIdLst
id_by_part = {prs.part.related_part(x.rId): x for x in list(sldIdLst)}
keep = {s.part for s in new_order}
for x in list(sldIdLst): sldIdLst.remove(x)
for s in new_order: sldIdLst.append(id_by_part[s.part])
for part, x in id_by_part.items():
    if part not in keep: prs.part.drop_rel(x.rId)

prs.save(OUT)
print("wrote", OUT, "slides:", len(prs.slides))
