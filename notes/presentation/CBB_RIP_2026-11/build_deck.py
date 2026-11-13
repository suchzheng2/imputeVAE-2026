"""Build the CBB research-in-progress (RIP) talk deck (2026-10-28).

Source decks: the qualifying follow-up pptx (Aug 2026; theme + reused slides) and the pre-qual PDF (May 2026;
pages inserted as full-slide images). New slides are placeholders carrying the main message (title), the main
visual to build (body), a NEW tag (bottom label) and the speaker notes.

Usage (impute2 env; PyMuPDF installed under the scratchpad pylib):
    PYTHONPATH=<pylib> python notes/presentation/CBB_RIP_2026-11/build_deck.py
"""
import copy, os, sys
from pptx import Presentation
from pptx.util import Inches, Pt

UP = "/home/sz477/.claude/uploads/86a8a919-16cc-5a84-8c48-e3eb3208a069"
FOLLOWUP = f"{UP}/bd939522-Qualifying_followup-meeting-slide_Suchen-v1_Autosaved.pptx"
PREQUAL = f"{UP}/b830e303-ImputeVAE_PreQual_Slides.pdf"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(OUT_DIR, "CBB_RIP_talk_v1.pptx")
IMG_DIR = os.path.join(OUT_DIR, "prequal_pages")
os.makedirs(IMG_DIR, exist_ok=True)

# ---------------------------------------------------------------- render pre-qual pages
import fitz
pdf = fitz.open(PREQUAL)
def page_png(n):
    p = os.path.join(IMG_DIR, f"prequal_p{n:02d}.png")
    if not os.path.exists(p):
        pdf[n - 1].get_pixmap(dpi=170).save(p)
    return p

# ---------------------------------------------------------------- deck plan
# kind: "keep" (follow-up slide index, 1-based), "pdf" (pre-qual page), "new" (placeholder)
PLAN = [
 dict(kind="keep", src=1, title="ImputeVAE: what a nucleus can tell you about its cytoplasm",
      sub="CBB Research in Progress talk  |  Suchen Zheng, Gerstein Lab  |  November 2026",
      notes="Hi everyone. I work on single-nucleus RNA-seq, and the question behind my project is simple: "
            "a nucleus only holds part of a cell's RNA. How much of the rest can we recover, and what does it change? "
            "I will show the biology we learned, the model we use as the instrument, and then one open question I want your opinion on."),
 dict(kind="pdf", src=4, title="Most brain transcriptomes are nuclei, not cells", tag="reused: pre-qual slide p4 (May 2026)",
      notes="Frozen, archived and postmortem tissue is where human brain data come from, and cells do not survive dissociation there. "
            "So the large human atlases, PsychENCODE, SEA-AD, ROSMAP, are nuclei. "
            "Anything designed for whole-cell data, from marker panels to many DE analyses, has to be applied to a partial transcriptome."),
 dict(kind="pdf", src=5, title="Nuclei are not cells, and the difference is structured", tag="reused: pre-qual slide p5 (May 2026)",
      notes="The missing part is not random. It is gene-specific: cytoplasmic transcripts like Fos and Arc are depleted. "
            "It is cell-type dependent: neurons move far more mRNA out to dendrites than glia do. "
            "And it is directional: the nuclear fraction of total mRNA ranges from 20 to 50 percent. That structure is what makes it learnable."),
 dict(kind="pdf", src=10, title="Reframe: learn the nucleus-to-cell map instead of erasing the gap", tag="reused: pre-qual slide p10 (May 2026)",
      notes="Batch correction treats the nucleus-versus-cell difference as noise and removes it. We treat it as biology and learn the mapping. "
            "Given a nucleus profile, generate what the whole-cell profile would look like. Training data are paired cells and nuclei from the same mouse motor cortex, 17 types, about 10 thousand genes."),
 dict(kind="keep", src=10, title=None, tag="reused: follow-up slide 10 (Aug 2026)",
      notes="First result: the learned shift is real. For every gene and cell type I compare the empirical cell-minus-nucleus delta with the predicted one. "
            "On the synaptic genes from SynGO the slope is one and the correlation is 0.94, so the magnitude is calibrated, not just the direction."),
 dict(kind="keep", src=11, title=None, tag="reused: follow-up slide 11 (Aug 2026)",
      notes="Within SynGO the neuronal shift is larger in every subgroup: presynaptic, postsynaptic, active zone, vesicle. "
            "The signal is not one mixed set; it appears across functional categories."),
 dict(kind="keep", src=12, title=None, tag="reused: follow-up slide 12 (Aug 2026)",
      notes="And it is specific. Transcription factor genes from JASPAR, a nuclear-associated contrast set, show a much smaller neuron-versus-glia contrast than synaptic genes. "
            "So the model is not inflating everything in neurons."),
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
            "TRAP and RiboTag profile ribosome-bound RNA in genetically defined populations. Imputed nuclei move from 0.43 to 0.73 correlation with the ribosome-bound profile, which is the ceiling set by real cells, and that holds in 19 populations including glia. "
            "One honest caveat: a per-type mean shift reaches the same numbers. I will come back to that."),
 dict(kind="keep", src=2, title=None, tag="reused: follow-up slide 2 (Aug 2026); update caption: single-stage, library-size covariate, no labels at inference",
      notes="The instrument. A conditional VAE with a shared encoder and decoder; a modality flag says whether a profile is a cell or a nucleus, and at inference we encode a nucleus and decode as a cell. "
            "Two things changed since this diagram was drawn: training is now single-stage, and the decoder takes an explicit library-size covariate, so the output RNA content is a parameter rather than something the latent has to carry. "
            "No cell-type label is needed at inference; the type is read from the nucleus itself."),
 dict(kind="new", title="Benchmark: best label-free translator; with labels, simple methods reach the ceiling",
      visual="MAIN VISUAL: native table (filled below). Optional: a one-axis ladder plot of the same numbers. Source: run/benchmark/test/results/ (split-aware benchmark, 3 seeds), notes/benchmark_review_2026-10-09.md §5b.",
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
 dict(kind="pdf", src=28, title="Known limitation: within-cell-type variance collapses", tag="reused: pre-qual slide p28 (May 2026)",
      notes="In May this was the open problem: within a cell type, imputed profiles had a fraction of the real variance. "
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
            "The variance-matching losses I proposed in August were dropped: they would be fitting noise."),
 dict(kind="keep", src=22, title=None, tag="reused: follow-up slide 22 (Aug 2026)",
      notes="Zero-shot on human motor cortex nuclei, with mouse-trained weights and one-to-one orthologs: neurons get the larger predicted shift, synaptic genes more than transcription factors, "
            "the same direction as mouse, and the per-class shifts correlate 0.72 with mouse across 17 classes. Plausibility only; there is no paired human truth."),
 dict(kind="new", title="Generalisation: unseen chemistry and an independent matched hold-out, with the ceiling and the baseline",
      visual="MAIN VISUAL: two panels. Left (10x v2, unseen chemistry): ladder of gene corr, old model 0.59 -> covariate model label-free scale 0.84-0.86 -> predicted type (reference classifier, 84 % acc.) 0.90 -> with label 0.93. "
             "Right (Bakken 2018 GSE123454, VISp SMART-seq, 461 nuclei / 445 cells, 11 matched clusters): predicted vs empirical delta scatter, r 0.58 pooled / 0.73 composition-free, sign agreement 0.94 = split-half ceiling; reliability ceiling 0.92; 'predicted type + training mean shift' baseline 0.56-0.59 drawn as a dashed line. "
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
 dict(kind="keep", src=25, title=None, tag=None, notes="Thank you. Questions, and especially opinions on the last slide, are welcome."),
]

# ---------------------------------------------------------------- build
prs = Presentation(FOLLOWUP)
E = 914400
layout_content = prs.slide_layouts[1]          # "Single Block (with citation)": idx 0 title, 1 body, 2 citation, 13 tag
src_slides = list(prs.slides)                  # keep references before deleting

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

new_order = []   # slide objects in final order
for item in PLAN:
    k = item["kind"]
    if k == "keep":
        s = src_slides[item["src"] - 1]
        if item.get("title"):
            fill_placeholder(s, 0, item["title"])
            if item.get("sub"): fill_placeholder(s, 1, item["sub"])
        if item.get("tag"):
            fill_placeholder(s, 13, item["tag"])
        set_notes(s, item["notes"])
        new_order.append(s)
    elif k == "pdf":
        s = prs.slides.add_slide(layout_content)
        fill_placeholder(s, 0, item["title"])
        # remove the empty body placeholder so the page image is not overlaid by prompt text
        for sh in list(s.placeholders):
            if sh.placeholder_format.idx in (1, 2):
                sh._element.getparent().remove(sh._element)
        fill_placeholder(s, 13, item["tag"])
        png = page_png(item["src"])
        # pre-qual pages are 16:9 (13.33 x 7.5): fit height 5.45 in under the title, centred
        h = Inches(5.45); w = int(h * 16 / 9)
        s.shapes.add_picture(png, int((prs.slide_width - w) / 2), Inches(1.3), width=w, height=h)
        set_notes(s, item["notes"])
        new_order.append(s)
    elif k == "new":
        s = prs.slides.add_slide(layout_content)
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
                        for run in para.runs:
                            run.font.size = Pt(15); run.font.bold = (r == 0)
        fill_placeholder(s, 13, item["tag"])
        set_notes(s, item["notes"])
        new_order.append(s)

# retitle stale headers on reused slides (run-level replacement keeps the formatting)
REPLACE = {
    "Qualifying Exam Follow-up Meeting ": "CBB Research in Progress Talk",
    "Qualifying Exam Follow-up Meeting": "CBB Research in Progress Talk",
    "Aug 19th, 2026": "November 2026",
    "Aim 1.2: Localization-aware biological plausibility": "The nucleus-to-cell shift is cell-type-specific biology",
    "Aim 1.2 Preliminary Result": "SynGO / JASPAR validation (mouse MOp)",
    "Aim 3: Human M1 Zero-Shot Transfer": "Transfer to human motor cortex (zero-shot)",
    "Recap - Method": "The instrument: a conditional VAE with a modality flag",
}
def retitle(slide):
    for sh in slide.shapes:
        if not sh.has_text_frame: continue
        for para in sh.text_frame.paragraphs:
            for run in para.runs:
                t = run.text
                if t in REPLACE: run.text = REPLACE[t]
            pt = para.text.strip()
            if pt in REPLACE and para.runs:          # text split across runs: collapse into the first run
                para.runs[0].text = REPLACE[pt]
                for run in para.runs[1:]: run.text = ""
for s in new_order:
    retitle(s)

# reorder: rewrite sldIdLst in PLAN order, drop the rest
sldIdLst = prs.slides._sldIdLst
id_by_slide = {}
for sldId in list(sldIdLst):
    rId = sldId.rId
    slide = prs.part.related_part(rId)
    id_by_slide[slide] = sldId
keep_parts = {s.part for s in new_order}
for sldId in list(sldIdLst):
    sldIdLst.remove(sldId)
for s in new_order:
    sldIdLst.append(id_by_slide[s.part])
# drop relationships to removed slides so their parts are not written
for sldId_part, sldId in id_by_slide.items():
    if sldId_part not in keep_parts:
        prs.part.drop_rel(sldId.rId)

prs.save(OUT)
print("wrote", OUT, "slides:", len(prs.slides))
