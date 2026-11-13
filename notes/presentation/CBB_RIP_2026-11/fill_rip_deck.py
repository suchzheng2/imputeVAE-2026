"""Fill the placeholder slides of Suchen's revised RIP deck (cbb-rip-talk-10-28-2026.pptx, 20 slides, 2026-10-28)
with the new result figures, retitle the remaining 'Aim' tags, drop the duplicate JASPAR slide, add the
variance / latent / open-question slides, and write speaker notes.  Output: CBB_RIP_talk_v3_filled.pptx.
"""
import os, copy
from pptx import Presentation
from pptx.util import Inches, Pt

SRC = "/home/sz477/.claude/uploads/86a8a919-16cc-5a84-8c48-e3eb3208a069/f8cff458-cbb-rip-talk-10-28-2026.pptx"
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figures"); OUT = os.path.join(HERE, "CBB_RIP_talk_v3_filled.pptx")
prs = Presentation(SRC); S = list(prs.slides); layout = prs.slide_layouts[1]
E = 914400

def ph(slide, idx):
    for sh in slide.placeholders:
        if sh.placeholder_format.idx == idx: return sh
def set_text(shape, lines, size=None, bold_first=False):
    tf = shape.text_frame; tf.text = ""; tf.word_wrap = True
    for i, line in enumerate(lines if isinstance(lines, list) else [lines]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = line
        if size: r.font.size = Pt(size)
        if bold_first and i == 0: r.font.bold = True
def notes(slide, text): slide.notes_slide.notes_text_frame.text = text
def title_box(slide):
    """The deck uses a free text box named 'Title 1' on most slides, a title placeholder on others."""
    t = ph(slide, 0)
    if t is not None: return t
    for sh in slide.shapes:
        if sh.has_text_frame and sh.name.startswith("Title"): return sh
def retitle(slide, text):
    tb = title_box(slide); p = tb.text_frame.paragraphs[0]
    if p.runs: p.runs[0].text = text; [setattr(r, "text", "") for r in p.runs[1:]]
    else: p.add_run().text = text
    for extra in tb.text_frame.paragraphs[1:]:
        for r in extra.runs: r.text = ""
def tag(slide, text):
    t = ph(slide, 13)
    if t is not None: set_text(t, text)
def figure_slide(slide, title, png, conclusion, tagtext, box_h=4.7):
    retitle(slide, title)
    body = ph(slide, 1)
    if body is not None:                     # body becomes a conclusion strip under the figure
        body.top, body.height = Inches(1.25 + box_h + 0.1), Inches(6.75 - (1.25 + box_h + 0.1))
        set_text(body, conclusion, size=14)
    from PIL import Image
    w0, h0 = Image.open(png).size
    h = Inches(box_h); w = int(h * w0 / h0)
    if w > Inches(12.5): w = Inches(12.5); h = int(w * h0 / w0)
    slide.shapes.add_picture(png, int((prs.slide_width - w) / 2), Inches(1.25), width=w, height=h)
    tag(slide, tagtext)
def new_slide(): return prs.slides.add_slide(layout)
def move_after(slide, after):
    lst = prs.slides._sldIdLst; ids = list(lst)
    me = [x for x in ids if prs.part.related_part(x.rId) is slide.part][0]
    ref = [x for x in ids if prs.part.related_part(x.rId) is after.part][0]
    lst.remove(me); lst.insert(list(lst).index(ref) + 1, me)
def delete(slide):
    lst = prs.slides._sldIdLst
    for x in list(lst):
        if prs.part.related_part(x.rId) is slide.part:
            lst.remove(x); prs.part.drop_rel(x.rId); return

# ---------------------------------------------------------------- 1 title
notes(S[0], "Hi everyone. I work on single-nucleus RNA-seq, and the question behind my project is simple: a nucleus only holds part of a cell's RNA. "
            "How much of the rest can we recover, and what does it change? I will show the biology of the gap, the model we use as the instrument, how we checked it against data it never saw, "
            "what we learned about what a nucleus can and cannot tell you, and then one open question I want your opinion on.")

# ---------------------------------------------------------------- 8 benchmark ladder
s = S[7]; retitle(s, "Benchmark: best label-free translator; with labels, simple methods reach the ceiling")
body = ph(s, 1); body.top, body.height = Inches(5.0), Inches(1.7)
set_text(body, ["Held-out test split of the paired MOp data (split-aware benchmark, 3 seeds). Gene correlation of per-type means.",
                "Method claim: the best translator when no cell-type label is available at inference. With labels, the type mean is the ceiling.",
                "scGen and scButterfly learn a map too; the difference here is the shared latent with a library-size covariate and no label at inference."], size=14)
rows = [["Setting", "Gene corr (test)", "Label at inference"],
        ["Cell-type mean of real cells (ceiling)", "0.981", "yes"], ["scGen", "0.975", "yes"],
        ["ImputeVAE, predicted type (covariate model)", "0.962", "no"], ["ImputeVAE, no labels anywhere", "0.921", "no"],
        ["ImputeVAE, no covariate (July model)", "0.900", "no"], ["scVI / scANVI, label-free", "0.86", "no"]]
tbl = s.shapes.add_table(len(rows), 3, Inches(1.2), Inches(1.3), Inches(10.9), Inches(3.4)).table
tbl.columns[0].width, tbl.columns[1].width, tbl.columns[2].width = Inches(6.2), Inches(2.4), Inches(2.3)
for r, row in enumerate(rows):
    for c, val in enumerate(row):
        cell = tbl.cell(r, c); cell.text = val
        for p in cell.text_frame.paragraphs:
            for run in p.runs: run.font.size = Pt(15); run.font.bold = (r == 0 or r == 3)
tag(s, "split-aware benchmark, notes/benchmark_review_2026-10-09.md §5b")
notes(s, "Before the validation, where the model stands. This is a held-out split of the paired data. If you have cell-type labels, the cell-type mean of real cells is the ceiling and scGen sits just under it. "
         "Without a label at inference, ImputeVAE reaches 0.96 by reading the type from the nucleus itself, well above label-free scVI and scANVI at 0.86. "
         "So the claim is: the best label-free translator. The claim is not that a large model beats the type mean; it does not, and I would rather say that than have you say it. "
         "The bold row is the model everything that follows uses.")

# ---------------------------------------------------------------- 9, 10 roadmap and internal fidelity
tag(S[8], "Validation roadmap")
notes(S[8], "Three tiers of validation, each against something the model never trained on. First, internal fidelity on held-out matched data. "
            "Second, annotation-based plausibility: do the recovered shifts follow known nuclear, cytoplasmic and synaptic gene programs? "
            "Third, an orthogonal assay that measures cytoplasmic RNA directly. The third tier is new since the qual.")
tag(S[9], "Validation 1: internal fidelity")
for sh in S[9].shapes:
    if sh.has_table:
        t = sh.table
        t.cell(1, 1).text = "delta corr 0.95; sign agreement 0.94"
        t.cell(2, 1).text = "neurons 4 to 5x glia"
        for r in (1, 2):
            for p in t.cell(r, 1).text_frame.paragraphs:
                for run in p.runs: run.font.size = Pt(14)
notes(S[9], "Tier one. On held-out motor cortex data, predicted and empirical cell-minus-nucleus shifts correlate 0.95 across gene and cell-type pairs with sign agreement 0.94, "
            "and the shift is four to five times larger in neurons than in glia. The model learned structured differences, not one average correction.")

# ---------------------------------------------------------------- 11 datasets (merged SynGO + JASPAR); 12 duplicate -> delete
retitle(S[10], "Biological Validity 2: Annotation-based plausibility"); tag(S[10], "Validation 2: gene sets")
for i in (12, 13, 14): tag(S[i], "Validation 2: results")

# ---------------------------------------------------------------- 16 TRAP intro, 17 TRAP result
s = S[15]; retitle(s, "Biological Validity 3: Orthogonal assay. Ribosome-bound RNA from genetically defined populations")
set_text(ph(s, 1), [
    "What TRAP / RiboTag measures",
    "A tagged ribosomal protein is expressed in one cell type inside intact tissue; tagged ribosomes are pulled down and the bound mRNA is sequenced. No dissociation, no nucleus / cell split. Ribosome-bound mRNA is cytoplasmic by definition.",
    "",
    "Why it is a fair reference",
    "Never entered training; population-level, so it tests the cell-type-level claim; nine datasets from at least six labs, other cortical regions.",
    "",
    "The comparison",
    "Map each external population to a training cell type. Correlate (Spearman, 7,700 to 8,700 genes) the external profile with three per-type pseudobulks: raw nuclei, nuclei imputed by the model, real whole cells. Real cells set the ceiling. Gene and replicate bootstraps.",
    "",
    "17 populations: pan-excitatory, L5a, three L5b lines (two from motor cortex), Pvalb, Sst, Vip, Htr3a interneurons, astrocytes, oligodendrocytes."], size=15)
tag(s, "Validation 3: orthogonal assay. GSE133291, GSE172484, GSE202172, GSE141337, GSE141874, GSE139684, GSE99791, GSE211162, GSE263082, GSE77243")
notes(s, "William suggested ribosome profiling at the qual. TRAP and RiboTag pull ribosome-bound mRNA out of one genetically defined cell type in intact tissue, so they give a cytoplasmic, cell-type-resolved reference "
         "that has nothing to do with our training data. The question is simple: after imputation, does a nucleus profile look more like that type's ribosome-bound profile, and how close does it get to what real whole cells of that type achieve? "
         "Real cells are the ceiling; the number to watch is how much of the nucleus-to-cell gap imputation closes.")
figure_slide(S[16], "Imputation closes the nucleus-to-cell gap to external ribosome-bound profiles in every population",
             f"{FIG}/ripfig_trap_h2.png",
             ["Imputed nuclei reach the real-cell ceiling in all 17 populations (median gap closed 0.99; every bootstrap CI excludes zero), for excitatory, inhibitory and glial types. Glia sit lower because the reference itself is lower.",
              "Caveats: ribosome-bound RNA is translation, so this supports direction and rank; and at pseudobulk level every method ties (CT-mean, scVI, scGen, scButterfly within 0.03), so this validates the shift, not the model over baselines."],
             "Validation 3: result. validation/trap_ribotag_external_validation/outputs/report/extension_newer_datasets.md", box_h=4.4)
notes(S[16], "Gray is the raw nucleus, blue the imputed nucleus, the orange tick is real whole cells. In all seventeen populations the imputed profile lands on the ceiling: median gap closed 0.99 and every confidence interval excludes zero. "
             "Excitatory, inhibitory and glial populations all behave the same way; glia are lower only because their reference is lower. Two things to keep in mind. Ribosome-bound RNA is translation, so this is a rank and direction test. "
             "And at this pseudobulk level a cell-type mean does as well as the model, so what this validates is that the nucleus-to-cell shift we learned is real biology. What the model adds over a mean is the question the next section answers.")

# ---------------------------------------------------------------- 18 variance limitation (+ new B2 and latent slides after it)
s = S[17]; retitle(s, "Known limitation: within-type variance after imputation is 15 % of real cells. Collapse, or scale change?")
body = ph(s, 1); body.left, body.top, body.width, body.height = Inches(7.3), Inches(1.3), Inches(5.7), Inches(5.3)
set_text(body, ["The concern (Prashant, Sep 2026)",
                "A model that assigns nearly the same value to every cell would also show low variance: every gene would end with a similar low variance.",
                "",
                "The test",
                "Per cell type and gene: within-type variance in real cells, raw nuclei and imputed cells (log1p scale, genes detected in > 1 % of cells).",
                "Readouts: the distribution of per-gene variances, and the Spearman rank correlation of each gene's variance in real cells against imputed cells. Reference: the same correlation for real cells vs raw nuclei.",
                "",
                "Prediction",
                "Uniform flattening: narrow imputed distribution, slope near zero. Scale change: ranks and shape kept."], size=14, bold_first=True)
from PIL import Image
png = os.path.join(HERE, "prequal_pages", "prequal_p28.png"); w0, h0 = Image.open(png).size
w = Inches(6.6); h = int(w * h0 / w0); s.shapes.add_picture(png, Inches(0.4), Inches(1.6), width=w, height=h)
tag(s, "pre-qual slide p28 (May 2026) + variance_rank/README.md")
notes(s, "At the qual this was the open problem: within a cell type, imputed profiles carry fifteen percent of the real-cell variance. "
         "Prashant asked the right question in September: is that a collapse, where every cell gets nearly the same value, or a scale change that keeps the same genes variable? "
         "The test is simple: compare the per-gene variance distributions and ask whether the genes that vary most in real cells are the genes that vary most after imputation.")
b2 = new_slide(); move_after(b2, s)
figure_slide(b2, "The ranks are kept and the distribution is stretched: a scale change, not a collapse",
             f"{FIG}/ripfig_l5it_variance.png",
             ["L5 IT: medians 0.39 (cells) / 0.21 (nuclei) / 0.08 (imputed). Spearman of per-gene variance, real vs imputed, 0.92, above the nucleus reference 0.72; 0.80 after removing Poisson and library-size variance. Low-variance genes are compressed most (ratio 0.02 below 10 % detection, 0.35 above 70 %). Same in Oligo (0.93) and 14 of 17 types.",
              "So the loss is a scale change on sparse genes, not flattening. What remains has the right ordering; its structure is the real problem, next slide."],
             "notes/experiments/variance_zero_nonzero_fullmodel/variance_rank/ (2026-09-24)", box_h=4.4)
notes(b2, "Left: the distribution of per-gene variance in L5 IT. Orange is real cells, gray nuclei, blue imputed. The imputed distribution is shifted down and stretched, not squeezed to a spike. "
          "Right: each gene's variance in real cells against its variance after imputation. Spearman 0.92, higher than the nucleus itself at 0.72, so the genes that vary most in real cells are the genes that vary most after imputation. "
          "The loss is a scale change concentrated on sparse genes. One caveat for questions: variance tracks mean on this scale, so part of the rank agreement is inherited from mean agreement; after regressing out the mean the residual Spearman is 0.55. "
          "That ruled out flattening and pointed at what was really going on: nearly all of the variance that survives sits on a single axis.")
lt = new_slide(); move_after(lt, b2)
figure_slide(lt, "What a nucleus can and cannot tell you about its cytoplasm: type identity plus RNA content",
             f"{FIG}/ripfig_latent_finding.png",
             ["Only 6 of 32 latent dimensions are used; one of them is library size and carries 95 % of the imputed within-type variance. Real within-type variance is half Poisson sampling, a quarter RNA content, a quarter biology.",
              "With RNA content as an explicit decoder covariate (the reported model), 2 identity dimensions still place 15 of 17 types. At 10x depth a nucleus determines its type-level cytoplasm and its RNA content; individual-nucleus cytoplasm beyond that is below the sampling ceiling. The variance losses proposed in August were dropped: they would fit noise."],
             "notes/experiments/variance_zero_nonzero_fullmodel/ (latent_usage, F9), libsize review §6.7", box_h=4.3)
notes(lt, "This is the finding of the summer. The model uses six of its thirty-two latent dimensions, and the one that carries nearly all of the within-type variation after imputation is library size, the RNA content of the nucleus. "
          "Real within-type variance is about half Poisson sampling, a quarter RNA content and a quarter biology, and the model reproduces the RNA-content part. "
          "Once RNA content is an explicit covariate, two identity dimensions are enough to place fifteen of seventeen types, so capacity was never the limit. "
          "The honest statement is: at 10x depth a nucleus tells you its type-level cytoplasm plus its RNA content, not its individual cytoplasm. I had proposed variance-matching losses in August; they were dropped, because they would be fitting sampling noise.")

# ---------------------------------------------------------------- 19 generalisation
figure_slide(S[18], "Generalisation: unseen chemistry and an independent matched hold-out, with the ceiling and the baseline",
             f"{FIG}/ripfig_generalisation.png",
             ["Left: on 10x v2 the July model fell to 0.59; with the explicit output scale and a label-free type assignment the covariate model reaches 0.90 (0.93 with the label). Right: Bakken 2018 SMART-seq, the only public matched mouse cortex set outside Allen 10x: direction at the split-half ceiling, magnitude r 0.58 against a reliability ceiling of 0.92.",
              "The baseline matters: predicting the type and adding its mean training shift scores the same. On external data the model's contribution is the label-free, chemistry-robust type assignment, not extra per-gene accuracy. Zero-shot human M1 and Lake kidney show the same conserved core (ribosomal up, MALAT1 down)."],
             "libsize review §6.4a; notes/experiments/bakken2018_holdout/README.md", box_h=4.3)
notes(S[18], "Two generalisation tests. Left, unseen chemistry: on 10x v2 the July model fell to 0.59. With the output scale as an explicit parameter and the type read from the nucleus by a reference classifier, the covariate model recovers 0.90, against 0.93 with labels. "
             "Right, an independent matched hold-out: Bakken 2018, SMART-seq nuclei and cells from visual cortex, eleven matched clusters. The direction of the shift is predicted at the split-half ceiling, the magnitude at 0.58 against a reliability ceiling of 0.92. "
             "But note the baseline: predicting the type and adding that type's mean training shift scores the same. So on external data what the model contributes is the label-free, chemistry-robust type assignment. That is the honest summary, and it leads to the question I want to ask you.")

# ---------------------------------------------------------------- open question (new, before Thank you)
oq = new_slide(); move_after(oq, S[18])
retitle(oq, "Open question: which nuclei-only result would change if we applied this?")
set_text(ph(oq, 1), [
    "Every external test so far is matched by 'predicted type + that type's mean nucleus-to-cell shift'.",
    "So the contribution is a validated, cell-type-resolved compartment map that can be applied label-free and across chemistries. The paper needs one nuclei-only result that changes once it is applied.",
    "",
    "Candidates",
    "(a) Cytoplasmic / activity gene recovery (Fos, Arc, ribosomal, Plp1) in an independent snRNA-seq set, checked against matched whole-cell data.",
    "(b) Disease DE on imputed profiles: SCORCH human snRNA-seq, 655k nuclei, HIV vs control, PFC / insula / striatum (on the group share). Does the DE list move toward what whole-cell data show?",
    "(c) Marker-based annotation of nuclei improves after imputation, on external data (0.73 -> 0.96 in the benchmark).",
    "",
    "Committee asks since the qual: TRAP / RiboTag done; nuclear lncRNA showcase (Malat1, Meg3; 45 genes, 100 % correct sign) done; subcellular spatial (MERFISH) not started.",
    "",
    "Which of these would you find convincing? Is there one I am missing?"], size=16, bold_first=True)
tag(oq, "discussion")
notes(oq, "Here is where I want your input. The compartment map is validated, label-free and chemistry-robust, but on every external test a type mean plus a mean shift does as well. "
          "So the contribution has to be an application: a nuclei-only result that changes once you impute. Three candidates: recovering activity and cytoplasmic genes in an independent nuclei set; "
          "re-running a disease DE study on imputed profiles, where we have a 655-thousand-nucleus human set on the group share; or showing that annotation of nuclei gets better after imputation on external data. "
          "Which of these would you find convincing, and is there one I am missing?")
notes(S[19], "Thank you. Questions, and especially opinions on the last slide, are welcome.")

delete(S[11])   # duplicate JASPAR slide; deleted last so new slides never reuse its part name
prs.save(OUT); print("wrote", OUT, "slides", len(prs.slides))
