"""Build pre-qual slides from scratch using python-pptx."""
from pptx import Presentation
from pptx.util import Pt, Inches
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]

ORANGE = RGBColor(0xE8, 0x6C, 0x00)
DBLUE = RGBColor(0x1B, 0x3A, 0x5C)
MBLUE = RGBColor(0x2C, 0x5F, 0x8A)
GRAY = RGBColor(0x99, 0x99, 0x99)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x33, 0x33, 0x33)
LTBLUE = RGBColor(0xAA, 0xCC, 0xEE)


def _bar(slide, color, h):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def title_slide(title, sub, line3="", line4=""):
    s = prs.slides.add_slide(blank)
    _bar(s, DBLUE, prs.slide_height)
    tx = s.shapes.add_textbox(Inches(1), Inches(1.5), Inches(11), Inches(2))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = WHITE
    tx2 = s.shapes.add_textbox(Inches(1), Inches(4), Inches(11), Inches(2.5))
    tf2 = tx2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = sub
    p2.font.size = Pt(28)
    p2.font.bold = True
    p2.font.color.rgb = WHITE
    if line3:
        p3 = tf2.add_paragraph()
        p3.text = line3
        p3.font.size = Pt(18)
        p3.font.color.rgb = LTBLUE
    if line4:
        p4 = tf2.add_paragraph()
        p4.text = line4
        p4.font.size = Pt(16)
        p4.font.color.rgb = RGBColor(0x88, 0xAA, 0xCC)


def content_slide(title, bullets, aim="", cite=""):
    s = prs.slides.add_slide(blank)
    _bar(s, DBLUE, Inches(0.75))
    # Title
    tx = s.shapes.add_textbox(Inches(0.5), Inches(0.1), Inches(12.3), Inches(0.6))
    tf = tx.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = WHITE
    # Body
    tx2 = s.shapes.add_textbox(Inches(0.5), Inches(0.95), Inches(12.3), Inches(5.3))
    tf2 = tx2.text_frame
    tf2.word_wrap = True
    for i, b in enumerate(bullets):
        txt = b[0]
        lvl = b[1] if len(b) > 1 else 0
        bld = b[2] if len(b) > 2 else False
        clr = b[3] if len(b) > 3 else None
        if i == 0:
            p = tf2.paragraphs[0]
        else:
            p = tf2.add_paragraph()
        p.text = txt
        p.level = lvl
        p.space_after = Pt(4)
        p.font.size = Pt(18) if lvl == 0 else Pt(16)
        p.font.color.rgb = DARK
        if bld:
            p.font.bold = True
        if clr:
            p.font.color.rgb = clr
    # Aim label
    if aim:
        tx3 = s.shapes.add_textbox(Inches(0.5), Inches(6.9), Inches(5), Inches(0.4))
        tf3 = tx3.text_frame
        p = tf3.paragraphs[0]
        p.text = aim
        p.font.size = Pt(11)
        p.font.color.rgb = GRAY
    # Citation
    if cite:
        tx4 = s.shapes.add_textbox(Inches(7), Inches(6.9), Inches(6), Inches(0.4))
        tf4 = tx4.text_frame
        p = tf4.paragraphs[0]
        p.text = cite
        p.font.size = Pt(10)
        p.font.color.rgb = GRAY
        p.alignment = PP_ALIGN.RIGHT


# ============================================================
# SLIDES
# ============================================================

# 1
title_slide(
    "ImputeVAE: Translating snRNA-seq Profiles\ninto scRNA-seq-like Expression",
    "Suchen Zheng",
    "Pre-Qualifying Exam",
    "Gerstein Lab  |  CBB  |  Yale University",
)

# 2
content_slide("The Committee", [
    ("[List committee members with affiliations]", 0, False, GRAY),
])

# 3
content_slide("Outline", [
    ("Background", 0, True),
    ("The nucleus-cell expression gap", 1),
    ("Why batch correction is insufficient", 1),
    ("Specific Aims", 0, True),
    ("Aim 1: Cross-reconstructive VAE framework", 1),
    ("Aim 2: Cross-species generalization & validation", 1),
    ("Timeline", 0, True),
])

# --- BACKGROUND ---

# 4
content_slide("snRNA-seq Is Now the Dominant Modality", [
    ("Frozen, archived, postmortem tissue -> nuclei only", 0),
    ("Large nuclei-only datasets:", 0),
    ("PsychENCODE: 2.8M human brain nuclei", 1),
    ("Human Cell Atlas, SEA-AD, ROSMAP", 1),
    ("Disease cohorts (Alzheimer's, HIV, psychiatric)", 1),
    ("", 0),
    ("But: nuclei capture only nuclear RNA", 0, True, ORANGE),
    ("", 0),
    ("[Figure: growth of snRNA-seq datasets]", 0, False, GRAY),
], "Background")

# 5
content_slide("Nuclei != Cells: Structured Expression Differences", [
    ("Fewer detected transcripts in nuclei", 0),
    ("Depletion of cytoplasmic mRNAs (Fos, Arc, ribosomal)", 0),
    ("Enrichment of nuclear-retained lncRNAs", 0),
    ("These differences are:", 0),
    ("Gene-specific (not random noise)", 1, True),
    ("Cell-type-dependent (neurons > glia)", 1, True),
    ("Directional (cytoplasmic genes depleted)", 1, True),
    ("", 0),
    ("[Figure: cell diagram]", 0, False, GRAY),
], "Background", "Bakken 2018; Ding 2020")

# 6
content_slide("The Gap Varies by Cell Type", [
    ("Neurons: long processes, lots of cytoplasmic RNA -> large gap", 0),
    ("Compact glia: smaller gap", 0),
    ("Cell-type dependence makes global correction inadequate", 0, True),
    ("", 0),
    ("[Figure: per-cell-type differences]", 0, False, GRAY),
], "Background", "Bakken 2018; Ding 2020")

# 7
content_slide("Why Batch Correction Doesn't Work Here", [
    ("Harmony, scVI, Scanorama treat modality as batch", 0),
    ("Assumption: variation is stochastic and global", 0),
    ("Nucleus-cell difference violates this:", 0, True),
    ("Gene-specific: Fos depleted, housekeeping unchanged", 1),
    ("Cell-type-dependent: neurons > glia", 1),
    ("Directional: cytoplasmic genes systematically down", 1),
    ("", 0),
    ("Result: under-correct or over-correct", 0, True, ORANGE),
], "Background", "Korsunsky 2019; Lopez 2018")

# 8
content_slide("Alternative: Pseudobulk / Linear Regression", [
    ("Average within cell types -> per-gene correction", 0),
    ("Captures mean shifts", 0),
    ("Limitations:", 0, True),
    ("Discards within-cell-type variation", 1),
    ("Cannot produce per-cell profiles", 1),
    ("Assumes linear relationship", 1),
], "Background", "Crowell 2020")

# 9
content_slide("Our Approach: Learn the Transformation", [
    ("Nucleus-to-cell difference is not noise to remove", 0),
    ("It is a structured, learnable biological transformation", 0, True, DBLUE),
    ("", 0),
    ("Batch correction: \"make these look the same\"", 0),
    ("ImputeVAE: \"learn the mapping from one to the other\"", 0, True, ORANGE),
    ("", 0),
    ("Goal: given nuclei, generate the whole cell profile", 0),
], "Background")

# 10
content_slide("Specific Aims", [
    ("Aim 1: A Cross-Reconstructive VAE Framework", 0, True, DBLUE),
    ("Learn the structured transformation snRNA-seq -> scRNA-seq via cross-reconstruction", 1),
    ("", 0),
    ("Aim 2: Cross-Species Generalization & Biological Validation", 0, True, DBLUE),
    ("Test transfer across datasets/species; apply to human brain nuclei", 1),
])

# --- AIM 1 ---

# 11
title_slide("Aim 1", "A Cross-Reconstructive VAE Framework")

# 12
content_slide("ImputeVAE Architecture", [
    ("Shared encoder -> common latent space", 0),
    ("Alignment loss pushes same-CT distributions together", 1),
    ("Conditioned decoder + modality indicator", 0),
    ("Modality-specific outputs, shared parameters", 1),
    ("Cross-reconstruction path:", 0, True, ORANGE),
    ("Encode nuclei -> decode with cell condition -> compare to real cells", 1),
    ("", 0),
    ("[Figure 1: Architecture diagram]", 0, False, GRAY),
], "Aim 1")

# 13
content_slide("Why a Conditioned Decoder?", [
    ("Problem: shared decoder serves competing objectives", 0),
    ("Reconstruct cells / nuclei / cross-reconstruct", 1),
    ("Solution: modality indicator -> decoder specializes", 0, True),
    ("Shared -> conditioned improves all metrics", 0, True, ORANGE),
    ("", 0),
    ("[Figure: UMAP shared vs conditioned]", 0, False, GRAY),
], "Aim 1")

# 14
content_slide("Two-Stage Training", [
    ("Stage 1: Pretrain on cell data only", 0, True, DBLUE),
    ("Reconstruction + JSD, gradually annealed", 1),
    ("", 0),
    ("Stage 2: Joint training, both modalities", 0, True, DBLUE),
    ("Loss terms introduced gradually (cascading annealing)", 1),
    ("Cell recons -> Nuclei recons -> JSD -> Alignment -> Cross-recons", 1, True, ORANGE),
    ("Each ramped 0->1, stays on once active", 1),
    ("", 0),
    ("[Figure: cascading loss timeline]", 0, False, GRAY),
], "Aim 1")

# 15
content_slide("Cross-Reconstruction: Key Ingredient", [
    ("Encode nuclei -> decode with cell condition", 0),
    ("Compare to real cell profiles (same cell type)", 0),
    ("Directly supervises nuc -> cell translation", 0, True, ORANGE),
    ("", 0),
    ("Without: each modality reconstructed separately", 0),
    ("With: explicit supervision for the mapping", 0, True),
], "Aim 1")

# 16
content_slide("Training Data", [
    ("Allen Brain Cell Atlas - mouse cortex", 0, True),
    ("10X Chromium v3", 0),
    ("71K cells + 40K nuclei, 17 cell types", 0),
    ("16,439 genes (PerCT_5pct filtering)", 0),
    ("", 0),
    ("Recovers 58/59 known markers", 0),
    ("(Global threshold missed 29/59)", 1, False, GRAY),
], "Aim 1")

# --- RESULTS ---

# 17
title_slide("Aim 1", "Preliminary Results")

# 18
content_slide("UMAP: Raw Nuclei -> Imputed -> Real Cells", [
    ("[Figure: 3-panel UMAP colored by cell type]", 0, False, GRAY),
    ("", 0),
    ("Imputed profiles cluster with real cells, not raw nuclei", 0, True),
], "Aim 1 Results")

# 19
content_slide("Cell-Type Classification Accuracy", [
    ("Raw nuclei: 85.1%", 0),
    ("Linear regression: 94.7%", 0),
    ("ImputeVAE (cond): 93.7%", 0),
    ("ImputeVAE (cond+xr): 95.8%", 0, True, ORANGE),
    ("", 0),
    ("Cross-recons adds +2.1% over cond alone", 0),
    ("", 0),
    ("[Figure: bar chart, Y-axis from 0]", 0, False, GRAY),
], "Aim 1 Results")

# 20
content_slide("DE Gene Recovery", [
    ("+5.1% DE overlap improvement (cond+xr)", 0, True),
    ("PerCT_5pct: 54.7% DE overlap", 0),
    ("Gene correlation: 0.963", 0),
    ("", 0),
    ("[Figure: DE overlap chart]", 0, False, GRAY),
], "Aim 1 Results")

# 21
content_slide("Gene-Level Correlation", [
    ("[Figure: scatter - CT mean, imputed vs real, r=0.96+]", 0, False, GRAY),
    ("", 0),
    ("5x5 coexpression: cross-decoded vs real = 0.989", 0),
], "Aim 1 Results")

# 22
content_slide("Ablation: Each Component Matters", [
    ("Shared -> conditioned: prevents degradation", 0),
    ("No xr -> with xr: largest improvement", 0, True, ORANGE),
    ("MSE vs NB: MSE wins everywhere", 0),
    ("Scalar vs learned embed: comparable", 0),
    ("", 0),
    ("Cross-reconstruction is the critical ingredient", 0, True, DBLUE),
], "Aim 1 Results")

# 23
content_slide("Known Limitation: Variance Collapse", [
    ("Cell-type metrics strong, but:", 0),
    ("Within-CT variance collapses", 0, True, ORANGE),
    ("Cond+xr: 3%, Cond: 11%", 1),
    ("Hallucination test: CT-mean z -> <0.1% change", 0),
    ("Model learns CT averages, ignores per-cell variation", 0, True, ORANGE),
    ("", 0),
    ("[Figure: variance comparison]", 0, False, GRAY),
], "Aim 1 Results")

# 24
content_slide("Ongoing: Variance Retention", [
    ("Normalized loss: 11% -> 45%", 0, True),
    ("CPM normalization", 0),
    ("Full conditional VAE", 0),
    ("Single-stage training", 0),
    ("", 0),
    ("[Table: ongoing experiment results]", 0, False, GRAY),
], "Aim 1 Results")

# 25
content_slide("Aim 1: Next Steps", [
    ("Baselines: scVI, Harmony", 0),
    ("Held-out cell type evaluation", 0),
    ("Gene-stratified evaluation", 0),
    ("Resolve variance collapse", 0),
], "Aim 1")

# --- AIM 2 ---

# 26
title_slide("Aim 2", "Cross-Species Generalization &\nBiological Validation")

# 27
content_slide("Does the Transformation Generalize?", [
    ("Nucleus-cell gap = RNA localization biology", 0),
    ("If conserved -> model transfers to primates", 0, True, DBLUE),
    ("If not -> RNA compartmentalization has diverged", 0, True, DBLUE),
    ("", 0),
    ("Either outcome is informative", 0, True, ORANGE),
], "Aim 2")

# 28
content_slide("Expression Conservation Across Species", [
    ("Orthologous genes preserve CT expression across mammals", 0),
    ("Core markers conserved; magnitudes differ 2-5x", 0),
    ("", 0),
    ("Which genes affected -> may be conserved", 0, True),
    ("How much -> may need species-specific tuning", 0),
], "Aim 2", "Hodge 2019; Bakken 2021")

# 29
content_slide("Available Paired Datasets", [
    ("Mouse cortex: Allen Brain (10X) - training", 0),
    ("Mouse cortex: Ding 2020 (multiple protocols) - cross-dataset", 0),
    ("Macaque dLGN: Bakken 2021 (SMART-Seq) - cross-species", 0),
    ("Human brain: nuclei only - no paired data", 0, True, ORANGE),
    ("", 0),
    ("Bakken = only paired primate brain dataset", 0, True),
], "Aim 2")

# 30
content_slide("2a: Cross-Dataset (Mouse Cortex)", [
    ("Apply model to Ding et al. mouse cortex", 0),
    ("Same tissue, different lab, multiple protocols", 0),
    ("", 0),
    ("Tests robustness to:", 0, True),
    ("Lab effects, sample prep, protocol", 1),
    ("", 0),
    ("Establishes baseline before adding species", 0, True, DBLUE),
], "Aim 2", "Ding 2020")

# 31
content_slide("2b: Cross-Species (Mouse -> Macaque)", [
    ("Three strategies on macaque paired data:", 0),
    ("", 0),
    ("1. Direct transfer via orthologs", 0, True),
    ("Is transformation conserved?", 1),
    ("2. Fine-tuning from mouse weights", 0, True),
    ("Shared structure + species tuning?", 1),
    ("3. Retrain from scratch", 0, True),
    ("Upper bound (small data, may overfit)", 1),
], "Aim 2", "Bakken 2021")

# 32
content_slide("What Each Outcome Tells Us", [
    ("Direct transfer works:", 0, True, DBLUE),
    ("RNA localization universal across mammals", 1),
    ("", 0),
    ("Fine-tuning needed:", 0, True, DBLUE),
    ("Shared structure, species-specific magnitude", 1),
    ("", 0),
    ("Only retraining works:", 0, True, DBLUE),
    ("Transformation is species-specific", 1),
    ("", 0),
    ("All outcomes informative", 0, True, ORANGE),
], "Aim 2")

# 33
content_slide("2c: Human Brain Nuclei", [
    ("No paired data -> biological plausibility evaluation", 0),
    ("", 0),
    ("1. Cytoplasmic RNA recovery", 0, True),
    ("Fos, Arc, ribosomal up; lncRNAs unchanged", 1),
    ("2. Cell-type marker improvement", 0, True),
    ("3. PsychENCODE downstream DE analysis", 0, True),
    ("Recover disease DE signatures?", 1),
    ("", 0),
    ("Failures -> species/tissue-specific localization", 0, False, GRAY),
], "Aim 2")

# 34
content_slide("Aim 2: Next Steps", [
    ("Macaque paired data (Bakken et al.)", 0),
    ("Ortholog gene mapping", 0),
    ("Three-strategy transfer experiment", 0),
    ("Human nuclei plausibility evaluation", 0),
], "Aim 2")

# --- CLOSING ---

# 35
content_slide("Summary", [
    ("Aim 1: Structured nuc -> cell transformation", 0, True, DBLUE),
    ("Cross-recons + conditioned decoder", 1),
    ("Strong CT metrics; variance ongoing", 1),
    ("", 0),
    ("Aim 2: Generalization + human application", 0, True, DBLUE),
    ("Cross-dataset -> cross-species -> human", 1),
    ("All outcomes informative", 1),
    ("", 0),
    ("Not batch correction - biological transformation", 0, True, ORANGE),
])

# 36
content_slide("Timeline", [
    ("Summer 2026: Aim 1 baselines + variance", 0),
    ("Fall 2026: Aim 2a cross-dataset, macaque prep", 0),
    ("Spring 2027: Aim 2b cross-species, qual exam", 0),
    ("Summer-Fall 2027: Aim 2c human application", 0),
    ("2028: Manuscript", 0),
    ("", 0),
    ("[Replace with Gantt chart]", 0, False, GRAY),
])

# 37
title_slide("Thank You", "Questions?", "Acknowledgements", "Mark Gerstein, Lab, Committee")

# --- BACKUP ---

# 38
title_slide("Backup Slides", "")

# 39
content_slide("Loss Function Details", [
    ("MSE on normalized data", 0),
    ("JSD latent alignment", 0),
    ("Cross-recons: encode nuc -> decode cell -> MSE vs real", 0),
    ("Cascading annealing: each loss 0->1 over ~100 epochs", 0),
], "Backup")

# 40
content_slide("Hyperparameters", [
    ("input -> 1024 -> 512 -> 32 -> 512 -> 1024 -> output", 0),
    ("Embed: dim=8 learned or scalar", 0),
    ("LR 1e-4, Adam, batch 128", 0),
    ("S1: 300 epochs, S2: 600 epochs", 0),
], "Backup")

# 41
content_slide("MSE vs Negative Binomial", [
    ("NB: principled but softmax bottleneck", 0),
    ("Dominant genes suppress low-expression", 1),
    ("MSE log1p: each gene independent", 0),
    ("MSE wins on every metric", 0, True, ORANGE),
], "Backup")

# 42
content_slide("Variance Collapse Deep Dive", [
    ("CT-mean z -> output changes <0.1%", 0),
    ("Within-CT var: cond+xr=0.035, cond=0.115", 0),
    ("Same-modality recon ~1.0", 0),
    ("Loss in cross-decode specifically", 0, True),
    ("Normloss -> 0.454", 0),
], "Backup")

out = "notes/qualify-exam/pre-qual/ImputeVAE_PreQual_Slides.pptx"
prs.save(out)
print(f"Saved {len(prs.slides)} slides to {out}")
