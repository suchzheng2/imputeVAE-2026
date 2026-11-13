# CBB RIP talk script (deck `cbb-rip-talk-10-28-2026.pptx`, 20 slides), 2026-10-28

Paced for ~110 words/min. Word counts and cumulative time in brackets. Short sentences; "/" marks a breath.
Total ~2,550 words ≈ 23 min.

---

## 1. Title (60 words, 0:30)

Hi everyone. / I'm Suchen, from the Gerstein lab. / I work on single-nucleus RNA-seq.

The question behind my project is simple. / A nucleus holds only part of a cell's RNA. / How much of the rest can we recover from the nucleus alone? / And what can it tell us?

Today I will show the biology of the gap, / the model we built, / how we tested it, / and what we learned about its limits.

## 2. snRNA-seq dominates (120 words, 1:30)

First, why this problem exists.

Most human brain tissue is frozen or postmortem. / To get single cells, you have to break the tissue apart. / Whole brain cells do not survive that. / Only the nucleus survives.

So for most brain samples, / single-nucleus RNA-seq is the only option. / And it is now the dominant data type. / PsychENCODE has 2.8 million nuclei. / SEA-AD, ROSMAP, the Human Cell Atlas: / all nuclei.

The catch is on the bottom line. / A nucleus is only part of the cell. / So all of this data is locked out of analyses that were designed for whole cells. / Marker panels, differential expression, many downstream tools.

## 3. A nucleus is not a smaller cell (130 words, 2:45)

You might hope a nucleus is just a shrunk-down cell. / Same information, smaller scale. / It is not.

The nucleus sits inside the cell. / The cytoplasm around it carries RNA that the nucleus does not have. / And what is missing is structured, in three ways.

First, it is gene-specific. / Ribosomal, cytoskeletal and synaptic transcripts are depleted in the nucleus.

Second, it is large and directional. / Nuclear RNA is only 20 to 50 percent of a cell's total mRNA. / So the nucleus always misses a big, consistent part.

Third, it is cell-type dependent. / Neurons ship much more RNA out into dendrites than glia do. / So the gap is larger in neurons.

That structure is what makes the gap learnable.

## 4. The gap is biology: RNA localization (150 words, 4:05)

Why is it structured this way? / Because it follows known rules of RNA localization.

On the left: / the nucleus deliberately holds some RNAs. / Malat1 and Neat1, the red dots above the line, / never leave the nucleus. / These are the genes a model should push down.

In the middle: / in neurons, a lot of RNA is shipped far out into the dendrites. / This image shows Camk2a mRNA, in red, spread through the whole dendritic tree. / About 2,500 transcripts do this. / These are the genes a model should add back.

And on the right, the most useful point: / other people have already measured where each gene's RNA lives. / APEX-seq, lncATLAS, RNALocate. / Public compartment maps that my model never saw.

That gives a clean, testable prediction. / Genes the model adds should be cytoplasmic. / Genes it removes should be nuclear. / I will test exactly that.

## 5. Why not batch correction (130 words, 5:15)

The obvious first idea is batch correction. / Treat "nucleus versus cell" as a batch, and run Harmony or scVI.

But batch correction makes three assumptions. / The batch label carries no biology. / The effect is unstructured, just noise and global shifts. / And the model never sees a matched pair.

Here, every assumption is wrong. / The difference IS biology, / it is gene-specific and cell-type-specific, / and we do have matched data.

So in practice, batch correction erases the signal. / It applies the same shift to ribosomal genes and to nuclear lncRNAs. / And scVI over-corrects: / it pushes everything down by about 1.3 log units, / which inverts the real biology.

We need a method that learns the difference, / instead of removing it.

## 6. Method: shared embedding (150 words, 6:35)

So here is the model. / ImputeVAE is a conditional variational autoencoder.

Cells and nuclei go through one shared encoder / into one latent space. / A modality flag, / one for cell, zero for nucleus, / tells the encoder and decoder which kind of profile it is looking at.

Training uses three losses. / Reconstruction, so each modality is preserved. / A prior term, to regularize the latent. / And an alignment term, / which pulls the cell and nucleus distributions together / within each cell type.

Training data are paired cells and nuclei from the same mouse motor cortex. / Allen Institute data, 10x version 3. / 71 thousand cells, 40 thousand nuclei, / 17 shared cell types, / about 10 thousand genes.

One thing to note: / the cell-type labels are used only in training, for the alignment loss. / At inference, the model sees expression only.

## 7. Method: flip the flag (70 words, 7:15)

At inference, it is one step.

We encode a nucleus with the nucleus flag. / Then we decode it with the cell flag. / The output is what the whole-cell profile of that nucleus should look like.

Two updates since the qual: / training is now single-stage, / and the decoder takes an explicit library-size covariate, / so the output RNA content is a parameter we set. / I will come back to why that matters.

## 8. Joint latent space (110 words, 8:15)

Does it work at all? / Three views.

Top left: the latent space colored by modality. / Cells and nuclei overlap. / The model found a shared representation.

Bottom left: real cells versus cross-decoded nuclei. / After decoding with the cell flag, / the imputed nuclei land where the real cells are.

Right: the final latent colored by cell type. / The 17 types stay separate. / So the alignment did not erase identity.

This is the qualitative check. / Now the numbers.

## 9. Benchmark (190 words, 10:00)

**[Script for the label-free figure, `ripfig_benchmark_labelfree.png`.]**

This is a held-out benchmark. / 8,000 nuclei the model never saw. / Every method trained on the same data.

I show the setting a nuclei-only user is actually in: / no cell-type label at inference. / The model sees the expression vector only.

Left panel: / correlation of the per-type means with real cells. / ImputeVAE reaches 0.90. / scVI and scANVI, decoded without a label, reach 0.86. / The error bars are three seeds, and they do not overlap.

Right panel: / energy distance. / It asks whether the imputed population looks like real cells inside a type, / not just on average. / ImputeVAE is at 6.8, / scVI and scANVI around 9, / raw nuclei at 129.

One sentence of honesty. / If you DO have labels, / a simple cell-type mean of real cells scores 0.98. / At the type level, nothing learned beats a mean. / What this model gives you is: / no label needed, / and a population that looks like cells, not a point.

**[Fallback if the old table stays: say only the ImputeVAE column and the CT-mean column. Do not read the scVI/scANVI numbers; they come from a decode at a fixed library that your own notes retired. "Our model reaches 0.93 on gene correlation and 96 percent classification; the cell-type mean is the ceiling at 1.0; and we keep more within-type variance than any other learned method."]**

## 10. Validation roadmap (90 words, 10:50)

So the model translates. / But is the translation right biologically? / I test it in three tiers.

Tier one: internal fidelity. / On held-out matched data, / does the predicted cell-minus-nucleus shift match the real one?

Tier two: annotation-based plausibility. / Do the shifts follow known gene programs? / Synaptic genes, transcription factors, nuclear-retained RNAs.

Tier three: an orthogonal assay. / An independent measurement of cytoplasmic RNA, from other labs.

Each tier uses something the model never trained on.

## 11. Internal fidelity (90 words, 11:40)

Tier one.

For every gene and every cell type, / I compute the empirical shift: / mean real cell minus mean nucleus. / And the predicted shift: / mean imputed cell minus mean nucleus.

On held-out motor cortex data, / predicted and empirical shifts correlate at 0.95, / with 94 percent sign agreement.

And the size of the correction depends on the cell type. / It is four to five times larger in neurons than in glia.

So the model learned structured differences, / not one average correction.

## 12. Datasets: SynGO and JASPAR (120 words, 12:45)

Tier two uses two gene sets.

SynGO is an expert-curated database of synaptic proteins. / 1,789 genes, with their function and location annotated. / Synaptic genes are translated far from the nucleus, / so they should show large cell-minus-nucleus shifts in neurons, / and small ones in glia. / That is the positive set.

JASPAR gives transcription factors. / 63 of them overlap our gene set. / Transcription factors act in the nucleus. / If the model were just inflating everything in neurons, / TFs would behave like synaptic genes. / If the correction is specific, they should not. / That is the contrast set.

One caveat: / SynGO annotates proteins, not RNA. / It is a proxy.

## 13. SynGO slope (100 words, 13:40)

Here is the SynGO result.

1,275 synaptic genes, / 17 cell types, / about 22 thousand gene-by-type pairs.

Empirical shift on the x-axis, / predicted shift on the y-axis. / Pearson 0.94, / sign agreement 94 percent, / and the slope is 1.0.

The slope is the important number. / It means the magnitude of the shift is calibrated. / Not just the direction.

## 14. SynGO subgroups (70 words, 14:20)

Inside SynGO, / the neuronal shift is larger in every subgroup. / Presynaptic, postsynaptic, active zone, synaptic vesicle.

So this is not one big mixed set driving the result. / It appears in every functional category.

Again, the caveat: / this is protein annotation used as a proxy for RNA.

## 15. SynGO versus TF (90 words, 15:10)

And here is the specificity test.

Synaptic genes show a large neuron-minus-glia contrast. / Transcription factors show a much smaller one. / About two and a half times smaller.

So the model is not applying a generic boost to neurons. / The recovered pattern depends on the gene set / and on the biology.

I also checked the restraint direction, / 45 nuclear-retained genes like Malat1 and Meg3. / All 45 have the correct, negative sign.

## 16. TRAP / RiboTag: the assay (140 words, 16:25)

Tier three is new since the qual. / William suggested ribosome profiling, / and this is it.

TRAP and RiboTag work like this. / You express a tagged ribosomal protein in one cell type, / inside intact tissue. / You pull down the tagged ribosomes / and sequence the mRNA bound to them.

Ribosome-bound mRNA is cytoplasmic by definition. / There is no dissociation, / no nucleus-versus-cell split. / And none of it was in training.

The test: / for each external population, / I map it to a training cell type. / Then I correlate the external profile with three pseudobulks: / raw nuclei, / imputed nuclei, / and real whole cells. / Real cells are the ceiling.

17 populations, 9 datasets, at least 6 labs. / Excitatory neurons, three interneuron types, astrocytes, oligodendrocytes.

## 17. TRAP / RiboTag: result (150 words, 17:50)

Here is the result.

Each row is one external population. / Gray is the raw nucleus. / Blue is the imputed nucleus. / Orange is real whole cells, the ceiling.

In all 17 populations, / the imputed profile lands on the ceiling. / The median gap closed is 0.99. / Every confidence interval excludes zero.

Excitatory, inhibitory and glial populations all behave the same way. / Glia are lower in absolute terms, / but only because the reference itself is lower.

Two things to keep in mind. / Ribosome-bound RNA measures translation, / so this is a test of direction and rank. / And at this pseudobulk level, / a cell-type mean would score the same as the model. / So what this validates is the nucleus-to-cell shift itself. / It is real biology. / What the model adds beyond a mean / is the within-type behaviour, / and that is also where its limit is.

## 18. Limitation: variance (170 words, 19:25)

This is the limitation I named at the qual.

Panel A: / within a cell type, / imputed cells carry about 15 percent of the real-cell variance. / That is more than any deep baseline, / which sit near 3 percent, / and the cell-type mean has none by construction. / But it is still a compression. / Linear regression overshoots, / because it multiplies nuclear noise gene by gene.

Panel B: / it depends on the cell type. / Large neuronal types like L5 IT keep about a quarter. / Rare glial types collapse to their mean.

Panel C is the question Prashant asked in September. / Is this a uniform flattening, / where every gene gets the same low variance? / Or a scale change, / where the same genes stay variable?

The Spearman correlation of per-gene variance, / real versus imputed, / is 0.7 to 0.96 in every type. / Higher than the nucleus itself. / So the genes that vary most in real cells / still vary most after imputation. / It is a scale change.

## 19. What the scale change looks like (200 words, 21:15)

Here is what that looks like, in L5 IT.

Left: the distribution of per-gene variance. / Orange real cells, gray nuclei, blue imputed. / The imputed distribution is shifted down and stretched. / Not squeezed to a spike. / The compression is strongest for sparse genes.

Right: four single genes.

Pcp4 is almost absent from nuclei, / 9 percent detected. / Yet it comes back with the full spread of real cells. / Most of that rides on RNA content.

Rorb is a well-detected marker. / The imputed keeps the main mode / but loses the upper tail. / The nucleus does carry that information, / so this one is a real model limitation.

Gm49164 collapses to one value. / And that is correct. / Its variance in cells comes from a whole-cell module / that does not exist in nuclei.

Xist collapses for a different reason. / Our cell libraries are half male, / our nuclei are all female. / The model outputs the mean of a mixture it cannot see. / That is a design issue, / and sex is the next covariate to add.

So the rule is: / variance survives where the nucleus carries the signal, / and collapses, correctly, where it does not.

## 20. Thank you (120 words, 22:20)

To sum up.

The nucleus-to-cell gap is structured biology, / and a conditional VAE can learn it. / Without labels, it is the best translator we tested. / The shift it learns is correct in direction, / calibrated in size, / specific to gene programs, / and confirmed by an independent assay of cytoplasmic RNA.

Its limit is within-type variance, / and that limit is a scale change, / set by what the nucleus actually carries.

The open question I am working on now: / which nuclei-only result changes once you apply this? / I would like to hear which application you would find convincing.

Thank you. / I am happy to take questions.

---

### Addendum (final versions from the 2026-10-28 session, saved 2026-10-29)

The deck as finished (`cbb-rip-talk-10-28-2026.pptx`, uploaded version 549fc3a1) kept the **full-dataset benchmark table**
on slide 9 and uses the **per-cell-type correction-magnitude figure** (3 panels) on slide 11. The scripts below replace
sections 9 and 11 above for that deck.

#### 9. Benchmark, simplified (150 words, ~1:00) — for the full-dataset table

> This is our first comparison, on the full dataset. Four baselines: cell-type mean, scVI, scANVI, and linear regression.
>
> Three things to see. First, gene correlation: ImputeVAE reaches 0.93. The cell-type mean is 1.0 by definition, so that is the ceiling, not a competitor.
>
> Second, downstream structure: 96 percent of imputed nuclei are classified to the right type, and gene-gene coexpression within a type is 0.93, the best in the table.
>
> Third, variance. Within a cell type, we keep about 16 percent of the real variance. scVI and scANVI keep 3 percent. The mean keeps none. Linear regression overshoots, because it amplifies noise gene by gene. That 16 percent is the limitation I will come back to.
>
> One note: the scVI numbers here are from a fixed decoding library. On a later held-out benchmark they rise to about 0.86, and we are at 0.90. So the lead is real, but smaller than this table suggests.

Shorter closing line if needed: "Later held-out numbers are closer, 0.90 versus 0.86, but the order is the same."

Slide fixes: the header paragraph should define gene correlation, variance ratio and coexpression (not per-cell correlation);
subtitle "Internal comparison, full paired dataset". Do not say "held-out, every method trained on the same data" for this table.
Split-aware label-free alternative (benchmark review §5): gene corr 0.900 / CT-mean 0.981 / scVI 0.857 / scANVI 0.865 / linear 0.759;
classif 96.0 / 100 / 92.2 / 92.9 / 92.3 %; coexpression L5 IT 0.790 / n/a / 0.691 / 0.717 / 0.655;
variance ratio 0.147 / 0 / 0.151 / 0.150 / 2.99; energy distance 7.5 / 25.1 / 9.2 / 9.0 / 12.6 (floor 0.62).

#### 11. Internal fidelity — for the per-cell-type magnitude figure (A: scatter, B: other methods, C: per type)

> Tier one: does the model recover the real cell-minus-nucleus shift?
>
> Panel A. Each dot is one cell type. On the x-axis, the real correction magnitude: the mean absolute difference between cells and nuclei across genes. On the y-axis, what the model predicts. The dots sit on the diagonal. Pearson 0.86 across 17 types.
>
> Two things to notice. Neurons, in blue, sit at the top right; glia, in green, at the bottom left. The real neuron-to-glia ratio is 3.3; the model predicts 4.6. So the model learned that the gap is larger in neurons, not one average correction. The one outlier is L5 ET, which the model over-predicts. That type has only 161 reference cells, so the real-cell mean itself is noisy.
>
> Panel B. The same test for two other learned methods. scGen and scButterfly do not recover the per-type magnitude at all; their correlation across types is near zero or negative. ImputeVAE is the only method that gets the cell-type dependence right.
>
> Panel C. Type by type: black is real, blue is predicted. For most types the two markers overlap; the gap is visible only for L5 ET.
>
> At the gene level the picture is the same: across all gene-by-type pairs the predicted and real shifts correlate 0.95, with 94 percent sign agreement.

Slide fixes: table cells with numbers ("r = 0.86 across 17 types; 0.95 per gene-type pair", "neurons 3 to 5x glia");
subtitle as a statement: "Predicted and real shifts agree across cell types; the gap is larger in neurons."

#### Gene count, if asked

10,134 genes (Config B, `data/training_gene_list_v3.csv`): 25,803 shared genes -> detection > 5 % and mean > 0.01 in
v3 cells (13,658) and v3 nuclei (11,048) -> intersection minus MitoCarta = 10,134. Not 16,439 (per-type filter in the
pre-qual deck, not adopted) and not ~24,700 (old NB runs, 1 % threshold). Human M1: 8,215 orthologs; Bakken 2018: 8,928 present.

#### Host introduction

> Our next speaker is Suchen Zheng, a PhD student in the Computational Biology and Bioinformatics program, working in Mark Gerstein's lab. Suchen works on single-nucleus RNA-seq of the brain, where most data come from nuclei rather than whole cells. The project asks how much of a cell's missing cytoplasmic RNA can be recovered from its nucleus alone, using a conditional variational autoencoder trained on paired cell and nucleus data from mouse cortex. Today's talk is a research-in-progress update on the model, its validation, and what it has taught us about the limits of nuclear data.

Two-sentence version:

> Suchen Zheng is a PhD student in CBB in the Gerstein lab. Suchen's project asks what a nucleus can tell you about its cytoplasm: a deep learning model that translates single-nucleus RNA-seq profiles into whole-cell profiles, and the biology learned along the way.

### Delivery notes
- Slides 2 to 5 are 5 minutes total; do not go over. If you are past 5:30 at slide 6, skip slide 14 later.
- Breath marks "/" are where a native speaker would pause anyway. Taking them makes you sound deliberate, not slow.
- Numbers to say in words: "zero point nine" is fine; "ninety-four percent" for sign agreement.
- Point at the figure when you name a colour (gray, blue, orange). Peers follow the pointer, not the sentence.
- Questions you should expect: why not scGen (needs the label; 0.975 with it); is TRAP translation or abundance (translation; rank test); why 15 % variance (next slide, and the latent is one library-size axis); what about human (zero-shot M1 done, no paired truth).
