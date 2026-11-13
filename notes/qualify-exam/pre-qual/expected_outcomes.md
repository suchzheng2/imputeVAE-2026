# Expected Outcomes for Proposed Experiments

Preparation notes for committee Q&A — "what do you expect to find?"

## 2a. Cross-dataset generalization (mouse cortex, Ding et al.)

- Expect good transfer on 10X v2 (similar protocol to training data, different lab)
- Expect degraded performance on Smart-seq2/Drop-seq (different protocol characteristics)
- This establishes baseline: how much does technical variation matter before we add species differences?

## 2b. Cross-species transfer (mouse → macaque)

- **Direct transfer**: expect degraded but above-random performance — core RNA localization biology (cytoplasmic vs nuclear) is conserved, but magnitude differs
- **Fine-tuning**: expect best results — mouse pretraining provides the structure, macaque data adjusts the scale
- **Retrain from scratch**: expect overfitting given the small SMART-Seq dataset — fine-tuning should beat this, which would demonstrate the value of transfer learning

### Q&A: Region confound (cortex training → dLGN testing)

**If committee asks**: "Isn't species confounded with brain region since Bakken data is dLGN not cortex?"

**Answer**: We acknowledge this confounds species and region. However:
1. Bakken et al. is the only paired primate brain dataset available — this is a data constraint, not a design choice
2. Experiment 2a establishes cross-dataset generalization on cortex, giving a baseline
3. The three-strategy comparison (direct / fine-tune / retrain) is informative regardless — if fine-tuning beats direct transfer, we know adaptation is needed, whether for species, region, or both
4. The practical conclusion (whether transfer learning helps) still holds even with the confound
5. The biological interpretation (which specific factor drives the gap) is more nuanced and would require additional paired datasets from other regions to fully disentangle

## 2c. Human application

- **Cytoplasmic RNA recovery**: expect Fos, Arc, ribosomal genes to increase — most confident prediction since these are universally cytoplasmic
- **Cell-type markers**: expect modest improvement — markers already detected in nuclei should sharpen
- **PsychENCODE DE**: hardest to predict — if the model works, some DE genes missed in raw nuclei may appear, but could also introduce false positives

## Failure modes

- Expect neurons to be harder than glia (more cytoplasmic RNA to recover)
- Expect species-specific genes (e.g., primate-specific lncRNAs) to fail — this is the biological insight

## Gene list considerations

When implementing Aim 2, characterize gene loss at each step:
1. Mouse overlap filtering: 16,439 genes kept out of ~30K+ total. What was dropped?
2. Mouse → macaque orthologs: how many of 16,439 have 1:1 orthologs?
3. Mouse → human orthologs: same question

For each step, check whether known marker genes (cell-type markers, immediate early genes like Fos/Arc, disease-relevant genes from PsychENCODE) survive or get filtered out.
