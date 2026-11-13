# Meeting Notes — Jun 3, 2026 (with PE)

## Topic: Model validation and biological analysis for VAE imputation

## Results Presented

### CPM Single-Stage cVAE (full dataset)
- Encoder+decoder conditioning, normalized losses, no annealing
- UMAPs look excellent: cell/imputed/cross-decoded all merge well
- Latent space well aligned between cell and nuclei

### Delta Analysis (nuclei vs imputed cells)
- **Top increased genes** (imputation adds relative to nuclei):
  - Ribosomal proteins (RPLs, RPSs), elongation factors, calmodulin, VAMP2
  - Cytoplasmic-enriched — expected and encouraging
- **Top decreased genes** (imputation removes relative to nuclei):
  - MALAT1 (known nucleus-enriched lncRNA — good that it decreases)
  - snoRNAs, lncRNAs
  - GM genes = predicted/placeholder mouse genes, poorly annotated
- **Cell-type-specific genes**: heat shock proteins, RAB signaling, tetraspanins, EV proteins
  - Neurons increase, non-neuronal decrease — PE found this "very promising"

### 20% Subset Runs
- UMAPs showed batch effects — PE hypothesizes low cell counts per type
- Suggested diagnostic: cell count table per cell type at each subset level
- Compromise: try 60-70% subset instead of 20% or 100%

## Action Items (Priority Order)

### 1. Literature search for validation datasets [HIGHEST PRIORITY]
- Mouse nucleus-enriched vs cytoplasm-enriched RNA datasets
- Human cells vs nuclei datasets
- Other tissue/region cells vs nuclei (check Allen Brain Atlas for other regions)
- Same technique (10X), same species (mouse cortical) preferred
- Critical for ISMB poster — people will ask "how do you validate?"

### 2. Gene Ontology analysis on top 50-100 genes
- Use Mus musculus in GO database
- Run on both increased and decreased gene lists
- Expect: lncRNAs in nucleus-decreased, ribosomal/cytoplasmic in nucleus-increased

### 3. Biological literature search
- RNA decay rates: are nuclear-enriched RNAs shorter-lived than cytoplasmic?
- Synaptic-related genes should increase on imputation side
- Look for overrepresentation of cytoplasmic genes in increased set

### 4. Run validation set through delta analysis
- Current delta analysis only on training data
- Need to confirm results hold on validation split

### 5. Quantitative cross-modal metrics (for paper)
- Cross-modal silhouette coefficient (cells vs imputed cells)
- PE: "UMAPs everyone is growing very tired of" — need quantitative metrics
- Don't need every run, but good models must have these

### 6. Cell count diagnostic table
- Per cell type, per modality, at each subset level (20%, 60%, 100%)
- Helps explain why subset models underperform

## Timeline
- ISMB poster: July 15-17, 2026
- Start poster prep: early July
- Next week focus: literature search + GO analysis
- If stuck on finding datasets, reach out to PE — he will help look

## PE's Key Quotes
- "This is really very promising results. Things look great."
- "UMAPs everyone is now growing very tired of... we need quantitative metrics"
- "What that analysis would prove is that if there's particular families of genes [enriched]"
- On heat shock / EV / neuron results: "That is actually very interesting... very cool"
