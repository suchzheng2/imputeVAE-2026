import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import json
import sys

# import scanpy as sc
import anndata as ad
import numpy as np
import json
import sys
import pandas as pds


def plot_umap_after_train_by_celltype_stage3(run_dpath,feature,data_split,epoch,file_path=None):
    # figure_save_folder = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/ABC_umap_May_6"
    print("plotting umap after train by cell type")
    figure_save_folder = os.path.join(run_dpath,"umap")
    os.makedirs(figure_save_folder, exist_ok=True)
    
    output_path = os.path.join(figure_save_folder, f"e{epoch}_{data_split}_{feature}.png")

    if file_path is None:
        file_path = os.path.join(run_dpath, f"epoch_dump/epoch_{epoch}_{data_split}.json")
        
    with open(file_path, 'r') as file:
        data = json.load(file)

    cell_type_name_fn = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cell_list.txt")

    with open(cell_type_name_fn, "r") as file:
        cell_names = [line.strip() for line in file]

    umap_data, cell_type_list, meta_batch_list, modality_list = [], [], [], []

    for cell_type, cell_batches in data.items():
        for batch_idx, batch in cell_batches.items():
            # print(batch.keys())
            print(f"cell_{feature} shape: {len(batch["cell_"+feature])}, {len(batch["cell_"+feature][0])}")
            print(f"nuclei_{feature} shape: {len(batch["nuclei_"+feature])}, {len(batch["nuclei_"+feature][0])}")

            cell_data = batch["cell_"+feature]
            nuclei_data = batch["nuclei_"+feature]
            smaller_length = min(len(cell_data), len(nuclei_data))  
            if len(cell_data) != len(nuclei_data):
                # subset the larger loader to match the smaller loader
                cell_data = cell_data[:smaller_length]
                nuclei_data = nuclei_data[:smaller_length]
                print(f"Subsetted cell_data: {len(cell_data)}")
                print(f"Subsetted nuclei_data: {len(nuclei_data)}")
                print(f"Smaller length: {smaller_length}")
            else:
                print(f"Cell and nuclei data have the same length")
            
            # exit()
            umap_data += cell_data
            modality_list += ["cell"]*len(cell_data)
            cell_type_list +=  batch["cell_type_original"][:smaller_length]
            meta_batch_list += batch["meta_batch_id_original"][:smaller_length]

            umap_data += nuclei_data
            modality_list += ["nuclei"]*len(nuclei_data)
            cell_type_list +=  batch["cell_type_original"][:smaller_length]
            meta_batch_list += batch["meta_batch_id_original"][:smaller_length]

    print(cell_type_list[:10])
    print(meta_batch_list[:10])
    print(modality_list[:10])
    print("Length of cell_type_list:", len(cell_type_list))
    print("Length of data_umap:", len(umap_data))   

    if np.isnan(umap_data).any():
        epoch = epoch - 1
        print("data contain nan")
        return 

    if len(cell_type_list) != len(umap_data):
        raise ValueError("Mismatch between cell type list and data length")

    umap_model = umap.UMAP(random_state=42)
    umap_embedding = umap_model.fit_transform(np.array(umap_data))

    umap_df = pd.DataFrame({
        'UMAP1': umap_embedding[:, 0],
        'UMAP2': umap_embedding[:, 1],
        'CellType': cell_type_list,
        'MetaBatch': meta_batch_list,
        'Modality': modality_list
    })

    fig, axes = plt.subplots(1, 3, figsize=(32, 10))

    cell_types = sorted(umap_df['CellType'].unique())
    n_cell_types = len(cell_types)

    # Create a colormap with a unique color for each cell type
    colors = cm.get_cmap('tab20', n_cell_types)

    for idx, cell_type in enumerate(cell_types):
        subset = umap_df[umap_df['CellType'] == cell_type]
        axes[0].scatter(subset['UMAP1'], subset['UMAP2'], 
                        label=cell_type, 
                        s=3, 
                        color=colors(idx))

    axes[0].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].set_title('UMAP Colored by Cell Type')
    axes[0].set_xlabel('UMAP1')
    axes[0].set_ylabel('UMAP2')

    # UMAP plot colored by meta-batch ID
    n_meta_batch = umap_df['MetaBatch'].nunique()
    # Create a colormap with a unique color for each cell type
    colors = cm.get_cmap('tab20', n_meta_batch)
    for idx, meta_batch in enumerate(sorted(umap_df['MetaBatch'].unique())):
        subset = umap_df[umap_df['MetaBatch'] == meta_batch]
        axes[1].scatter(subset['UMAP1'], subset['UMAP2'], label=meta_batch, s=3,color=colors(idx))
        
    axes[1].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].set_title('UMAP Colored by Meta-batch ID')
    axes[1].set_xlabel('UMAP1')
    axes[1].set_ylabel('UMAP2')

    # UMAP plot colored by modality
    n_modality = umap_df['Modality'].nunique()
    # Create a colormap with a unique color for each cell type
    colors = cm.get_cmap('tab20', n_modality)
    for idx, mod in enumerate(sorted(umap_df['Modality'].unique())):
        subset = umap_df[umap_df['Modality'] == mod]
        axes[2].scatter(subset['UMAP1'], subset['UMAP2'], label=mod, s=3,color=colors(idx))

    axes[2].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[2].set_title('UMAP Colored by Modality')
    axes[2].set_xlabel('UMAP1')
    axes[2].set_ylabel('UMAP2')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Two-panel UMAP plot saved to {output_path}")

def plot_umap_after_train_by_celltype(run_dpath,feature,data_split,epoch,file_path=None):
    # figure_save_folder = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/ABC_umap_May_6"
    print("plotting umap after train by cell type")
    figure_save_folder = os.path.join(run_dpath,"umap")
    os.makedirs(figure_save_folder, exist_ok=True)
    
    output_path = os.path.join(figure_save_folder, f"e{epoch}_{data_split}_{feature}.png")

    if file_path is None:
        file_path = os.path.join(run_dpath, f"epoch_dump/epoch_{epoch}_{data_split}.json")
        
    with open(file_path, 'r') as file:
        data = json.load(file)

    cell_type_name_fn = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cell_list.txt")

    with open(cell_type_name_fn, "r") as file:
        cell_names = [line.strip() for line in file]

    umap_data, cell_type_list, meta_batch_list = [], [], []

    for cell_type, cell_batches in data.items():
        for batch_idx, batch in cell_batches.items():
            print(batch.keys())
            umap_data += batch[feature]
            cell_type_list +=  batch["cell_type_original"]
            meta_batch_list += batch["meta_batch_id_original"]

    print(cell_type_list[:10])
    print(meta_batch_list[:10])

    print("Length of cell_type_list:", len(cell_type_list))
    print("Length of data_umap:", len(umap_data))   

    if np.isnan(umap_data).any():
        epoch = epoch - 1
        print("data contain nan")
        return 

    if len(cell_type_list) != len(umap_data):
        raise ValueError("Mismatch between cell type list and data length")

    umap_model = umap.UMAP(random_state=42)
    umap_embedding = umap_model.fit_transform(np.array(umap_data))

    umap_df = pd.DataFrame({
        'UMAP1': umap_embedding[:, 0],
        'UMAP2': umap_embedding[:, 1],
        'CellType': cell_type_list,
        'MetaBatch': meta_batch_list
    })

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    cell_types = sorted(umap_df['CellType'].unique())
    n_cell_types = len(cell_types)

    # Create a colormap with a unique color for each cell type
    colors = cm.get_cmap('tab20', n_cell_types)

    for idx, cell_type in enumerate(cell_types):
        subset = umap_df[umap_df['CellType'] == cell_type]
        axes[0].scatter(subset['UMAP1'], subset['UMAP2'], 
                        label=cell_type, 
                        s=3, 
                        color=colors(idx))

    axes[0].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].set_title('UMAP Colored by Cell Type')
    axes[0].set_xlabel('UMAP1')
    axes[0].set_ylabel('UMAP2')

    # UMAP plot colored by meta-batch ID
    n_meta_batch = umap_df['MetaBatch'].nunique()
    # Create a colormap with a unique color for each cell type
    colors = cm.get_cmap('tab20', n_meta_batch)

    for idx, meta_batch in enumerate(sorted(umap_df['MetaBatch'].unique())):
        subset = umap_df[umap_df['MetaBatch'] == meta_batch]
        axes[1].scatter(subset['UMAP1'], subset['UMAP2'], label=meta_batch, s=3,color=colors(idx))
        
    axes[1].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].set_title('UMAP Colored by Meta-batch ID')
    axes[1].set_xlabel('UMAP1')
    axes[1].set_ylabel('UMAP2')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Two-panel UMAP plot saved to {output_path}")

def plot_umap_for_both_modaility(output_dir, nuclei_file, cell_file, feature):
    figure_save_folder = os.path.join(output_dir,"umap")
    os.makedirs(figure_save_folder, exist_ok=True)

    output_path = os.path.join(figure_save_folder, f"cell_nulcei_{feature}.png")

    with open(nuclei_file, 'r') as file:
        nuclei_data = json.load(file)

    with open(cell_file, 'r') as file:
        cell_data = json.load(file)
    
    umap_data, cell_type_list, meta_batch_list, modality_list = [], [], [], []

    for cell_batch in cell_data.values():
        for batch in cell_batch.values():
            umap_data += batch[feature]
            cell_type_list +=  batch["cell_type_original"]
            meta_batch_list += batch["meta_batch_id_original"]  
            modality_list += ["cell"]*len(batch[feature])
    for nuclei_batch in nuclei_data.values():
        for batch in nuclei_batch.values():
            umap_data += batch[feature]
            cell_type_list +=  batch["cell_type_original"]
            meta_batch_list += batch["meta_batch_id_original"]
            modality_list += ["nuclei"]*len(batch[feature])

    print(cell_type_list[:10])
    print(meta_batch_list[:10])
    print(modality_list[:10])

    print("Length of cell_type_list:", len(cell_type_list))
    print("Length of data_umap:", len(umap_data))       

    if np.isnan(umap_data).any():
        epoch = epoch - 1
        print("data contain nan")
        return 
    
    if len(cell_type_list) != len(umap_data):
        raise ValueError("Mismatch between cell type list and data length") 

    umap_model = umap.UMAP(random_state=42)
    umap_embedding = umap_model.fit_transform(np.array(umap_data))

    umap_df = pd.DataFrame({
        'UMAP1': umap_embedding[:, 0],
        'UMAP2': umap_embedding[:, 1],
        'CellType': cell_type_list,
        'MetaBatch': meta_batch_list,
        'Modality': modality_list
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(32, 10))

    cell_types = sorted(umap_df['CellType'].unique())
    n_cell_types = len(cell_types)

    # Create a colormap with a unique color for each cell type
    colors = cm.get_cmap('tab20', n_cell_types)

    for idx, cell_type in enumerate(cell_types):
        subset = umap_df[umap_df['CellType'] == cell_type]
        axes[0].scatter(subset['UMAP1'], subset['UMAP2'], 
                        label=cell_type, 
                        s=3, 
                        color=colors(idx))

    axes[0].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].set_title('UMAP Colored by Cell Type')
    axes[0].set_xlabel('UMAP1')
    axes[0].set_ylabel('UMAP2')

    # UMAP plot colored by meta-batch ID
    n_meta_batch = umap_df['MetaBatch'].nunique()
    # Create a colormap with a unique color for each cell type
    colors = cm.get_cmap('tab20', n_meta_batch)

    for idx, meta_batch in enumerate(sorted(umap_df['MetaBatch'].unique())):
        subset = umap_df[umap_df['MetaBatch'] == meta_batch]
        axes[1].scatter(subset['UMAP1'], subset['UMAP2'], label=meta_batch, s=3,color=colors(idx))
        
    axes[1].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].set_title('UMAP Colored by Meta-batch ID')
    axes[1].set_xlabel('UMAP1')
    axes[1].set_ylabel('UMAP2')

    # UMAP plot colored by modality
    n_modality = umap_df['Modality'].nunique()
    # Create a colormap with a unique color for each cell type  
    colors = cm.get_cmap('tab20', n_modality)

    for idx, mod in enumerate(sorted(umap_df['Modality'].unique())):
        subset = umap_df[umap_df['Modality'] == mod]
        axes[2].scatter(subset['UMAP1'], subset['UMAP2'], label=mod, s=3,color=colors(idx))
        
    axes[2].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[2].set_title('UMAP Colored by Modality')
    axes[2].set_xlabel('UMAP1')
    axes[2].set_ylabel('UMAP2')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Three-panel UMAP plot saved to {output_path}")  
    
    
def plot_umap_after_finetune(run_dpath,feature,data_split,epoch,args, file_path=None):
    # figure_save_folder = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/ABC_umap_May_6"
    figure_save_folder = os.path.join(run_dpath,"umap")
    os.makedirs(figure_save_folder, exist_ok=True)
    
    output_path = os.path.join(figure_save_folder, f"e{epoch}_{data_split}_{feature}.png")

    if file_path is None:
        file_path = os.path.join(run_dpath, f"epoch_dump/epoch_{epoch}_{data_split}.json")
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    umap_data, cell_type_list, meta_batch_list, modality_list = [], [], [], []

    for cell_batch in data.values():
        for batch in cell_batch.values():
            umap_data += batch[feature]
            cell_type_list +=  batch["cell_type_original"]
            meta_batch_list += batch["meta_batch_id_original"]
            modality_list += ["nuclei"]*len(batch[feature])

    #load cell model validation results:
    try:
        cell_train_data_path = sorted([
            f for f in os.listdir(os.path.join(args.cell_model_run, "epoch_dump"))
            if f.startswith("epoch_") and f.endswith("_train.json")
        ])[-1]
    except:
        cell_train_data_path = os.path.join(args.cell_model_run, "epoch_dump", "valid.json")
    
    cell_train_data = json.load(open(os.path.join(args.cell_model_run, "epoch_dump", cell_train_data_path)))
    
    # add cell data to umap_data
    if args.was_train_by_cell_type:
        for cell_type, cell_batches in cell_train_data.items():
            for batch_idx, batch in cell_batches.items():
                umap_data += batch[feature]
                cell_type_list +=  batch["cell_type_original"]
                meta_batch_list += batch["meta_batch_id_original"]
                modality_list += ["cell"]*len(batch[feature])
    else:   
        for batch in cell_train_data.values():
            umap_data += batch[feature]
            cell_type_list +=  batch["cell_type_original"]
            meta_batch_list += batch["meta_batch_id_original"]
            modality_list += ["cell"]*len(batch[feature])

    # print(cell_type_list[:10])
    # print(meta_batch_list[:10])

    # print("Length of cell_type_list:", len(cell_type_list))
    # print("Length of data_umap:", len(umap_data))   
    
    if np.isnan(umap_data).any():
        epoch = epoch - 1
        print("data contain nan")
        return 
    
    if len(cell_type_list) != len(umap_data):
        raise ValueError("Mismatch between cell type list and data length") 

    umap_model = umap.UMAP(random_state=42)
    umap_embedding = umap_model.fit_transform(np.array(umap_data))

    umap_df = pd.DataFrame({
        'UMAP1': umap_embedding[:, 0],
        'UMAP2': umap_embedding[:, 1],
        'CellType': cell_type_list, 
        'MetaBatch': meta_batch_list,
        'Modality': modality_list
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(32, 10))

    cell_types = sorted(umap_df['CellType'].unique())
    n_cell_types = len(cell_types)

    # Create a colormap with a unique color for each cell type
    colors = cm.get_cmap('tab20', n_cell_types)

    for idx, cell_type in enumerate(cell_types):
        subset = umap_df[umap_df['CellType'] == cell_type]
        axes[0].scatter(subset['UMAP1'], subset['UMAP2'], 
                        label=cell_type, 
                        s=3, 
                        color=colors(idx))

    axes[0].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].set_title('UMAP Colored by Cell Type')
    axes[0].set_xlabel('UMAP1')
    axes[0].set_ylabel('UMAP2')

    # UMAP plot colored by meta-batch ID
    n_meta_batch = umap_df['MetaBatch'].nunique()
    # Create a colormap with a unique color for each cell type
    colors = cm.get_cmap('tab20', n_meta_batch)

    for idx, meta_batch in enumerate(sorted(umap_df['MetaBatch'].unique())):
        subset = umap_df[umap_df['MetaBatch'] == meta_batch]
        axes[1].scatter(subset['UMAP1'], subset['UMAP2'], label=meta_batch, s=3,color=colors(idx))
        
    axes[1].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].set_title('UMAP Colored by Meta-batch ID')
    axes[1].set_xlabel('UMAP1')
    axes[1].set_ylabel('UMAP2')

    # UMAP plot colored by modality
    n_modality = umap_df['Modality'].nunique()
    # Create a colormap with a unique color for each cell type
    colors = cm.get_cmap('tab20', n_modality)

    for idx, mod in enumerate(sorted(umap_df['Modality'].unique())):
        subset = umap_df[umap_df['Modality'] == mod]
        axes[2].scatter(subset['UMAP1'], subset['UMAP2'], label=mod, s=3,color=colors(idx))

    axes[2].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[2].set_title('UMAP Colored by Modality')
    axes[2].set_xlabel('UMAP1')
    axes[2].set_ylabel('UMAP2')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Three-panel UMAP plot saved to {output_path}")
    
    

def plot_umap_after_train(run_dpath,feature,data_split,epoch):
    # figure_save_folder = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/ABC_umap_May_6"
    figure_save_folder = os.path.join(run_dpath,"umap")
    os.makedirs(figure_save_folder, exist_ok=True)

    # run = sys.argv[1]
    # modality = sys.argv[2]
    # feature = sys.argv[3]
    # data_split = sys.argv[4]

    # start_epoch = 19

    # epoch = start_epoch
    
    
    output_path = os.path.join(figure_save_folder, f"e{epoch}_{data_split}_{feature}.png")

    file_path = os.path.join(run_dpath, f"epoch_dump/epoch_{epoch}_{data_split}.json")
    with open(file_path, 'r') as file:
        data = json.load(file)

    cell_type_name_fn = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cell_list.txt")
    with open(cell_type_name_fn, "r") as file:
        cell_names = [line.strip() for line in file]

    umap_data, cell_type_list, meta_batch_list = [], [], []

    for batch in data.values():
        umap_data += batch[feature]
        cell_type_list +=  batch["cell_type_original"]
        # cell_type_list += [cell_names[id] for id in cell_types]
        meta_batch_list += batch["meta_batch_id_original"]

    print(cell_type_list[:10])
    print(meta_batch_list[:10])

    print("Length of cell_type_list:", len(cell_type_list))
    print("Length of data_umap:", len(umap_data))   

    if np.isnan(umap_data).any():
        epoch = epoch - 1
        print("data contain nan")
        return 

    if len(cell_type_list) != len(umap_data):
        raise ValueError("Mismatch between cell type list and data length")

    umap_model = umap.UMAP(random_state=42)
    umap_embedding = umap_model.fit_transform(np.array(umap_data))

    umap_df = pd.DataFrame({
        'UMAP1': umap_embedding[:, 0],
        'UMAP2': umap_embedding[:, 1],
        'CellType': cell_type_list,
        'MetaBatch': meta_batch_list
    })

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    cell_types = sorted(umap_df['CellType'].unique())
    n_cell_types = len(cell_types)

    # Create a colormap with a unique color for each cell type
    colors = cm.get_cmap('tab20', n_cell_types)

    for idx, cell_type in enumerate(cell_types):
        subset = umap_df[umap_df['CellType'] == cell_type]
        axes[0].scatter(subset['UMAP1'], subset['UMAP2'], 
                        label=cell_type, 
                        s=3, 
                        color=colors(idx))

    axes[0].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].set_title('UMAP Colored by Cell Type')
    axes[0].set_xlabel('UMAP1')
    axes[0].set_ylabel('UMAP2')

    # UMAP plot colored by meta-batch ID
    n_meta_batch = umap_df['MetaBatch'].nunique()
    # Create a colormap with a unique color for each cell type
    colors = cm.get_cmap('tab20', n_meta_batch)

    for idx, meta_batch in enumerate(sorted(umap_df['MetaBatch'].unique())):
        subset = umap_df[umap_df['MetaBatch'] == meta_batch]
        axes[1].scatter(subset['UMAP1'], subset['UMAP2'], label=meta_batch, s=3,color=colors(idx))
        
    axes[1].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].set_title('UMAP Colored by Meta-batch ID')
    axes[1].set_xlabel('UMAP1')
    axes[1].set_ylabel('UMAP2')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Two-panel UMAP plot saved to {output_path}")
    

def plot_umap_after_test(run_dpath,feature,data_split):
    # figure_save_folder = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/ABC_umap_May_6"
    figure_save_folder = os.path.join(run_dpath,"umap")
    os.makedirs(figure_save_folder, exist_ok=True)

    # run = sys.argv[1]
    # modality = sys.argv[2]
    # feature = sys.argv[3]
    # data_split = sys.argv[4]
    # start_epoch = 19
    # epoch = start_epoch
    
    output_path = os.path.join(figure_save_folder, f"{data_split}_{feature}.png")

    file_path = os.path.join(run_dpath, "test_results.json")
    
    with open(file_path, 'r') as file:
        data = json.load(file)

    cell_type_name_fn = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cell_list.txt")
    with open(cell_type_name_fn, "r") as file:
        cell_names = [line.strip() for line in file]

    umap_data, cell_type_list, meta_batch_list = [], [], []

    for batch in data.values():
        umap_data += batch[feature]
        cell_type_list +=  batch["cell_type_original"]
        # cell_type_list += [cell_names[id] for id in cell_types]
        meta_batch_list += batch["meta_batch_id_original"]

    print(cell_type_list[:10])
    print(meta_batch_list[:10])

    print("Length of cell_type_list:", len(cell_type_list))
    print("Length of data_umap:", len(umap_data))   

    if np.isnan(umap_data).any():
        epoch = epoch - 1
        print("data contain nan")
        return 

    if len(cell_type_list) != len(umap_data):
        raise ValueError("Mismatch between cell type list and data length")

    umap_model = umap.UMAP(random_state=42)
    umap_embedding = umap_model.fit_transform(np.array(umap_data))

    umap_df = pd.DataFrame({
        'UMAP1': umap_embedding[:, 0],
        'UMAP2': umap_embedding[:, 1],
        'CellType': cell_type_list,
        'MetaBatch': meta_batch_list
    })

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    cell_types = sorted(umap_df['CellType'].unique())
    n_cell_types = len(cell_types)

    # Create a colormap with a unique color for each cell type
    colors = cm.get_cmap('tab20', n_cell_types)

    for idx, cell_type in enumerate(cell_types):
        subset = umap_df[umap_df['CellType'] == cell_type]
        axes[0].scatter(subset['UMAP1'], subset['UMAP2'], 
                        label=cell_type, 
                        s=3, 
                        color=colors(idx))

    axes[0].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].set_title('UMAP Colored by Cell Type')
    axes[0].set_xlabel('UMAP1')
    axes[0].set_ylabel('UMAP2')

    # UMAP plot colored by meta-batch ID
    n_meta_batch = umap_df['MetaBatch'].nunique()
    # Create a colormap with a unique color for each cell type
    colors = cm.get_cmap('tab20', n_meta_batch)

    for idx, meta_batch in enumerate(sorted(umap_df['MetaBatch'].unique())):
        subset = umap_df[umap_df['MetaBatch'] == meta_batch]
        axes[1].scatter(subset['UMAP1'], subset['UMAP2'], label=meta_batch, s=3,color=colors(idx))
        
    axes[1].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].set_title('UMAP Colored by Meta-batch ID')
    axes[1].set_xlabel('UMAP1')
    axes[1].set_ylabel('UMAP2')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Two-panel UMAP plot saved to {output_path}")


def plot_umap_after_test_mixed_modality(run_dpath,feature,data_split):
    # figure_save_folder = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/ABC_umap_May_6"
    figure_save_folder = os.path.join(run_dpath,"umap")
    os.makedirs(figure_save_folder, exist_ok=True)

    output_path = os.path.join(figure_save_folder, f"{data_split}_{feature}.png")

    file_path = os.path.join(run_dpath, "test_results.json")
    
    with open(file_path, 'r') as file:
        data = json.load(file)

    cell_type_name_fn = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cell_list.txt")
    with open(cell_type_name_fn, "r") as file:
        cell_names = [line.strip() for line in file]

    umap_data, cell_type_list, meta_batch_list = [], [], []

    for batch in data.values():
        umap_data += batch[feature]
        cell_type_list +=  batch["cell_type_original"]
        # cell_type_list += [cell_names[id] for id in cell_types]
        meta_batch_list += batch["meta_batch_id_original"]

    print(cell_type_list[:10])
    print(meta_batch_list[:10])

    print("Length of cell_type_list:", len(cell_type_list))
    print("Length of data_umap:", len(umap_data))   
    print("Length of umap_data:", len(umap_data))
    print("Sample data (first 5):", umap_data[:5])      

    modality_list = [batch[-3] for batch in meta_batch_list]
    # print("modality list:",modality_list[10:])
    umap_data = np.array(umap_data)
    if np.isnan(umap_data).any():
        epoch = epoch - 1
        print("data contain nan")
        return 

    if len(cell_type_list) != len(umap_data):
        raise ValueError("Mismatch between cell type list and data length")

    umap_model = umap.UMAP(random_state=42)
    umap_embedding = umap_model.fit_transform(np.array(umap_data))

    umap_df = pd.DataFrame({
        'UMAP1': umap_embedding[:, 0],
        'UMAP2': umap_embedding[:, 1],
        'CellType': cell_type_list,
        'MetaBatch': meta_batch_list,
        'Modality': modality_list
    })

    fig, axes = plt.subplots(1, 3, figsize=(36, 10))

    cell_types = sorted(umap_df['CellType'].unique())
    n_cell_types = len(cell_types)

    # Create a colormap with a unique color for each cell type
    colors = cm.get_cmap('tab20', n_cell_types)

    for idx, cell_type in enumerate(cell_types):
        subset = umap_df[umap_df['CellType'] == cell_type]
        axes[0].scatter(subset['UMAP1'], subset['UMAP2'], 
                        label=cell_type, 
                        s=3, 
                        color=colors(idx))

    axes[0].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].set_title('UMAP Colored by Cell Type')
    axes[0].set_xlabel('UMAP1')
    axes[0].set_ylabel('UMAP2')

    # UMAP plot colored by meta-batch ID
    n_meta_batch = umap_df['MetaBatch'].nunique()
    # Create a colormap with a unique color for each cell type
    colors = cm.get_cmap('tab20', n_meta_batch)

    for idx, meta_batch in enumerate(sorted(umap_df['MetaBatch'].unique())):
        subset = umap_df[umap_df['MetaBatch'] == meta_batch]
        axes[1].scatter(subset['UMAP1'], subset['UMAP2'], label=meta_batch, s=3,color=colors(idx))
        
    axes[1].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].set_title('UMAP Colored by Meta-batch ID')
    axes[1].set_xlabel('UMAP1')
    axes[1].set_ylabel('UMAP2')


    # UMAP plot colored by meta-batch ID
    n_modality = umap_df['Modality'].nunique()
    # Create a colormap with a unique color for each cell type
    colors = cm.get_cmap('tab20', n_modality)

    for idx, mod in enumerate(sorted(umap_df['Modality'].unique())):
        subset = umap_df[umap_df['Modality'] == mod]
        if mod == 'n':
            label = "Nuclei"
        if mod == 'c':
            label = "Cell"
        axes[2].scatter(subset['UMAP1'], subset['UMAP2'], label=label, s=3,color=colors(idx))
        
    axes[2].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[2].set_title('UMAP Colored by Modality')
    axes[2].set_xlabel('UMAP1')
    axes[2].set_ylabel('UMAP2')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Three-panel UMAP plot saved to {output_path}")


# ============================================================
# Conditioned decoder UMAP plots
# ============================================================

def _plot_2panel_umap(umap_data, cell_type_list, batch_list, output_path, title_suffix=""):
    """2-panel UMAP: cell type + batch. For single-modality plots."""
    if np.isnan(umap_data).any():
        print(f"  Skipping UMAP (NaN in data): {output_path}")
        return

    umap_embedding = umap.UMAP(random_state=42).fit_transform(np.array(umap_data))

    umap_df = pd.DataFrame({
        'UMAP1': umap_embedding[:, 0],
        'UMAP2': umap_embedding[:, 1],
        'CellType': cell_type_list,
        'Batch': batch_list,
    })

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    cell_types = sorted(umap_df['CellType'].unique())
    cmap = cm.get_cmap('tab20', len(cell_types))
    for idx, ct in enumerate(cell_types):
        sub = umap_df[umap_df['CellType'] == ct]
        axes[0].scatter(sub['UMAP1'], sub['UMAP2'], label=ct, s=3, color=cmap(idx))
    axes[0].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].set_title(f'Cell Type{title_suffix}')
    axes[0].set_xlabel('UMAP1')
    axes[0].set_ylabel('UMAP2')

    batches = sorted(umap_df['Batch'].unique())
    cmap_b = cm.get_cmap('tab20', len(batches))
    for idx, b in enumerate(batches):
        sub = umap_df[umap_df['Batch'] == b]
        axes[1].scatter(sub['UMAP1'], sub['UMAP2'], label=b, s=3, color=cmap_b(idx))
    axes[1].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].set_title(f'Batch{title_suffix}')
    axes[1].set_xlabel('UMAP1')
    axes[1].set_ylabel('UMAP2')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  2-panel UMAP saved to {output_path}")


def _plot_3panel_umap(umap_data, cell_type_list, batch_list, modality_list, output_path, title_suffix=""):
    """3-panel UMAP: cell type + batch + modality. For multi-modality plots."""
    if np.isnan(umap_data).any():
        print(f"  Skipping UMAP (NaN in data): {output_path}")
        return

    umap_embedding = umap.UMAP(random_state=42).fit_transform(np.array(umap_data))

    umap_df = pd.DataFrame({
        'UMAP1': umap_embedding[:, 0],
        'UMAP2': umap_embedding[:, 1],
        'CellType': cell_type_list,
        'Batch': batch_list,
        'Modality': modality_list,
    })

    fig, axes = plt.subplots(1, 3, figsize=(32, 10))

    cell_types = sorted(umap_df['CellType'].unique())
    cmap = cm.get_cmap('tab20', len(cell_types))
    for idx, ct in enumerate(cell_types):
        sub = umap_df[umap_df['CellType'] == ct]
        axes[0].scatter(sub['UMAP1'], sub['UMAP2'], label=ct, s=3, color=cmap(idx))
    axes[0].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].set_title(f'Cell Type{title_suffix}')
    axes[0].set_xlabel('UMAP1')
    axes[0].set_ylabel('UMAP2')

    batches = sorted(umap_df['Batch'].unique())
    cmap_b = cm.get_cmap('tab20', len(batches))
    for idx, b in enumerate(batches):
        sub = umap_df[umap_df['Batch'] == b]
        axes[1].scatter(sub['UMAP1'], sub['UMAP2'], label=b, s=3, color=cmap_b(idx))
    axes[1].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].set_title(f'Batch{title_suffix}')
    axes[1].set_xlabel('UMAP1')
    axes[1].set_ylabel('UMAP2')

    modalities = sorted(umap_df['Modality'].unique())
    cmap_m = cm.get_cmap('tab10', len(modalities))
    for idx, mod in enumerate(modalities):
        sub = umap_df[umap_df['Modality'] == mod]
        axes[2].scatter(sub['UMAP1'], sub['UMAP2'], label=mod, s=3, color=cmap_m(idx))
    axes[2].legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[2].set_title(f'Modality{title_suffix}')
    axes[2].set_xlabel('UMAP1')
    axes[2].set_ylabel('UMAP2')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  3-panel UMAP saved to {output_path}")


def plot_cond_umaps(run_dpath, data_split, epoch, file_path=None):
    """Generate all 5 conditioned-decoder UMAP plots for a given epoch.

    Produces:
      umap/{data_split}/latent/e{epoch}.png          — cell + nuclei encoded (3-panel)
      umap/{data_split}/decoded_cell/e{epoch}.png     — decoded cell only (2-panel)
      umap/{data_split}/decoded_nuclei/e{epoch}.png   — decoded nuclei only (2-panel)
      umap/{data_split}/cross_decoded/e{epoch}.png    — cross-decoded nuclei only (2-panel)
      umap/{data_split}/cell_vs_cross/e{epoch}.png    — decoded cell + cross-decoded nuclei (3-panel)
    """
    if file_path is None:
        file_path = os.path.join(run_dpath, f"epoch_dump/epoch_{epoch}_{data_split}.json")

    with open(file_path, 'r') as f:
        data = json.load(f)

    # Collect data from all cell types and batches
    cell_encoded_all, nuclei_encoded_all = [], []
    cell_decoded_all, nuclei_decoded_all = [], []
    cross_decoded_all = []
    cell_ct_all, nuclei_ct_all = [], []
    cell_batch_all, nuclei_batch_all = [], []

    for cell_type, cell_batches in data.items():
        for batch_idx, batch in cell_batches.items():
            # Determine sizes and balance cell vs nuclei
            n_cell = len(batch["cell_encoded"])
            n_nuclei = len(batch["nuclei_encoded"])
            n = min(n_cell, n_nuclei)

            # Cell type labels (same for both modalities within a batch)
            cell_types_here = batch["cell_type_original"][:n]

            # Batch IDs — use per-modality keys
            nuclei_bids = batch.get("nuclei_meta_batch_id_original",
                                    batch.get("meta_batch_id_original", []))[:n]
            cell_bids = batch.get("cell_meta_batch_id_original", nuclei_bids)[:n]

            # Encoded
            cell_encoded_all += batch["cell_encoded"][:n]
            nuclei_encoded_all += batch["nuclei_encoded"][:n]

            # Decoded
            cell_decoded_all += batch["cell_decoded"][:n]
            nuclei_decoded_all += batch["nuclei_decoded"][:n]

            # Cross-decoded (nuclei → cell embedding)
            if "cross_decoded" in batch and batch["cross_decoded"]:
                cross_decoded_all += batch["cross_decoded"][:n]

            # Metadata
            cell_ct_all += cell_types_here
            nuclei_ct_all += cell_types_here
            cell_batch_all += cell_bids
            nuclei_batch_all += nuclei_bids

    umap_base = os.path.join(run_dpath, "umap", data_split)

    # 1) Latent: cell + nuclei encoded (3-panel with modality)
    print(f"  Cond UMAP: latent (epoch {epoch}, {data_split})")
    _plot_3panel_umap(
        umap_data=cell_encoded_all + nuclei_encoded_all,
        cell_type_list=cell_ct_all + nuclei_ct_all,
        batch_list=cell_batch_all + nuclei_batch_all,
        modality_list=["cell"] * len(cell_ct_all) + ["nuclei"] * len(nuclei_ct_all),
        output_path=os.path.join(umap_base, "latent", f"e{epoch}.png"),
    )

    # 2) Decoded cell only (2-panel)
    print(f"  Cond UMAP: decoded_cell (epoch {epoch}, {data_split})")
    _plot_2panel_umap(
        umap_data=cell_decoded_all,
        cell_type_list=cell_ct_all,
        batch_list=cell_batch_all,
        output_path=os.path.join(umap_base, "decoded_cell", f"e{epoch}.png"),
        title_suffix=" (decoded cell)",
    )

    # 3) Decoded nuclei only (2-panel)
    print(f"  Cond UMAP: decoded_nuclei (epoch {epoch}, {data_split})")
    _plot_2panel_umap(
        umap_data=nuclei_decoded_all,
        cell_type_list=nuclei_ct_all,
        batch_list=nuclei_batch_all,
        output_path=os.path.join(umap_base, "decoded_nuclei", f"e{epoch}.png"),
        title_suffix=" (decoded nuclei)",
    )

    # 4) Cross-decoded only (2-panel)
    if cross_decoded_all:
        print(f"  Cond UMAP: cross_decoded (epoch {epoch}, {data_split})")
        _plot_2panel_umap(
            umap_data=cross_decoded_all,
            cell_type_list=nuclei_ct_all,
            batch_list=nuclei_batch_all,
            output_path=os.path.join(umap_base, "cross_decoded", f"e{epoch}.png"),
            title_suffix=" (cross-decoded nuclei)",
        )

        # 5) Cell + cross-decoded nuclei (3-panel with modality)
        print(f"  Cond UMAP: cell_vs_cross (epoch {epoch}, {data_split})")
        _plot_3panel_umap(
            umap_data=cell_decoded_all + cross_decoded_all,
            cell_type_list=cell_ct_all + nuclei_ct_all,
            batch_list=cell_batch_all + nuclei_batch_all,
            modality_list=["cell"] * len(cell_ct_all) + ["cross-decoded nuclei"] * len(nuclei_ct_all),
            output_path=os.path.join(umap_base, "cell_vs_cross", f"e{epoch}.png"),
        )
    else:
        print(f"  Cond UMAP: no cross_decoded data in dump, skipping plots 4 & 5")


