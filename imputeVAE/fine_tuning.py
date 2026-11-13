import os
import time
import json
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from .utils import device, setup_device, save_epoch_save_dict, load_args_from_json
from .models import VAE_JSD
from .data import create_dataloaders
from .testing import load_vae_jsd, combine_model, evaluate_nuclei_model, evaluate_combined_model
from .visualization import plot_umap_after_finetune

def fine_tune_model(adata, cell_types,save_path, args):
    # Clear GPU memory at start
    global device  # Add this as the first line
    print(f"Device at start of function: {device}") 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    # Check if GPU is actually free
    try:
        test_tensor = torch.randn(10, 10, device=device)
        del test_tensor
        torch.cuda.empty_cache()
        print("GPU test successful")
    except Exception as e:
        print(f"GPU test failed: {e}")
        print("Switching to CPU")
        device = torch.device('cpu')
    
    print("fine_tune_model AAA")
    def load_args_from_json(path):
        with open(path, 'r') as f:
            return json.load(f)

    nuclei_args = load_args_from_json(os.path.join(args.nuclei_model_run, 'arguments.json'))
    cell_args = load_args_from_json(os.path.join(args.cell_model_run, 'arguments.json'))

    
    nuclei_model = load_vae_jsd(nuclei_args, args.nuclei_model_run)
    cell_model = load_vae_jsd(cell_args, args.cell_model_run)

    # ======== Load mu and log_var from cell model for by cell type for training nuclei model=====
    cell_mu = {}
    cell_log_var = {}
    # open last epoch file
    try:
        cell_train_data_path = sorted([
            f for f in os.listdir(os.path.join(args.cell_model_run, "epoch_dump"))
                if f.startswith("epoch_") and f.endswith("_train.json")
            ])[-1]
    except:
        cell_train_data_path = os.path.join(args.cell_model_run, "epoch_dump", "train.json")
    
    #load cell train data
    cell_train_data = json.load(open(os.path.join(args.cell_model_run, "epoch_dump", cell_train_data_path)))
    if args.was_train_by_cell_type:
        for cell_type, celltype_data in cell_train_data.items():
            for batch_data in celltype_data.values():
                cell_types_in_batch = batch_data["cell_type_original"]
                mus_in_batch = batch_data["mu"]
                log_vars_in_batch = batch_data["log_var"]

                for i in range(len(cell_types_in_batch)):
                    cell_type = cell_types_in_batch[i]
                    mu = mus_in_batch[i]
                    log_var = log_vars_in_batch[i]

                    if cell_type not in cell_mu:
                        cell_mu[cell_type] = []
                        cell_log_var[cell_type] = []
                    
                    cell_mu[cell_type].append(mu)
                    cell_log_var[cell_type].append(log_var)

    else:   
        for batch_data in cell_train_data.values():
            cell_types_in_batch = batch_data["cell_type_original"]
            mus_in_batch = batch_data["mu"]
            log_vars_in_batch = batch_data["log_var"]

            for i in range(len(cell_types_in_batch)):
                cell_type = cell_types_in_batch[i]
                mu = mus_in_batch[i]
                log_var = log_vars_in_batch[i]

                if cell_type not in cell_mu:
                    cell_mu[cell_type] = []
                    cell_log_var[cell_type] = []
                
                cell_mu[cell_type].append(mu)
                cell_log_var[cell_type].append(log_var)

    # cell mu and log_var statistics, number of data under each cell type
    for cell_type in cell_mu.keys():
        print(f"Cell type: {cell_type}")
        print(f"Cell mu length: {len(cell_mu[cell_type])}")
        print(f"Cell log_var length: {len(cell_log_var[cell_type])}")
        print(f"Number of data under cell type: {len(cell_mu[cell_type])}")
    
    
    # ======== Load mu and log_var from cell model for by cell type for evaluating nuclei model=====
    cell_mu_valid = {}
    cell_log_var_valid = {}
    # open last epoch file
    try:
        cell_valid_data_path = sorted([
            f for f in os.listdir(os.path.join(args.cell_model_run, "epoch_dump"))
            if f.startswith("epoch_") and f.endswith("_valid.json")
        ])[-1]
    except:
        cell_valid_data_path = os.path.join(args.cell_model_run, "epoch_dump", "valid.json")
    
    cell_valid_data = json.load(open(os.path.join(args.cell_model_run, "epoch_dump", cell_valid_data_path)))

    if args.was_train_by_cell_type:
        for cell_type, celltype_data in cell_valid_data.items():
            for batch_data in celltype_data.values():
                cell_types_in_batch = batch_data["cell_type_original"]
                mus_in_batch = batch_data["mu"]
                log_vars_in_batch = batch_data["log_var"]

                for i in range(len(cell_types_in_batch)):
                    cell_type = cell_types_in_batch[i]
                    mu = mus_in_batch[i]
                    log_var = log_vars_in_batch[i]

                    if cell_type not in cell_mu_valid:
                        cell_mu_valid[cell_type] = []
                        cell_log_var_valid[cell_type] = []
                    
                    cell_mu_valid[cell_type].append(mu)
                    cell_log_var_valid[cell_type].append(log_var)

    else:
        for batch_data in cell_valid_data.values():
            cell_types_in_batch = batch_data["cell_type_original"]
            mus_in_batch = batch_data["mu"]
            log_vars_in_batch = batch_data["log_var"]

            for i in range(len(cell_types_in_batch)):
                cell_type = cell_types_in_batch[i]
                mu = mus_in_batch[i]
                log_var = log_vars_in_batch[i]

                if cell_type not in cell_mu_valid:
                    cell_mu_valid[cell_type] = []
                    cell_log_var_valid[cell_type] = []
                
                cell_mu_valid[cell_type].append(mu)
                cell_log_var_valid[cell_type].append(log_var)

    # ==========================start fine tuning nuclei model with cell type information==========================
    model = nuclei_model
    model_str = str(model)
    with open('{}/config.txt'.format(save_path), 'w') as f:
        f.write(model_str)
    
    if args.no_lr:
        optimizer = optim.Adam(model.parameters())
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    model.to(device)
    loss_list = []
    loss_list.append(["epoch","loss","recons_loss","KL_div","L1_reg"])
    # load data
    
    #create dataloader by cell type
    # print("====debug cell types=====")
    # print(cell_types)
    # print(adata.obs["cell_type"].unique())
    cell_type_dataloader = {}
    remove_list = []
    for cell_type in cell_types:
        print(f"Cell type: {cell_type}")
        print(adata[adata.obs["cell_type"] == cell_type].X.shape)
       
        if adata[adata.obs["cell_type"] == cell_type].X.shape[0] > 0:
            dataloaders = create_dataloaders(adata[adata.obs["cell_type"] == cell_type], args)
            if dataloaders[0] is not None:
                cell_type_dataloader[cell_type] = dataloaders
            else:
                print(f"No data available for cell type: {cell_type}, remove it from cell_types and skip it")
                remove_list.append(cell_type)
        else:
            print(f"No data available for cell type: {cell_type}, remove it from cell_types and skip it")
            remove_list.append(cell_type)
        
    cell_types = list(set(cell_types) - set(remove_list))
    print(cell_types)
    # print train, valid, test dataloder size of each cell type
    for cell_type in cell_types:
        print(f"Cell type: {cell_type}")
        print(f"Train dataloader size: {len(cell_type_dataloader[cell_type][0])}")
        print(f"Valid dataloader size: {len(cell_type_dataloader[cell_type][1])}")
        print(f"Test dataloader size: {len(cell_type_dataloader[cell_type][2])}")

    # start training
    args.alpha_original = args.alpha
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(args.num_epochs):
        # anneal alpha:
        if args.anneal_alpha:
            if epoch < args.anneal_alpha_start_epoch:
                args.alpha = 0
            elif epoch >= args.anneal_alpha_start_epoch and epoch <= args.anneal_alpha_epoch:
                args.alpha = args.alpha_original * (epoch - args.anneal_alpha_start_epoch) / (args.anneal_alpha_epoch - args.anneal_alpha_start_epoch)
            else:
                args.alpha = args.alpha_original

        total_loss = 0.0
        total_recons_loss = 0.0
        total_nuclei_JS_div = 0.0
        total_cell_type_JS_div = 0.0
        total_num_data = 0.0
        cell_type_save_dict = {}
        start_time = time.time()
        for cell_type in cell_types:
            # if cell_type is not in cell_mu or cell_log_var, skip it
            if cell_type not in cell_mu or cell_type not in cell_log_var:
                print(f"Cell type: {cell_type} not in cell_mu or cell_log_var, skip it")
                continue
            print(f"Epoch: {epoch}, Cell type: {cell_type}")
            
            train_loader, valid_loader, test_loader = cell_type_dataloader[cell_type]
            total_num_data += len(train_loader.dataset)
            #get cell mu and log_var from cell model
            current_cell_mu = cell_mu[cell_type]    
            current_cell_log_var = cell_log_var[cell_type]

            if args.use_all_cell_mu_log_var:
                current_cell_mu = current_cell_mu
                current_cell_log_var = current_cell_log_var
            else:
                #check if cell_mu and cell_log_var are the same length as train_loader,
                # if cell_mu > train_loader, then truncate cell_mu and cell_log_var to the same length as train_loader
                # if cell_mu < train_loader, then repeat cell_mu and cell_log_var to the same length as train_loader
                num_samples_in_loader = len(train_loader.dataset)
                if len(current_cell_mu) > num_samples_in_loader:
                    current_cell_mu = current_cell_mu[:num_samples_in_loader]
                    current_cell_log_var = current_cell_log_var[:num_samples_in_loader]
                elif len(current_cell_mu) < num_samples_in_loader:
                    repeats = num_samples_in_loader // len(current_cell_mu)
                    remainder = num_samples_in_loader % len(current_cell_mu)
                    current_cell_mu = current_cell_mu * repeats + current_cell_mu[:remainder]
                    current_cell_log_var = current_cell_log_var * repeats + current_cell_log_var[:remainder]
                
                
    

            
            cell_type_save_dict[cell_type] = {}
            # start fine tuning
            for batch_idx, batch in enumerate(train_loader):
                data = batch['data']
                data = data.to(device)
                encoded, decoded, mu, log_var = model(data) 

                # get next batch of cell mu and log_var from cell model for this cell type
                start_idx = batch_idx * args.batch_size
                end_idx = start_idx + data.size(0)

                if args.use_all_cell_mu_log_var:
                    cell_mu_batch = torch.tensor(current_cell_mu, dtype=torch.float32).to(device)
                    cell_log_var_batch = torch.tensor(current_cell_log_var, dtype=torch.float32).to(device)
                else:
                    cell_mu_batch = torch.tensor(current_cell_mu[start_idx:end_idx], dtype=torch.float32).to(device)
                    cell_log_var_batch = torch.tensor(current_cell_log_var[start_idx:end_idx], dtype=torch.float32).to(device)

                all_loss = model.fine_tune_loss_function(decoded, data, mu, log_var, cell_mu_batch, cell_log_var_batch, args)
                loss = all_loss[0]
                recons_loss = all_loss[1]
                nuclei_JS_div = all_loss[2]
                cell_type_JS_div = all_loss[3]
                # L1_reg = all_loss[3]

                # # save_latent distribution histogram
                # if epoch % 10 == 0 and batch_idx == len(train_loader) - 1: 
                #     print("Gettting histograms..." )
                #     save_latent_distribution_histograms(mu, log_var, loss,recons_loss, nuclei_JS_div, cell_type_JS_div, epoch, save_path)
                #     print("Histograms saved")
                
                print(f"Epoch: {epoch}, Cell type: {cell_type}, Batch: {batch_idx}, Loss: {loss.item()}, Recon Loss: {recons_loss.item()}, Nuclei JS Div: {nuclei_JS_div.item()}, Cell Type JS Div: {cell_type_JS_div.item()}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()*data.size(0)
                total_recons_loss += recons_loss.item()*data.size(0)
                total_nuclei_JS_div += nuclei_JS_div.item()*data.size(0)
                total_cell_type_JS_div += cell_type_JS_div.item()*data.size(0)
                total_num_data += data.size(0)

                encoded_np = encoded.cpu().detach().numpy() 
                decoded_np = decoded.cpu().detach().numpy()
                mu_np = mu.cpu().detach().numpy()
                log_var_np = log_var.cpu().detach().numpy()
                loss_np = loss.cpu().detach().numpy()
                recons_loss_np = recons_loss.cpu().detach().numpy()
                nuclei_JS_div_np = nuclei_JS_div.cpu().detach().numpy() 
                cell_type_JS_div_np = cell_type_JS_div.cpu().detach().numpy() 

                celltypes = batch["cell_type"].cpu().detach().numpy()
                meta_batch_id = batch["batch_id"].cpu().detach().numpy()

                original_cell_types = []
                for cell_code in batch["cell_type"].cpu().numpy():
                    original_cell_types.append(train_loader.dataset.get_original_value('cell_type', cell_code))

                # Get the original batch IDs if they exist
                original_batch_ids = []
                if train_loader.dataset.has_batch_id:
                    for batch_code in batch["batch_id"].cpu().numpy():
                        original_batch_ids.append(train_loader.dataset.get_original_value('batch_id', batch_code))
                


                # save loss, recons_loss, KL_div, encoded, decoded, mu, log_var
                batch_results = {}
                batch_results["loss"] = loss_np.tolist()
                batch_results["recons_loss"] = recons_loss_np.tolist()
                batch_results["nuclei_JS_div"] = nuclei_JS_div_np.tolist()
                batch_results["cell_type_JS_div"] = cell_type_JS_div_np.tolist()
                batch_results["encoded"] = encoded_np.tolist()
                batch_results["decoded"] = decoded_np.tolist()
                batch_results["mu"] = mu_np.tolist()
                batch_results["log_var"] = log_var_np.tolist()
                # batch_results["cell_type_mu"] = cell_mu_batch.tolist()
                # batch_results["cell_type_log_var"] = cell_log_var_batch.tolist()
                batch_results["cell_type"] = celltypes.tolist()
                batch_results["meta_batch_id"] = meta_batch_id.tolist()
                batch_results["cell_type_original"] = original_cell_types
                batch_results["meta_batch_id_original"] = original_batch_ids
                cell_type_save_dict[cell_type][batch_idx] = batch_results
                
                if args.debug and batch_idx == 1:
                    break
                # for test purpose, break after 2 batch
                # if batch_idx == 1:      
                #     break

            
            # for test purpose, break after 1 st cell type:
            if args.debug and cell_type == cell_types[0] and epoch == 0:
                break

    
        # save cell_type_save_dict
        # ============Save model output, loss for each batch in one file==========
        # make epoch dump
        # save using the function save_epoch_save_dict
        # if total_loss / total_num_data < best_loss:
        #     best_loss = total_loss / total_num_data
        #     best_epoch = epoch
        #     save_epoch_save_dict(cell_type_save_dict, save_path, epoch, "train_finetune")    
        

        os.makedirs(os.path.join(save_path, "finetune_loss"), exist_ok=True)
        with open(os.path.join(save_path, "finetune_loss", f"epoch_{epoch}_finetune_loss.txt"), 'w') as fp:
            fp.write("epoch,cell_type,loss,recons_loss,nuclei_JS_div,cell_type_JS_div\n")
            for cell_type in cell_types:
                fp.write(f"{epoch},{cell_type},{total_loss},{total_recons_loss},{total_nuclei_JS_div},{total_cell_type_JS_div}\n")      
                
        print("cell_type_save_dict saved")

        #=============Saving Epoch loss, model, print run time====================
        epoch_loss = total_loss / total_num_data
        epoch_recons_loss = total_recons_loss / total_num_data
        epoch_nuclei_JS_div = total_nuclei_JS_div / total_num_data
        epoch_cell_type_JS_div = total_cell_type_JS_div / total_num_data
        print(f"Epoch: {epoch}, Cell type: {cell_type}, Loss: {epoch_loss}, Recon Loss: {epoch_recons_loss}, Nuclei JS Div: {epoch_nuclei_JS_div}, Cell Type JS Div: {epoch_cell_type_JS_div}")
        loss_list.append([epoch,epoch_loss,epoch_recons_loss,epoch_nuclei_JS_div,epoch_cell_type_JS_div])
        
        # save the model and epoch dict every 10 epochs and last epoch
        if epoch % 10 == 0 or epoch == args.num_epochs - 1:
            os.makedirs(os.path.join(save_path,"models"), exist_ok=True)
            torch.save(model.state_dict(),os.path.join(save_path,"models","model.pt"))
            os.makedirs(os.path.join(save_path,"epoch_dump"), exist_ok=True)
            train_finetune_save_path = os.path.join(save_path, "epoch_dump", f"train_finetune.json")
            save_epoch_save_dict(cell_type_save_dict, save_path, epoch, "train_finetune", save_file_path=train_finetune_save_path)    
            
        end_time = time.time()
        epoch_duration = end_time - start_time
        hours = int(epoch_duration // 3600)
        minutes = int((epoch_duration % 3600) // 60)
        seconds = epoch_duration % 60
        print(f"Epoch {epoch+1}/{args.num_epochs} completed in {hours}h:{minutes}m:{seconds:.2f}s.")
        print(f"loss={epoch_loss}, recon_loss={epoch_recons_loss}, KL_div={epoch_recons_loss}")
        # runtime_str = "{hours}:{minutes}:{seconds:.2f}"

        print("Save epoch training loss info")
        # os.makedirs(os.path.join(save_path, "train_epoch_loss"), exist_ok=True)
        loss_file = os.path.join(save_path, "train_epoch_loss.txt")
        header = "epoch,loss,recons_loss,nuclei_JS_div,cell_type_JS_div"

        # Check if the file exists
        file_exists = os.path.isfile(loss_file)

        # Open the file in append mode if it exists, otherwise in write mode
        with open(loss_file, 'a' if file_exists else 'w') as file:
            # Write the header only if the file doesn't exist
            if not file_exists:
                file.write(header + '\n')
            # Append the new line with the epoch data
            line = ','.join(map(str, [epoch, epoch_loss, epoch_recons_loss, epoch_nuclei_JS_div, epoch_cell_type_JS_div]))
            file.write(line + '\n')

        
        
        # ===================== Model Evaluation =====================
        
        # ==================Evaluate nuclei mode using valid set, load cell type by cell type=====================
        # aggregate loss
        print("Evaluating nuclei model")
        nuclei_loss = 0.0
        nuclei_recons_loss = 0.0
        nuclei_JS_div = 0.0
        nuclei_JS_div_cell_nuclei = 0.0
        nuclei_num_data = 0
        nuclei_valid_epoch_save_dict = {}
        for cell_type in cell_types:
            # check if cell_type is in cell_mu_valid and cell_log_var_valid
            if cell_type not in cell_mu_valid or cell_type not in cell_log_var_valid:
                print(f"No data available for cell type: {cell_type}, skip it")
                continue
            print(f"Evaluating nuclei model using valid set for cell type: {cell_type}")
            valid_loader = cell_type_dataloader[cell_type][1]
            # load cell mu and log_var from cell model for this cell type
            current_cell_mu = cell_mu_valid[cell_type]
            current_cell_log_var = cell_log_var_valid[cell_type]

            # if current_cell_mu is empty, skip this cell type
            if len(current_cell_mu) == 0:
                print(f"No data available for cell type: {cell_type}, skip it")
                continue

            #check if cell_mu and cell_log_var are the same length as valid_loader,
            # if cell_mu > valid_loader, then truncate cell_mu and cell_log_var to the same length as valid_loader
            # if cell_mu < valid_loader, then repeat cell_mu and cell_log_var to the same length as valid_loader    
            if args.use_all_cell_mu_log_var:
                current_cell_mu = current_cell_mu
                current_cell_log_var = current_cell_log_var
            else:   
                num_samples_in_loader = len(valid_loader.dataset)
                if len(current_cell_mu) > num_samples_in_loader:
                    current_cell_mu = current_cell_mu[:num_samples_in_loader]
                    current_cell_log_var = current_cell_log_var[:num_samples_in_loader]
                elif len(current_cell_mu) < num_samples_in_loader:
                    repeats = num_samples_in_loader // len(current_cell_mu) 
                    remainder = num_samples_in_loader % len(current_cell_mu)
                    current_cell_mu = current_cell_mu * repeats + current_cell_mu[:remainder]
                    current_cell_log_var = current_cell_log_var * repeats + current_cell_log_var[:remainder]

            # evaluate nuclei model using valid set with cell type information
            # print(current_cell_mu, current_cell_log_var)
            nuclei_model_cell_type_valid_epoch_save_dict, cell_type_total_loss, cell_type_total_recons_loss, cell_type_total_JS_div_nuclei, cell_type_total_JS_div_cell_nuclei =\
                 evaluate_nuclei_model(model, valid_loader, cell_type, save_path, epoch,current_cell_mu, current_cell_log_var, args)
            
            # store validation results for each cell type
            nuclei_valid_epoch_save_dict[cell_type] = nuclei_model_cell_type_valid_epoch_save_dict
            # aggregate loss for calculating average loss, recons_loss, KL_div for the whole epoch for nuclei model
            nuclei_loss += cell_type_total_loss
            nuclei_recons_loss += cell_type_total_recons_loss
            nuclei_JS_div += cell_type_total_JS_div_nuclei
            nuclei_JS_div_cell_nuclei += cell_type_total_JS_div_cell_nuclei
            nuclei_num_data += len(valid_loader.dataset)
            print(f"Nuclei model evaluation for cell type: {cell_type} completed")

            # for test purpose, break after 1 st cell type:
            if args.debug and cell_type == cell_types[0] and epoch == 0:
                break

        # calculate average loss, recons_loss, KL_div for the whole epoch for nuclei model
        nuclei_loss = nuclei_loss / nuclei_num_data
        nuclei_recons_loss = nuclei_recons_loss / nuclei_num_data
        nuclei_JS_div = nuclei_JS_div / nuclei_num_data
        nuclei_JS_div_cell_nuclei = nuclei_JS_div_cell_nuclei / nuclei_num_data

        # save nuclei_valid_epoch_save_dict every 10 epochs and last epoch
        if epoch % 10 == 0 or epoch == args.num_epochs - 1:
            valid_nuclei_save_path = os.path.join(save_path, "epoch_dump", f"valid_nuclei.json")
            save_epoch_save_dict(nuclei_valid_epoch_save_dict, save_path, epoch,"valid_nuclei", save_file_path=valid_nuclei_save_path)

        # save average loss to file, is file exists, append to file, otherwise create new file
        nuclei_loss_file = os.path.join(save_path, "nuclei_model_validation_loss.txt")
        file_exists = os.path.isfile(nuclei_loss_file)
        with open(nuclei_loss_file, 'a' if file_exists else 'w') as f:
            if not file_exists:
                f.write("epoch,loss,recons_loss,nuclei_JS_div,cell_type_JS_div\n")
            f.write(f"{epoch},{nuclei_loss},{nuclei_recons_loss},{nuclei_JS_div},{nuclei_JS_div_cell_nuclei}\n")    
        print(f"Nuclei model loss saved to {nuclei_loss_file}")

        # ==================Evaluate combined model using validation set, load cell type by cell type=====================
        # first create combined model
        print("Evaluating combined model")
        nuclei_model = model
        combined_model = combine_model(nuclei_model, cell_model, nuclei_args)
        combined_loss = 0.0
        combined_recons_loss = 0.0
        combined_JS_div_nuclei = 0.0
        combined_JS_div_cell_nuclei = 0.0
        combined_num_data = 0
        combined_valid_epoch_save_dict = {}
        #  evaludate cell by cell type
        for cell_type in cell_types:
            print(f"Evaluating combined model using test set for cell type: {cell_type}")
            if cell_type not in cell_mu_valid or cell_type not in cell_log_var_valid:
                print(f"No data available for cell type: {cell_type}, skip it")
                continue
            valid_loader = cell_type_dataloader[cell_type][1]
            # load cell mu and log_var from cell model for this cell type
            current_cell_mu = cell_mu_valid[cell_type]
            current_cell_log_var = cell_log_var_valid[cell_type]

            # if current_cell_mu is empty, skip this cell type
            if len(current_cell_mu) == 0:
                print(f"No data available for cell type: {cell_type}, skip it")
                continue

            #check if cell_mu and cell_log_var are the same length as valid_loader,
            # if cell_mu > valid_loader, then truncate cell_mu and cell_log_var to the same length as valid_loader
            # if cell_mu < valid_loader, then repeat cell_mu and cell_log_var to the same length as valid_loader    
            num_samples_in_loader = len(valid_loader.dataset)
            if len(current_cell_mu) > num_samples_in_loader:
                current_cell_mu = current_cell_mu[:num_samples_in_loader]
                current_cell_log_var = current_cell_log_var[:num_samples_in_loader]
            elif len(current_cell_mu) < num_samples_in_loader:
                repeats = num_samples_in_loader // len(current_cell_mu) 
                remainder = num_samples_in_loader % len(current_cell_mu)
                current_cell_mu = current_cell_mu * repeats + current_cell_mu[:remainder]
                current_cell_log_var = current_cell_log_var * repeats + current_cell_log_var[:remainder]

            combined_model_cell_type_valid_epoch_save_dict, cell_type_total_loss, cell_type_total_recons_loss, cell_type_total_JS_div_nuclei, cell_type_total_JS_div_cell_nuclei = evaluate_combined_model(combined_model, valid_loader, cell_type, save_path, epoch, args, current_cell_mu, current_cell_log_var)

            # store validation results for each cell type
            combined_valid_epoch_save_dict[cell_type] = combined_model_cell_type_valid_epoch_save_dict
            # aggregate loss for calculating average loss, recons_loss, KL_div for the whole epoch for combined model
            print(f"Combined model evaluation for cell type: {cell_type} completed")
            combined_loss += cell_type_total_loss
            combined_recons_loss += cell_type_total_recons_loss
            combined_JS_div_nuclei += cell_type_total_JS_div_nuclei
            combined_JS_div_cell_nuclei += cell_type_total_JS_div_cell_nuclei
            combined_num_data += len(valid_loader.dataset)
            # combined_valid_epoch_save_dict[cell_type] = cell_type_valid_epoch_save_dict

            # for test purpose, break after 1 st cell type:
            if args.debug and cell_type == cell_types[0] and epoch == 0:
                break

        # calculate average loss, recons_loss, KL_div for the whole epoch for combined model
        combined_loss = combined_loss / combined_num_data
        combined_recons_loss = combined_recons_loss / combined_num_data
        combined_JS_div_nuclei = combined_JS_div_nuclei / combined_num_data
        combined_JS_div_cell_nuclei = combined_JS_div_cell_nuclei / combined_num_data
        # save combined_valid_epoch_save_dict
        if epoch % 10 == 0 or epoch == args.num_epochs - 1:
            valid_combined_save_path = os.path.join(save_path, "epoch_dump", f"valid_combined.json")
            save_epoch_save_dict(combined_valid_epoch_save_dict, save_path, epoch,"valid_combined", save_file_path=valid_combined_save_path)     
        
        # save_epoch_save_dict(combined_valid_epoch_save_dict, save_path, epoch,"valid_combined")     

        # save average loss to file, is file exists, append to file, otherwise create new file
        combined_loss_file = os.path.join(save_path, "combined_model_validation_loss.txt")
        file_exists = os.path.isfile(combined_loss_file)
        with open(combined_loss_file, 'a' if file_exists else 'w') as f:
            if not file_exists:
                f.write("epoch,loss,recons_loss,nuclei_JS_div,cell_type_JS_div\n")
            f.write(f"{epoch},{combined_loss},{combined_recons_loss},{combined_JS_div_nuclei},{combined_JS_div_cell_nuclei}\n")    

        print(f"Nuclei model evaluation completed")
        print(f"Nuclei model loss: {nuclei_loss}, Nuclei model recons_loss: {nuclei_recons_loss}, Nuclei model JS_div_nuclei: {nuclei_JS_div}, Nuclei model JS_div_cell_nuclei: {nuclei_JS_div_cell_nuclei}")
        print("--------------------------------")
        print(f"Combined model evaluation completed")
        print(f"Combined model loss: {combined_loss}, Combined model recons_loss: {combined_recons_loss}, Combined model JS_div_nuclei: {combined_JS_div_nuclei}, Combined model JS_div_cell_nuclei: {combined_JS_div_cell_nuclei}")


        # ===================== Umap Plot =====================
        if epoch % 10 == 0 or epoch == args.num_epochs - 1:
            plot_umap_after_finetune(save_path,feature="encoded",data_split= "train_finetune", epoch=epoch, args=args, file_path=train_finetune_save_path)
            plot_umap_after_finetune(save_path,feature="decoded",data_split= "train_finetune", epoch=epoch, args=args, file_path=train_finetune_save_path)
            plot_umap_after_finetune(save_path,feature="encoded",data_split= "valid_nuclei", epoch=epoch, args=args, file_path=valid_nuclei_save_path)
            plot_umap_after_finetune(save_path,feature="decoded",data_split= "valid_nuclei", epoch=epoch, args=args, file_path=valid_nuclei_save_path)
            plot_umap_after_finetune(save_path,feature="encoded",data_split= "valid_combined", epoch=epoch, args=args, file_path=valid_combined_save_path)
            plot_umap_after_finetune(save_path,feature="decoded",data_split= "valid_combined", epoch=epoch, args=args, file_path=valid_combined_save_path)

        if args.debug:
            break
        # plot loss curve

        # print(f"Best loss: {best_loss}, Best epoch: {best_epoch}")

def debug_model(cell_types,save_path,args):
    print("Debugging model")
    print("args: ",args)
    
    # ======== Load mu and log_var from cell model for by cell type for training nuclei model=====
    # load cell_mu, cell_log_var
    cell_mu = {}
    cell_log_var = {}

    cell_train_data_path = sorted([
            f for f in os.listdir(os.path.join(args.cell_model_run, "epoch_dump"))
            if f.startswith("epoch_") and f.endswith("_train.json")
        ])[-1]
    
    cell_train_data = json.load(open(os.path.join(args.cell_model_run, "epoch_dump", cell_train_data_path)))
    
    for batch_data in cell_train_data.values():
        cell_types_in_batch = batch_data["cell_type_original"]
        mus_in_batch = batch_data["mu"]
        log_vars_in_batch = batch_data["log_var"]

        for i in range(len(cell_types_in_batch)):
            cell_type = cell_types_in_batch[i]
            mu = mus_in_batch[i]
            log_var = log_vars_in_batch[i]

            if cell_type not in cell_mu:
                cell_mu[cell_type] = []
                cell_log_var[cell_type] = []
            
            cell_mu[cell_type].append(mu)
            cell_log_var[cell_type].append(log_var)
    
    # load nuclei_mu, nuclei_log_var
    nuclei_mu = {}
    nuclei_log_var = {}

    nuclei_train_data_path = sorted([
            f for f in os.listdir(os.path.join(args.nuclei_model_run, "epoch_dump"))
            if f.startswith("epoch_") and f.endswith("_train.json")
        ])[-1]

    nuclei_train_data = json.load(open(os.path.join(args.nuclei_model_run, "epoch_dump", nuclei_train_data_path)))

    for batch_data in nuclei_train_data.values():
        cell_types_in_batch = batch_data["cell_type_original"]
        mus_in_batch = batch_data["mu"]
        log_vars_in_batch = batch_data["log_var"]

        for i in range(len(cell_types_in_batch)):
            cell_type = cell_types_in_batch[i]
            mu = mus_in_batch[i]
            log_var = log_vars_in_batch[i]

            if cell_type not in nuclei_mu:  
                nuclei_mu[cell_type] = []
                nuclei_log_var[cell_type] = []
            
            nuclei_mu[cell_type].append(mu)
            nuclei_log_var[cell_type].append(log_var)

    # compare mu and logvar by cell type between cell model and nuclei model
    for cell_type in cell_types:
        print(f"Comparing mu and logvar by cell type: {cell_type}")
        # print length of cell_mu and nuclei_mu
        print(f"Cell model mu length: {len(cell_mu[cell_type])}")
        print(f"Nuclei model mu length: {len(nuclei_mu[cell_type])}")
        # print(f"Cell model mu: {cell_mu[cell_type]}")
        # print(f"Cell model log_var: {cell_log_var[cell_type]}")
        # print(f"Nuclei model mu: {nuclei_mu[cell_type]}")
        # print(f"Nuclei model log_var: {nuclei_log_var[cell_type]}")

        # calculate mean and std of cell_mu and nuclei_mu and cell_log_var and nuclei log_var, and plot histogram and save to to save_path
        cell_mu_mean = np.mean(cell_mu[cell_type])
        cell_mu_std = np.std(cell_mu[cell_type])
        nuclei_mu_mean = np.mean(nuclei_mu[cell_type])  
        nuclei_mu_std = np.std(nuclei_mu[cell_type])
        cell_log_var_mean = np.mean(cell_log_var[cell_type])
        cell_log_var_std = np.std(cell_log_var[cell_type])
        nuclei_log_var_mean = np.mean(nuclei_log_var[cell_type])
        nuclei_log_var_std = np.std(nuclei_log_var[cell_type])

        print(f"Cell model mu mean: {cell_mu_mean}, Cell model mu std: {cell_mu_std}")
        print(f"Nuclei model mu mean: {nuclei_mu_mean}, Nuclei model mu std: {nuclei_mu_std}")
        print(f"Cell model log_var mean: {cell_log_var_mean}, Cell model log_var std: {cell_log_var_std}")
        print(f"Nuclei model log_var mean: {nuclei_log_var_mean}, Nuclei model log_var std: {nuclei_log_var_std}")

        # plot histogram of cell_mu and nuclei_mu
        #create a histogram folder in save_path
        os.makedirs(os.path.join(save_path, "histogram"), exist_ok=True)
        plt.hist(cell_mu[cell_type], bins=100, alpha=0.5, label="Cell model")
        plt.hist(nuclei_mu[cell_type], bins=100, alpha=0.5, label="Nuclei model")
        plt.legend()
        plt.savefig(os.path.join(save_path, "histogram", f"cell_mu_histogram_{cell_type}.png"))
        plt.close()

        # plot histogram of cell_log_var and nuclei_log_var
        #create a histogram folder in save_path
        # os.makedirs(os.path.join(save_path, "histogram"), exist_ok=True)
        plt.hist(cell_log_var[cell_type], bins=100, alpha=0.5, label="Cell model")
        plt.hist(nuclei_log_var[cell_type], bins=100, alpha=0.5, label="Nuclei model")
        plt.legend()
        plt.savefig(os.path.join(save_path, "histogram", f"cell_log_var_histogram_{cell_type}.png"))
        plt.close()
        print("--------------------------------")


   
    # load cell_type_dataloader
    cell_type_dataloader = np.load(os.path.join(save_path, "cell_type_dataloader.npy"))
    
if __name__ == "__main__":
    pass
