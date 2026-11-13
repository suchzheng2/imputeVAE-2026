import os
import time
import json
import gc
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from .utils import device, setup_device, verify_gpu_usage, save_epoch_save_dict
from .models import VAE, VAE_L1, VAE_JSD, VAE_NB
from .data import (create_dataloaders, create_dataloaders_by_cell_type,
                   minmax_normalize_data, log_transform_data)
from .visualization import plot_umap_after_train_by_celltype
from .plot_loss import plot_loss_after_train_general

def train_loop_ABC(adata, cell_types, save_path,args):


    global device
    
    # Set device based on CUDA availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        # Test GPU functionality
        try:
            test_tensor = torch.randn(10, 10, device=device)
            del test_tensor
            torch.cuda.empty_cache()
            print("GPU test successful")
        except Exception as e:
            print(f"GPU test failed: {e}")
            print("Falling back to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU")

    print(f"Final device: {device}")

    print("number of features: ",args.num_feature)

    if args.model == "vae_nb":
        model = VAE_NB(input_size=args.num_feature,
            num_internal=args.num_internal,
            num_latent=args.num_latent,
            use_zinb=getattr(args, 'use_zinb', False))
        if not getattr(args, 'use_raw_data', False) or getattr(args, 'raw_data_normalization', 'log1p') != 'raw':
            print("WARNING: VAE_NB is designed for raw count data. "
                  "Consider using --use_raw_data True --raw_data_normalization raw")
    elif args.model == "vae_js":
            model = VAE_JSD(input_size = args.num_feature, \
            num_internal = args.num_internal, \
            num_latent = args.num_latent)
    else:
        model = VAE_L1(input_size = args.num_feature, \
            num_internal = args.num_internal, \
            num_latent = args.num_latent,\
            l1_coef = args.l1_coef)

    if args.load_trained_model:
        trained_model_path = os.path.join(args.load_trained_model_path)
        model.load_state_dict(torch.load(trained_model_path, map_location=device))
        model = model.to(device)  # Ensure model is on the correct device
        print(f"Loaded trained model from {args.load_trained_model}")


    model_str = str(model)

    # print the model architecture
    print(model)

    # Writing to a text file
    with open('{}/config.txt'.format(save_path), 'w') as f:
        f.write(model_str)

    
    
    if args.no_lr:
        optimizer = optim.Adam(model.parameters())
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    model.to(device)
    loss_list = []
    loss_list.append(["epoch","loss","recons_loss","KL_div","L1_reg"])
    valid_loss_list = []

    save_count = 0
    
    args.beta_original = args.beta

    _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_loader_path = os.path.join(_project_dir, "cache", "dataloaders", args.data_path)
    os.makedirs(data_loader_path, exist_ok=True)

    print("data_loader_path: ",data_loader_path)

    train_loaders, valid_loaders, test_loaders = create_dataloaders_by_cell_type(adata,args)

    # after create the dataloader, remove the adata to save memory
    del adata
    gc.collect()
    torch.cuda.empty_cache()
    print("Adata removed and memory cleared")
    

    best_loss = float('inf')
    best_epoch = 0
    train_range = range(args.start_epoch,args.num_epochs)

    for epoch in train_range:
        start_time = time.time()
        print("======================Epoch {}===================".format(epoch))

        if getattr(args, "kl_anneal", False): 
            if epoch < args.kl_anneal_start_epoch:
                args.beta = 0.0
            elif epoch < args.kl_anneal_start_epoch + args.kl_anneal_epoch:
                progress = (epoch - args.kl_anneal_start_epoch) / max(1, args.kl_anneal_epoch)
                args.beta = min(args.beta_original, args.beta_original * progress)
            else:    
                args.beta = args.beta_original
            print(f"[Epoch {epoch}] Annealed beta: {args.beta}")
        
        total_loss = 0.0
        total_recons_loss = 0.0
        total_JSD_div = 0.0
        total_L1_reg = 0.0

        num_data = 0
        # debug_path = os.path.join(save_path,"debug")
        # os.makedirs(debug_path, exist_ok=True)
        batch_count = 0
        epoch_save_dict = {}
        batch_loss_list = []



        for celltype, train_loader in train_loaders.items():
            epoch_save_dict[celltype] = {}
            print(f"Training cell type: {celltype}")
            print(f"Number of samples: {len(train_loader.dataset)}")
            print(f"Number of batches: {len(train_loader)}")
            print(f"Batch size: {train_loader.batch_size}")
            print("-"*50)
            for batch_idx, batch in enumerate(train_loader):
                data = batch['data'].to(device)
                verify_gpu_usage(model, data, f"Batch {batch_idx}")

                if torch.isnan(data).any() or torch.isinf(data).any():
                    print(f"Data-check: NaN or Inf detected in input at batch {batch_idx}")
                if torch.all(data == 0):
                    print(f"Data-check: All-zero input at batch {batch_idx}")

                encoded, decoded, mu, log_var = model(data)

                all_loss = model.loss_function(decoded, data, mu, log_var, args)

                loss = all_loss[0]
                recons_loss = all_loss[1]
                JSD_div = all_loss[2]

                # if epoch % 1 == 0 and batch_idx == len(train_loader) - 1:
                #     print("Gettting histograms..." )
                #     save_latent_distribution_histograms(mu, log_var, loss,recons_loss, JSD_div, epoch, save_path)

                print(f"[Epoch {epoch}, {batch_idx}/{len(train_loader)}]: loss:{loss},recons:{recons_loss}, KL:{JSD_div}")

                print(f"Loss device: {loss.device}")

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                total_loss += loss.item()*data.size(0)
                total_recons_loss += recons_loss.item()*data.size(0)
                total_JSD_div += JSD_div.item()*data.size(0)
            

            
                data_size = int(data.size(0))


                encoded = encoded.cpu().detach().numpy()
                decoded = decoded.cpu().detach().numpy()
                mu = mu.cpu().detach().numpy()
                log_var = log_var.cpu().detach().numpy()
                loss = loss.cpu().detach().numpy()
                recons_loss = recons_loss.cpu().detach().numpy()
                JSD_div = JSD_div.cpu().detach().numpy()
                # L1_reg = L1_reg.cpu().detach().numpy()

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
                    

                epoch_save_dict[celltype][batch_idx] = {}
                epoch_save_dict[celltype][batch_idx]["batch_idx"] = batch_idx
                epoch_save_dict[celltype][batch_idx]["cell_type"] = celltypes.tolist() 
                epoch_save_dict[celltype][batch_idx]["cell_type_original"] = original_cell_types
                epoch_save_dict[celltype][batch_idx]["meta_batch_id"] = meta_batch_id.tolist() 
                epoch_save_dict[celltype][batch_idx]["meta_batch_id_original"] = original_batch_ids
                epoch_save_dict[celltype][batch_idx]["encoded"] = encoded.tolist()    
                epoch_save_dict[celltype][batch_idx]["decoded"] = decoded.tolist()        
                epoch_save_dict[celltype][batch_idx]["mu"] = mu.tolist()
                epoch_save_dict[celltype][batch_idx]["log_var"] = log_var.tolist()
                epoch_save_dict[celltype][batch_idx]["loss"] = loss.tolist()
                epoch_save_dict[celltype][batch_idx]["recons_loss"] = recons_loss.tolist()
                epoch_save_dict[celltype][batch_idx]["JSD_div"] = JSD_div.tolist()
                # epoch_save_dict[celltype][batch_idx]["L1_reg"] = L1_reg.tolist()

                batch_loss_list.append([celltype,batch_idx,loss,recons_loss,JSD_div,data_size])

            num_data += len(train_loader.dataset)


        # ============Save model output, loss for each batch in one file==========
        # save the epoch save dict every 10 epochs and last epoch
        

        #=============Saving Epoch loss, model, print run time====================
        epoch_loss = total_loss / num_data
        epoch_recons_loss = total_recons_loss / num_data
        epoch_JSD_div_loss = total_JSD_div / num_data
        # epoch_L1_reg = total_L1_reg / num_data
        
        # Auto-compute recon loss normalization constant from epoch 0
        if getattr(args, "normalize_losses", False) and epoch == 0 and getattr(args, "recon_loss_norm", 0.0) == 0.0:
            args.recon_loss_norm = epoch_recons_loss
            print(f"[normalize_losses] Set recon_loss_norm = {args.recon_loss_norm:.6f} from epoch 0")

        loss_list.append([epoch,epoch_loss,epoch_recons_loss,epoch_JSD_div_loss])
        if epoch % 50 == 0 or epoch == args.num_epochs - 1:
            print(f"model saved on epoch: {epoch}")
            os.makedirs(os.path.join(save_path,"models"), exist_ok=True)
            torch.save(model.state_dict(),os.path.join(save_path,"models","model.pt"))

            os.makedirs(os.path.join(save_path,"epoch_dump"), exist_ok=True)
            train_file_path = os.path.join(save_path,"epoch_dump","train_epoch_{}.json".format(epoch))
            with open(train_file_path, 'w') as fp:
                json.dump(epoch_save_dict, fp)
            
            plot_umap_after_train_by_celltype(save_path,feature="encoded",data_split= "train", epoch = epoch, file_path=train_file_path)
            plot_umap_after_train_by_celltype(save_path,feature="decoded",data_split= "train", epoch = epoch, file_path=train_file_path)

        

       


        os.makedirs(os.path.join(save_path,"train_loss"), exist_ok=True)
        with open(os.path.join(save_path,"train_loss","epoch_{}_train_loss.txt".format(epoch)), 'w') as fp:
            header = ["celltype","batch","loss","recons_loss","JSD_div","data_size"]
            fp.write('\t'.join(header) + '\n')
            for row in batch_loss_list:
                fp.write('\t'.join(map(str, row)) + '\n')
            
       
        end_time = time.time()
        epoch_duration = end_time - start_time
        hours = int(epoch_duration // 3600)
        minutes = int((epoch_duration % 3600) // 60)
        seconds = epoch_duration % 60
        print(f"Epoch {epoch+1}/{args.num_epochs} completed in {hours}h:{minutes}m:{seconds:.2f}s.")
        print(f"loss={epoch_loss}, recon_loss={epoch_recons_loss}, JSD_div={epoch_JSD_div_loss}")
        if getattr(args, "normalize_losses", False) and getattr(args, "recon_loss_norm", 0.0) > 0:
            print(f"  [scaled] recon={epoch_recons_loss/args.recon_loss_norm:.4f}, JSD/ln2={epoch_JSD_div_loss/0.6931:.4f}")
        runtime_str = "{hours}:{minutes}:{seconds:.2f}"
        
        print("Save epoch training loss info")
        os.makedirs(os.path.join(save_path, "train_epoch_loss"), exist_ok=True)
        loss_file = os.path.join(save_path, "train_epoch_loss", "loss.txt")
        header = "epoch,loss,recons_loss,JSD_div,runtime"

        # Check if the file exists
        file_exists = os.path.isfile(loss_file)

        # Open the file in append mode if it exists, otherwise in write mode
        with open(loss_file, 'a' if file_exists else 'w') as file:
            # Write the header only if the file doesn't exist
            if not file_exists:
                file.write(header + '\n')

            
            # Append the new line with the epoch data
            line = ','.join(map(str, [epoch, epoch_loss, epoch_recons_loss, epoch_JSD_div_loss,runtime_str]))
            file.write(line + '\n')

        # ===================== Model Evaluation =====================

        # Plot Umap for the current epoch:
        # do this every 5 epochs:
        # only plot umap if train dict was saved
        # if epoch % 10 == 0 or epoch == args.num_epochs - 1:
        #     print("------------------------ Model Evaluation =====================")
        #     # os.makedirs(fig_save_dir,exist_ok=True)
        #     plot_umap_after_train_by_celltype(save_path,feature="encoded",data_split= "train", epoch = epoch, )
        #     plot_umap_after_train_by_celltype(save_path,feature="decoded",data_split= "train", epoch = epoch)

        # Evaluate Batch Effect
        # evaludate_batch_effect_handle(save_path,epoch)


        
        # ===================== Validate Model after epoch =================================
        if args.valid:
            valid_total_loss = 0.0
            valid_total_recons_loss = 0.0
            valid_total_JSD_div = 0.0
            # valid_total_L1_reg = 0.0

            valid_num_data = 0
            # debug_path = os.path.join(save_path,"debug")
            # os.makedirs(debug_path, exist_ok=True)
            valid_batch_count = 0
            valid_epoch_save_dict = {}
            valid_this_epoch_all_loss = []

            print("========= Epoch {} Validation ===========".format(epoch))
            
            # valid_loader_path = "dataloders/{}_valid_{}.pkl".format(cell_print,args.norm)
            # valid_loader = torch.load(valid_loader_path)
            for celltype, valid_loader in valid_loaders.items():
                valid_epoch_save_dict[celltype] = {}
              
                print(f"Validating cell type: {celltype}")
                print(f"Number of samples: {len(valid_loader.dataset)}")
                print(f"Number of batches: {len(valid_loader)}")
                print(f"Batch size: {valid_loader.batch_size}")

                print("-"*50)

                for batch_idx, batch in enumerate(valid_loader):
                    print("Epoch: {}, Batch: {}".format(epoch,batch_idx))
                    data = batch['data']
                    data = data.to(device)

                    encoded, decoded, mu, log_var = model(data)
                    all_loss = model.loss_function(decoded, data, mu, log_var, args)

                    loss = all_loss[0]
                    recons_loss = all_loss[1]
                    JSD_div = all_loss[2]
                    # L1_reg = all_loss[3]

                    print("loss: ",loss, recons_loss, JSD_div)

                    print("data.size(0): ",data.size(0))

                    valid_total_loss += loss.item()*data.size(0)
                    valid_total_recons_loss += recons_loss.item()*data.size(0)
                    valid_total_JSD_div += JSD_div.item()*data.size(0)
                    # valid_total_L1_reg += L1_reg.item()*data.size(0)

                    ############# Save model output ################

                    data_size = int(data.size(0))
                    data = data.cpu().detach().numpy()
                    encoded = encoded.cpu().detach().numpy()
                    decoded = decoded.cpu().detach().numpy()
                    loss = loss.cpu().detach().numpy()
                    recons_loss = recons_loss.cpu().detach().numpy()
                    JSD_div = JSD_div.cpu().detach().numpy()
                    # L1_reg = L1_reg.cpu().detach().numpy()

                    celltypes = batch["cell_type"].cpu().detach().numpy()
                    meta_batch_id = batch["batch_id"].cpu().detach().numpy()
                    # For the validation part
                    original_valid_cell_types = []
                    for cell_code in batch["cell_type"].cpu().numpy():
                        original_valid_cell_types.append(valid_loader.dataset.get_original_value('cell_type', cell_code))

                    original_valid_batch_ids = []
                    if valid_loader.dataset.has_batch_id:
                        for batch_code in batch["batch_id"].cpu().numpy():
                            original_valid_batch_ids.append(valid_loader.dataset.get_original_value('batch_id', batch_code))


                    valid_epoch_save_dict[celltype][valid_batch_count] = {} 
                    valid_epoch_save_dict[celltype][valid_batch_count]["cell_type"] = celltypes.tolist()
                    valid_epoch_save_dict[celltype][valid_batch_count]["cell_type_original"] = original_valid_cell_types
                    valid_epoch_save_dict[celltype][valid_batch_count]["meta_batch_id"] = meta_batch_id.tolist()
                    valid_epoch_save_dict[celltype][valid_batch_count]["meta_batch_id_original"] = original_valid_batch_ids
                    valid_epoch_save_dict[celltype][valid_batch_count]["encoded"] = encoded.tolist()
                    valid_epoch_save_dict[celltype][valid_batch_count]["decoded"] = decoded.tolist()
                    valid_epoch_save_dict[celltype][valid_batch_count]["mu"] = mu.tolist()
                    valid_epoch_save_dict[celltype][valid_batch_count]["log_var"] = log_var.tolist()
                    valid_epoch_save_dict[celltype][valid_batch_count]["loss"] = loss.tolist()
                    valid_epoch_save_dict[celltype][valid_batch_count]["recons_loss"] = recons_loss.tolist()
                    valid_epoch_save_dict[celltype][valid_batch_count]["JSD_div"] = JSD_div.tolist()
                    # json_data = json.dumps(cell_print)    

       

                    valid_this_epoch_all_loss.append([celltype,batch_idx,loss,recons_loss,JSD_div,data_size])
                    valid_batch_count +=1
                
                    valid_num_data+=len(valid_loader.dataset)

            valid_epoch_loss = valid_total_loss / valid_num_data
            valid_epoch_recons_loss = valid_total_recons_loss / valid_num_data
            valid_epoch_JSD_div_loss = valid_total_JSD_div / valid_num_data
            # valid_epoch_L1_reg = valid_total_L1_reg / valid_num_data
            # valid_loss_list.append([epoch,valid_epoch_loss,valid_epoch_recons_loss,valid_epoch_KL_div_loss,valid_epoch_L1_reg])

            #============= Saving Validation loss ====================
            # save the epoch save dict every 10 epochs and last epoch
            if epoch % 50 == 0 or epoch == args.num_epochs - 1:
                print(f"validation results saved on epoch: {epoch}")
                os.makedirs(os.path.join(save_path,"epoch_dump"), exist_ok=True)
                valid_file_path = os.path.join(save_path,"epoch_dump","valid.json".format(epoch))
                with open(valid_file_path, 'w') as fp:
                    json.dump(valid_epoch_save_dict, fp)
            
            if epoch % 50 == 0 or epoch == args.num_epochs - 1:
                plot_umap_after_train_by_celltype(save_path,feature="encoded",data_split= "valid", epoch = epoch, file_path=valid_file_path)
                plot_umap_after_train_by_celltype(save_path,feature="decoded",data_split= "valid", epoch = epoch, file_path=valid_file_path)
            
            os.makedirs(os.path.join(save_path, "valid_loss"), exist_ok=True)
            valid_loss_file = os.path.join(save_path, "valid_loss","valid_batch_loss.txt")
            header = ["epoch","batch_idx","loss","recons_loss","JSD_div","data_size"]
            
            file_exists = os.path.isfile(valid_loss_file)

            with open(valid_loss_file, 'a' if file_exists else 'w') as file:
                if not file_exists:
                    file.write(','.join(header) + '\n')

                # Append the new line with the epoch data
                for row in valid_this_epoch_all_loss:
                    file.write('\t'.join(map(str, row)) + '\n')
                
            header = ["epoch","loss","recons_loss","JSD_div"]
            os.makedirs(os.path.join(save_path,"valid_loss"), exist_ok=True)
            loss_file = os.path.join(save_path,"valid_loss","valid_epoch_loss.txt".format(epoch))

            file_exists = os.path.isfile(loss_file)
            print("save epoch loss info")
            with open(loss_file, 'a' if file_exists else 'w') as file:
                if not file_exists:
                    file.write('\t'.join(header) + '\n')
                
                line = '\t'.join(map(str, [epoch,valid_epoch_loss,valid_epoch_recons_loss,valid_epoch_JSD_div_loss]))
                file.write(line + '\n')


   # plot loss curve
   #convert loss_list to dict
    loss_list_temp = loss_list.copy()[1:]
    train_loss_dict = {
        "loss": [item[1] for item in loss_list_temp],
        "recons_loss": [item[2] for item in loss_list_temp],
        "JSD_div": [item[3] for item in loss_list_temp]
    }
    plot_loss_after_train_general(train_loss_dict, os.path.join(save_path, "loss_curves"))

    return model,loss_list

def train_loop(cell_types, cellsfile, save_path,args):
    
    # with open(cellsfile, 'r') as file:
    #     row_count = sum(1 for row in file) - 1
    print("number of features: ",args.num_feature)
    
    model = VAE(input_size = args.num_feature, num_internal = args.num_internal, num_latent = args.num_latent)


    model_str = str(model)

    # Writing to a text file
    with open('{}/config.txt'.format(save_path), 'w') as f:
        f.write(model_str)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model.to(device)
    loss_list = []
    loss_list.append(["epoch","loss","recons_loss","KL_div"])
    valid_loss_list = []


    save_count = 0
    for epoch in range(args.num_epochs):
        start_time = time.time()
        print("======================Epoch {}===================".format(epoch))
        total_loss = 0.0
        total_recons_loss = 0.0
        total_KL_div = 0.0
        total_L1_reg = 0.0

        num_data = 0
        # debug_path = os.path.join(save_path,"debug")
        # os.makedirs(debug_path, exist_ok=True)
        all_cell_all_loss = []
        batch_count = 0
        for cell in cell_types:
            cell_save_dict = {}
            
            cell_print = cell.replace(" ","-")
            cell_print = cell_print.replace("/","-")
            train_loader_path = "dataloders/{}_train_{}.pkl".format(cell_print,args.norm)
            valid_loader_path = "dataloders/{}_valid_{}.pkl".format(cell_print,args.norm)
            test_loader_path = "dataloders/{}_test_{}.pkl".format(cell_print,args.norm)

            if os.path.exists(train_loader_path) and os.path.exists(valid_loader_path) and os.path.exists(test_loader_path):
                print("loading existing dataloader objects...")
                train_loader = torch.load(train_loader_path)
                valid_loader = torch.load(valid_loader_path)
                test_loader = torch.load(test_loader_path)
            else:
                cells_chunks = pd.read_csv(cellsfile, sep="\t", header=0, usecols = lambda x: x.split(".")[0]==cell)
                transposed_cells_chunks = cells_chunks.T 
                X_all = transposed_cells_chunks.values
                if args.norm == "minmax":
                    X_all = minmax_normalize_data(X_all)
                elif args.norm == "log":
                    X_all = log_transform_data(X_all)
            
                    
            
                ############ Train, validate, test split ###################
                num_train = int(len(X_all)*0.6)
                num_valid = int(len(X_all)*0.2)
                num_test = len(X_all) - num_train - num_valid

                X_train = X_all[:num_train]
                X_valid = X_all[num_train:num_train+num_valid]
                X_test = X_all[num_train+num_valid:]
                
                X_train = torch.from_numpy(X_train).float().to(device)
                X_valid = torch.from_numpy(X_valid).float().to(device)
                X_test = torch.from_numpy(X_test).float().to(device)

                train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=False)
                valid_loader = torch.utils.data.DataLoader(X_valid, batch_size=batch_size, shuffle=False)
                test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=False)

                torch.save(train_loader, train_loader_path)
                torch.save(valid_loader, valid_loader_path)
                torch.save(test_loader, test_loader_path)


            for batch_idx, data in enumerate(train_loader):

                print("Epoch: {}, Cell: {}, Batch: {}".format(epoch,cell,batch_idx))
                data = data.to(device)
                encoded, decoded, mu, log_var = model(data)
                # loss, recons_loss, KL_div  = model.loss_function(decoded, data, mu, log_var, beta)
                loss,recons_loss, KL_div= model.loss_function(decoded, data, mu, log_var, args)
                # loss = loss_list[0]
                # recons_loss = loss_list[1]
                # KL_div = loss_list[2]
                print("3 loss: ",loss, recons_loss, KL_div)
                optimizer.zero_grad()
                loss.backward()

                
                

                # ################################################
                optimizer.step()
                # print("data.size(0): ",data.size(0))
                total_loss += loss.item()*data.size(0)
                total_recons_loss += recons_loss.item()*data.size(0)
                total_KL_div += KL_div.item()*data.size(0)
                
                ############# Save model output ################

                data_size = int(data.size(0))
                data = data.cpu().detach().numpy()
                encoded = encoded.cpu().detach().numpy()
                decoded = decoded.cpu().detach().numpy()
                mu = mu.cpu().detach().numpy()
                log_var = log_var.cpu().detach().numpy()
                loss = loss.cpu().detach().numpy()
                recons_loss = recons_loss.cpu().detach().numpy()
                KL_div = KL_div.cpu().detach().numpy()

                # print("cell: ",cell_print)
                # print("batch_idx: ",batch_idx)
                # print("data.tolist(): ",data.tolist())
                # print("encoded.tolist() ",encoded.tolist())
                # print("decoded.tolist(): ",decoded.tolist())
                # print("mu: ",mu)
                # print("log_var: ",log_var)
                # print("loss: ",loss)
                # print("recons_loss: ",recons_loss)
                # print("KL_div: ",KL_div)

                cell_save_dict[batch_count] = {}
                cell_save_dict[batch_count]["cell"] = cell_print
                # json_data = json.dumps(cell_print)
                cell_save_dict[batch_count]["batch_idx"] = batch_idx
                # json_data = json.dumps(batch_idx)
                # cell_save_dict[batch_count]["data"] = data.tolist()
                # json_data = json.dumps(data.tolist())
                cell_save_dict[batch_count]["encoded"] = encoded.tolist()
                # json_data = json.dumps(encoded.tolist())
                cell_save_dict[batch_count]["decoded"] = decoded.tolist()
                # json_data = json.dumps(decoded.tolist())
                cell_save_dict[batch_count]["mu"] = mu.tolist()
                # json_data = json.dumps(mu.tolist())
                cell_save_dict[batch_count]["log_var"] = log_var.tolist()
                # json_data = json.dumps(log_var.tolist())
                cell_save_dict[batch_count]["loss"] = loss.tolist()
                # json_data = json.dumps(loss.tolist())
                cell_save_dict[batch_count]["recons_loss"] = recons_loss.tolist()
                # json_data = json.dumps(recons_loss.tolist())
                cell_save_dict[batch_count]["KL_div"] = KL_div.tolist()
                # json_data = json.dumps(KL_div.tolist())

                
                all_cell_all_loss.append([batch_count,cell,batch_idx,loss,recons_loss,KL_div,data_size])
                
                batch_count +=1
            
            num_data += len(train_loader.dataset)
            print("num data",num_data)

    
            # ============Save model output, loss for each batch in one file==========
            os.makedirs(os.path.join(save_path,"epoch_dump"), exist_ok=True)
            with open(os.path.join(save_path,"epoch_dump","epoch_{}_cell_{}_train.json".format(epoch,cell_print)), 'w') as fp:
                json.dump(cell_save_dict, fp)

        os.makedirs(os.path.join(save_path,"train_batch_loss"), exist_ok=True)
        with open(os.path.join(save_path,"train_batch_loss","epoch_{}_train_loss.txt".format(epoch)), 'w') as fp:
            header = ["batch_count","cell","batch_idx","loss","recons_loss","KL_div","data_size"]
            fp.write('\t'.join(header) + '\n')
            for row in all_cell_all_loss:
                fp.write('\t'.join(map(str, row)) + '\n')

        #=============Saving Epoch loss, model, print run time====================
        epoch_loss = total_loss / num_data
        epoch_recons_loss = total_recons_loss / num_data
        epoch_KL_div_loss = total_KL_div / num_data

        loss_list.append([epoch,epoch_loss,epoch_recons_loss,epoch_KL_div_loss])

        print("saving model.pt")
        torch.save(model.state_dict(),os.path.join(save_path,"model_{}.pt".format(epoch)))

        end_time = time.time()
        epoch_duration = end_time - start_time
        hours = int(epoch_duration // 3600)
        minutes = int((epoch_duration % 3600) // 60)
        seconds = epoch_duration % 60
        print(f"Epoch {epoch+1}/{args.num_epochs} completed in {hours}h:{minutes}m:{seconds:.2f}s.")
        print(f"loss={epoch_loss}, recon_loss={epoch_recons_loss}, KL_div={epoch_recons_loss}")

        
        print("save epoch trainning loss info")
        os.makedirs(os.path.join(save_path,"train_epoch_loss"), exist_ok=True)
        loss_file = os.path.join(save_path,"train_epoch_loss","loss_{}.txt".format(epoch))
        header = "epoch,loss,recons_loss,KL_div"
        with open(loss_file, 'w') as file:
            file.write(header + '\n')  
            line = ','.join(map(str, [epoch,epoch_loss,epoch_recons_loss,epoch_KL_div_loss]))
            file.write(line + '\n')


        # ===================== Validate Model after epoch =================================
        if args.valid:
            valid_total_loss = 0.0
            valid_total_recons_loss = 0.0
            valid_total_KL_div = 0.0

            valid_num_data = 0
            # debug_path = os.path.join(save_path,"debug")
            # os.makedirs(debug_path, exist_ok=True)
            valid_batch_count = 0
            valid_epoch_save_dict = {}
            valid_this_epoch_all_loss = []

            print("========= Epoch {} Validation ===========".format(epoch))
            for cell in cell_types:
                valid_loader_path = "dataloders/{}_valid_{}.pkl".format(cell_print,args.norm)
                valid_loader = torch.load(valid_loader_path)
                
                for batch_idx, data in enumerate(valid_loader):
                    print("Cell: {}, Batch: {}".format(epoch,cell,batch_idx))
                    data = data.to(device)
                    encoded, decoded, mu, log_var = model(data)
                    loss, recons_loss, KL_div = model.loss_function(decoded, data, mu, log_var, args)

                    print("3 loss: ",loss, recons_loss, KL_div)

                    print("data.size(0): ",data.size(0))
                    
                    valid_total_loss += loss.item()*data.size(0)
                    valid_total_recons_loss += recons_loss.item()*data.size(0)
                    valid_total_KL_div += KL_div.item()*data.size(0)
                    
                    ############# Save model output ################

                    data_size = int(data.size(0))
                    data = data.cpu().detach().numpy()
                    encoded = encoded.cpu().detach().numpy()
                    decoded = decoded.cpu().detach().numpy()
                    loss = loss.cpu().detach().numpy()
                    recons_loss = recons_loss.cpu().detach().numpy()
                    KL_div = KL_div.cpu().detach().numpy()

                    valid_epoch_save_dict[valid_batch_count] = {}
                    valid_epoch_save_dict[valid_batch_count]["cell"] = cell_print
                    json_data = json.dumps(cell_print)
                    valid_epoch_save_dict[valid_batch_count]["batch_idx"] = batch_idx
                    json_data = json.dumps(batch_idx)
                    # valid_epoch_save_dict[valid_batch_count]["data"] = data.tolist()
                    json_data = json.dumps(data.tolist())
                    valid_epoch_save_dict[valid_batch_count]["encoded"] = encoded.tolist()
                    json_data = json.dumps(encoded.tolist())
                    valid_epoch_save_dict[valid_batch_count]["decoded"] = decoded.tolist()
                    json_data = json.dumps(decoded.tolist())
                    valid_epoch_save_dict[valid_batch_count]["mu"] = mu.tolist()
                    json_data = json.dumps(mu.tolist())
                    valid_epoch_save_dict[valid_batch_count]["log_var"] = log_var.tolist()
                    json_data = json.dumps(log_var.tolist())
                    valid_epoch_save_dict[valid_batch_count]["loss"] = loss.tolist()
                    json_data = json.dumps(loss.tolist())
                    valid_epoch_save_dict[valid_batch_count]["recons_loss"] = recons_loss.tolist()
                    # json_data = json.dumps(recons_loss.tolist())
                    valid_epoch_save_dict[valid_batch_count]["KL_div"] = KL_div.tolist()
                    # json_data = json.dumps(KL_div.tolist())

                    valid_this_epoch_all_loss.append([valid_batch_count,cell,batch_idx,loss,recons_loss,KL_div,data_size])
                    valid_batch_count +=1
                
                valid_num_data+=len(train_loader.dataset)

            valid_epoch_loss = valid_total_loss / valid_num_data
            valid_epoch_recons_loss = valid_total_recons_loss / valid_num_data
            valid_epoch_KL_div_loss = valid_total_KL_div / valid_num_data
            valid_loss_list.append([epoch,valid_epoch_loss,valid_epoch_recons_loss,valid_epoch_KL_div_loss])

            #============= Saving Validation loss ====================
            os.makedirs(os.path.join(save_path,"epoch_dump"), exist_ok=True)
            with open(os.path.join(save_path,"epoch_dump","epoch_{}_valid.json".format(epoch)), 'w') as fp:
                json.dump(valid_epoch_save_dict, fp)

            os.makedirs(os.path.join(save_path,"valid_batch_loss"), exist_ok=True)
            with open(os.path.join(save_path,"valid_batch_loss","epoch_{}_valid_loss.txt".format(epoch)), 'w') as fp:
                header = ["batch_count","cell","batch_idx","loss","recons_loss","KL_div","data_size"]
                fp.write('\t'.join(header) + '\n')
                for row in valid_this_epoch_all_loss:
                    fp.write('\t'.join(map(str, row)) + '\n')
                
            header = "epoch,loss,recons_loss,KL_div"
            os.makedirs(os.path.join(save_path,"valid_epoch_loss"), exist_ok=True)
            loss_file = os.path.join(save_path,"valid_epoch_loss","loss_{}.txt".format(epoch))
            print("save epoch loss info")
            with open(loss_file, 'w') as file:
                file.write(header + '\n')  
                line = ','.join(map(str, [epoch,epoch_loss,epoch_recons_loss,epoch_KL_div_loss]))
                file.write(line + '\n')

    return model,loss_list

def train_loop_test(cell_types, cellsfile, save_path, args):
    
    # with open(cellsfile, 'r') as file:
    #     row_count = sum(1 for row in file) - 1
    n_features = args.num_feature
    print("number of features: ",n_features)
    
    model = VAE_L1(input_size = args.num_feature, num_internal = args.num_internal, num_latent = args.num_latent,l1_coef = args.l1_coef)
    
    model_str = str(model)

    # Writing to a text file
    with open('{}/config.txt'.format(save_path), 'w') as f:
        f.write(model_str)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model.to(device)
    loss_list = []
    loss_list.append(["epoch","loss","recons_loss","KL_div","L1_reg"])
    valid_loss_list = []

    # num_epochs = 0
    # save_count = 0
    for epoch in range(args.num_epochs):
        start_time = time.time()
        print("======================Epoch {}===================".format(epoch))

        total_train = 0
        total_valid = 0
        total_test = 0
        for cell in cell_types:
            cell_save_dict = {}
            this_cell_all_loss = []
            cell_print = cell.replace(" ","-")
            cell_print = cell_print.replace("/","-")

            cells_chunks = pd.read_csv(cellsfile, sep="\t", header=0, usecols = lambda x: x.split(".")[0]==cell)
            transposed_cells_chunks = cells_chunks.T 
            X_all = transposed_cells_chunks.values
            if args.norm == "minmax":
                X_all = minmax_normalize_data(X_all)
            elif args.norm == "log":
                X_all = log_transform_data(X_all)
            
                    
                ############ Train, validate, test split ###################
            num_train = int(len(X_all)*0.6)
            num_valid = int(len(X_all)*0.2)
            num_test = len(X_all) - num_train - num_valid

            print(num_train,num_valid,num_test)
            total_train += num_train
            total_valid += num_valid
            total_test += num_test

        print("==============================")
        print(total_train,total_valid,total_test)
        print(total_train+total_valid+total_test)

        exit()
    return model,loss_list


def train_loop_L1(cell_types, cellsfile, save_path,args):
    
    # with open(cellsfile, 'r') as file:
    #     row_count = sum(1 for row in file) - 1

    print("number of features: ",args.num_feature)
    
    model = VAE_L1(input_size = args.num_feature, num_internal = args.num_internal, num_latent = args.num_latent,l1_coef = args.l1_coef)
    
    model_str = str(model)

    # Writing to a text file
    with open('{}/config.txt'.format(save_path), 'w') as f:
        f.write(model_str)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model.to(device)
    loss_list = []
    loss_list.append(["epoch","loss","recons_loss","KL_div","L1_reg"])
    valid_loss_list = []


    save_count = 0
    for epoch in range(args.num_epochs):
        start_time = time.time()
        print("======================Epoch {}===================".format(epoch))
        total_loss = 0.0
        total_recons_loss = 0.0
        total_KL_div = 0.0
        total_L1_reg = 0.0

        num_data = 0
        # debug_path = os.path.join(save_path,"debug")
        # os.makedirs(debug_path, exist_ok=True)
        batch_count = 0





        for cell in cell_types:
            cell_save_dict = {}
            this_cell_all_loss = []
            cell_print = cell.replace(" ","-")
            cell_print = cell_print.replace("/","-")
            train_loader_path = "dataloders/{}_train_{}.pkl".format(cell_print,args.norm)
            valid_loader_path = "dataloders/{}_valid_{}.pkl".format(cell_print,args.norm)
            test_loader_path = "dataloders/{}_test_{}.pkl".format(cell_print,args.norm)

            if os.path.exists(train_loader_path) and os.path.exists(valid_loader_path) and os.path.exists(test_loader_path):
                print("loading existing dataloader objects...")
                train_loader = torch.load(train_loader_path)
                valid_loader = torch.load(valid_loader_path)
                test_loader = torch.load(test_loader_path)
            else:
                cells_chunks = pd.read_csv(cellsfile, sep="\t", header=0, usecols = lambda x: x.split(".")[0]==cell)
                transposed_cells_chunks = cells_chunks.T 
                X_all = transposed_cells_chunks.values
                if args.norm == "minmax":
                    X_all = minmax_normalize_data(X_all)
                elif args.norm == "log":
                    X_all = log_transform_data(X_all)
            
                    
            
                ############ Train, validate, test split ###################
                num_train = int(len(X_all)*0.6)
                num_valid = int(len(X_all)*0.2)
                num_test = len(X_all) - num_train - num_valid

                print(num_train,num_valid,num_test)

                X_train = X_all[:num_train]
                X_valid = X_all[num_train:num_train+num_valid]
                X_test = X_all[num_train+num_valid:]
                
                X_train = torch.from_numpy(X_train).float().to(device)
                X_valid = torch.from_numpy(X_valid).float().to(device)
                X_test = torch.from_numpy(X_test).float().to(device)

                train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=False)
                valid_loader = torch.utils.data.DataLoader(X_valid, batch_size=batch_size, shuffle=False)
                test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=False)

                torch.save(train_loader, train_loader_path)
                torch.save(valid_loader, valid_loader_path)
                torch.save(test_loader, test_loader_path)


            for batch_idx, data in enumerate(train_loader):

                print("Epoch: {}, Cell: {}, Batch: {}".format(epoch,cell,batch_idx))
                data = data.to(device)
                encoded, decoded, mu, log_var = model(data)
                # loss, recons_loss, KL_div  = model.loss_function(decoded, data, mu, log_var, beta)
                all_loss = model.loss_function(decoded, data, mu, log_var, args)
                loss = all_loss[0]
                recons_loss = all_loss[1]
                KL_div = all_loss[2]
                L1_reg = all_loss[3]
                print("3 loss: ",loss, recons_loss, KL_div)
                optimizer.zero_grad()
                loss.backward()

                
                

                # ################################################
                optimizer.step()
                # print("data.size(0): ",data.size(0))
                total_loss += loss.item()*data.size(0)
                total_recons_loss += recons_loss.item()*data.size(0)
                total_KL_div += KL_div.item()*data.size(0)
                
                total_L1_reg += L1_reg.item()*data.size(0)
                
                ############# Save model output ################

                data_size = int(data.size(0))
                data = data.cpu().detach().numpy()
                encoded = encoded.cpu().detach().numpy()
                decoded = decoded.cpu().detach().numpy()
                mu = mu.cpu().detach().numpy()
                log_var = log_var.cpu().detach().numpy()
                loss = loss.cpu().detach().numpy()
                recons_loss = recons_loss.cpu().detach().numpy()
                KL_div = KL_div.cpu().detach().numpy()
                L1_reg = L1_reg.cpu().detach().numpy()

                # print("cell: ",cell_print)
                # print("batch_idx: ",batch_idx)
                # print("data.tolist(): ",data.tolist())
                # print("encoded.tolist() ",encoded.tolist())
                # print("decoded.tolist(): ",decoded.tolist())
                # print("mu: ",mu)
                # print("log_var: ",log_var)
                # print("loss: ",loss)
                # print("recons_loss: ",recons_loss)
                # print("KL_div: ",KL_div)

                cell_save_dict[batch_count] = {}
                cell_save_dict[batch_count]["cell"] = cell_print
                # json_data = json.dumps(cell_print)
                cell_save_dict[batch_count]["batch_idx"] = batch_idx
                # json_data = json.dumps(batch_idx)
                # cell_save_dict[batch_count]["data"] = data.tolist()
                # json_data = json.dumps(data.tolist())
                cell_save_dict[batch_count]["encoded"] = encoded.tolist()
                # json_data = json.dumps(encoded.tolist())
                cell_save_dict[batch_count]["decoded"] = decoded.tolist()
                # json_data = json.dumps(decoded.tolist())
                cell_save_dict[batch_count]["mu"] = mu.tolist()
                # json_data = json.dumps(mu.tolist())
                cell_save_dict[batch_count]["log_var"] = log_var.tolist()
                # json_data = json.dumps(log_var.tolist())
                cell_save_dict[batch_count]["loss"] = loss.tolist()
                # json_data = json.dumps(loss.tolist())
                cell_save_dict[batch_count]["recons_loss"] = recons_loss.tolist()
                # json_data = json.dumps(recons_loss.tolist())
                cell_save_dict[batch_count]["KL_div"] = KL_div.tolist()
                # json_data = json.dumps(KL_div.tolist())

                cell_save_dict[batch_count]["L1_reg"] = L1_reg.tolist()

                this_cell_all_loss.append([batch_count,cell,batch_idx,loss,recons_loss,KL_div,L1_reg,data_size])
                
                batch_count +=1

            
            num_data += len(train_loader.dataset)
            print("num data",num_data)

    
            # ============Save model output, loss for each batch in one file==========
            os.makedirs(os.path.join(save_path,"epoch_dump"), exist_ok=True)
            with open(os.path.join(save_path,"epoch_dump","epoch_{}_cell_{}_train.json".format(epoch,cell_print)), 'w') as fp:
                json.dump(cell_save_dict, fp)

            os.makedirs(os.path.join(save_path,"train_batch_loss"), exist_ok=True)
            with open(os.path.join(save_path,"train_batch_loss","epoch_{}_cell_{}_train_loss.txt".format(epoch,cell_print)), 'w') as fp:
                header = ["batch_count","cell","batch_idx","loss","recons_loss","KL_div","L1_reg","data_size"]
                fp.write('\t'.join(header) + '\n')
                for row in this_cell_all_loss:
                    fp.write('\t'.join(map(str, row)) + '\n')

        #=============Saving Epoch loss, model, print run time====================
        epoch_loss = total_loss / num_data
        epoch_recons_loss = total_recons_loss / num_data
        epoch_KL_div_loss = total_KL_div / num_data
        epoch_L1_reg = total_L1_reg / num_data
        
        loss_list.append([epoch,epoch_loss,epoch_recons_loss,epoch_KL_div_loss,epoch_L1_reg])

        print("saving model.pt")
        torch.save(model.state_dict(),os.path.join(save_path,"model_{}.pt".format(epoch)))

        end_time = time.time()
        epoch_duration = end_time - start_time
        hours = int(epoch_duration // 3600)
        minutes = int((epoch_duration % 3600) // 60)
        seconds = epoch_duration % 60
        print(f"Epoch {epoch+1}/{args.num_epochs} completed in {hours}h:{minutes}m:{seconds:.2f}s.")
        print(f"loss={epoch_loss}, recon_loss={epoch_recons_loss}, KL_div={epoch_recons_loss}")

        
        print("save epoch trainning loss info")
        os.makedirs(os.path.join(save_path,"train_epoch_loss"), exist_ok=True)
        loss_file = os.path.join(save_path,"train_epoch_loss","loss_{}.txt".format(epoch))
        header = "epoch,loss,recons_loss,KL_div"
        with open(loss_file, 'w') as file:
            file.write(header + '\n')  
            line = ','.join(map(str, [epoch,epoch_loss,epoch_recons_loss,epoch_KL_div_loss,epoch_L1_reg]))
            file.write(line + '\n')


        # ===================== Validate Model after epoch =================================
        if args.valid:
            valid_total_loss = 0.0
            valid_total_recons_loss = 0.0
            valid_total_KL_div = 0.0
            valid_total_L1_reg = 0.0

            valid_num_data = 0
            # debug_path = os.path.join(save_path,"debug")
            # os.makedirs(debug_path, exist_ok=True)
            valid_batch_count = 0
            valid_epoch_save_dict = {}
            valid_this_epoch_all_loss = []

            print("========= Epoch {} Validation ===========".format(epoch))
            for cell in cell_types:
                valid_loader_path = "dataloders/{}_valid_{}.pkl".format(cell_print,args.norm)
                valid_loader = torch.load(valid_loader_path)
                
                for batch_idx, data in enumerate(valid_loader):
                    print("Cell: {}, Batch: {}".format(epoch,cell,batch_idx))
                    data = data.to(device)
                    encoded, decoded, mu, log_var = model(data)
                    all_loss = model.loss_function(decoded, data, mu, log_var, args)
                    loss = all_loss[0]
                    recons_loss = all_loss[1]
                    KL_div = all_loss[2]
                    L1_reg = all_loss[3]
                
                    # print("4 loss: ",loss, recons_loss, KL_div,L1_reg)

                    print("data.size(0): ",data.size(0))
                    
                    valid_total_loss += loss.item()*data.size(0)
                    valid_total_recons_loss += recons_loss.item()*data.size(0)
                    valid_total_KL_div += KL_div.item()*data.size(0)
                    valid_total_L1_reg += L1_reg.item()*data.size(0)
                    
                    ############# Save model output ################

                    data_size = int(data.size(0))
                    data = data.cpu().detach().numpy()
                    encoded = encoded.cpu().detach().numpy()
                    decoded = decoded.cpu().detach().numpy()
                    loss = loss.cpu().detach().numpy()
                    recons_loss = recons_loss.cpu().detach().numpy()
                    KL_div = KL_div.cpu().detach().numpy()
                    L1_reg = L1_reg.cpu().detach().numpy()

                    valid_epoch_save_dict[valid_batch_count] = {}
                    valid_epoch_save_dict[valid_batch_count]["cell"] = cell_print
                    json_data = json.dumps(cell_print)
                    valid_epoch_save_dict[valid_batch_count]["batch_idx"] = batch_idx
                    json_data = json.dumps(batch_idx)
                    # valid_epoch_save_dict[valid_batch_count]["data"] = data.tolist()
                    json_data = json.dumps(data.tolist())
                    valid_epoch_save_dict[valid_batch_count]["encoded"] = encoded.tolist()
                    json_data = json.dumps(encoded.tolist())
                    valid_epoch_save_dict[valid_batch_count]["decoded"] = decoded.tolist()
                    json_data = json.dumps(decoded.tolist())
                    valid_epoch_save_dict[valid_batch_count]["mu"] = mu.tolist()
                    json_data = json.dumps(mu.tolist())
                    valid_epoch_save_dict[valid_batch_count]["log_var"] = log_var.tolist()
                    json_data = json.dumps(log_var.tolist())
                    valid_epoch_save_dict[valid_batch_count]["loss"] = loss.tolist()
                    json_data = json.dumps(loss.tolist())
                    valid_epoch_save_dict[valid_batch_count]["recons_loss"] = recons_loss.tolist()
                    # json_data = json.dumps(recons_loss.tolist())
                    valid_epoch_save_dict[valid_batch_count]["KL_div"] = KL_div.tolist()
                    # json_data = json.dumps(KL_div.tolist())
                    valid_epoch_save_dict[valid_batch_count]["L1_reg"] = L1_reg.tolist()

                    

                
                    valid_this_epoch_all_loss.append([valid_batch_count,cell,batch_idx,loss,recons_loss,KL_div,L1_reg,data_size])
                    valid_batch_count +=1
                
                valid_num_data+=len(train_loader.dataset)

            valid_epoch_loss = valid_total_loss / valid_num_data
            valid_epoch_recons_loss = valid_total_recons_loss / valid_num_data
            valid_epoch_KL_div_loss = valid_total_KL_div / valid_num_data
            valid_epoch_L1_reg = valid_total_L1_reg / valid_num_data
            valid_loss_list.append([epoch,valid_epoch_loss,valid_epoch_recons_loss,valid_epoch_KL_div_loss,valid_epoch_L1_reg])
            print(f"Validation Loss: total loss: {valid_epoch_loss}, recons:{valid_epoch_recons_loss},KL: {valid_epoch_KL_div_loss}")
            #============= Saving Validation loss ====================
            os.makedirs(os.path.join(save_path,"epoch_dump"), exist_ok=True)
            with open(os.path.join(save_path,"epoch_dump","epoch_{}_valid.json".format(epoch)), 'w') as fp:
                json.dump(valid_epoch_save_dict, fp)

            os.makedirs(os.path.join(save_path,"valid_batch_loss"), exist_ok=True)
            with open(os.path.join(save_path,"valid_batch_loss","epoch_{}_valid_loss.txt".format(epoch)), 'w') as fp:
                header = ["batch_count","cell","batch_idx","loss","recons_loss","KL_div","L1_reg","data_size"]
                fp.write('\t'.join(header) + '\n')
                for row in valid_this_epoch_all_loss:
                    fp.write('\t'.join(map(str, row)) + '\n')
                
            header = "epoch,loss,recons_loss,KL_div"
            os.makedirs(os.path.join(save_path,"valid_epoch_loss"), exist_ok=True)
            loss_file = os.path.join(save_path,"valid_epoch_loss","loss_{}.txt".format(epoch))
            # print("Saving epoch validation loss info")
            with open(loss_file, 'w') as file:
                file.write(header + '\n')  
                line = ','.join(map(str, [epoch,valid_epoch_loss,valid_epoch_recons_loss,valid_epoch_KL_div_loss,valid_epoch_L1_reg]))
                file.write(line + '\n')


    return model,loss_list

def train_loop_SL(cell_types, cellsfile, save_path,args):

    
    
    # with open(cellsfile, 'r') as file:
    #     row_count = sum(1 for row in file) - 1
    n_features = num_feature
    print("number of features: ",n_features)
    
    model = VAE_L1(input_size = n_features, num_internal = num_internal, num_latent = num_latent)
    
    model_str = str(model)

    # Writing to a text file
    with open('{}/config.txt'.format(save_path), 'w') as f:
        f.write(model_str)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    loss_list = []
    valid_loss_list = []


    save_count = 0
    for epoch in range(args.num_epochs):
        start_time = time.time()
        print("======================Epoch {}===================".format(epoch))
        total_loss = 0.0
        total_recons_loss = 0.0
        total_KL_div = 0.0
        total_L1_reg = 0.0

        num_data = 0
        # debug_path = os.path.join(save_path,"debug")
        # os.makedirs(debug_path, exist_ok=True)
        batch_count = 0
        for cell in cell_types:
            cell_save_dict = {}
            this_cell_all_loss = []
            cell_print = cell.replace(" ","-")
            cell_print = cell_print.replace("/","-")
            train_loader_path = "dataloders/{}_train_{}.pkl".format(cell_print,args.norm)
            valid_loader_path = "dataloders/{}_valid_{}.pkl".format(cell_print,args.norm)
            test_loader_path = "dataloders/{}_test_{}.pkl".format(cell_print,args.norm)

            if os.path.exists(train_loader_path) and os.path.exists(valid_loader_path) and os.path.exists(test_loader_path):
                print("loading existing dataloader objects...")
                train_loader = torch.load(train_loader_path)
                valid_loader = torch.load(valid_loader_path)
                test_loader = torch.load(test_loader_path)
            else:
                cells_chunks = pd.read_csv(cellsfile, sep="\t", header=0, usecols = lambda x: x.split(".")[0]==cell)
                transposed_cells_chunks = cells_chunks.T 
                X_all = transposed_cells_chunks.values
                if args.norm == "minmax":
                    X_all = minmax_normalize_data(X_all)
                elif args.norm == "log":
                    X_all = log_transform_data(X_all)
            
                    
            
                ############ Train, validate, test split ###################
                num_train = int(len(X_all)*0.6)
                num_valid = int(len(X_all)*0.2)
                num_test = len(X_all) - num_train - num_valid

                X_train = X_all[:num_train]
                X_valid = X_all[num_train:num_train+num_valid]
                X_test = X_all[num_train+num_valid:]
                
                X_train = torch.from_numpy(X_train).float().to(device)
                X_valid = torch.from_numpy(X_valid).float().to(device)
                X_test = torch.from_numpy(X_test).float().to(device)

                train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=False)
                valid_loader = torch.utils.data.DataLoader(X_valid, batch_size=batch_size, shuffle=False)
                test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=False)

                torch.save(train_loader, train_loader_path)
                torch.save(valid_loader, valid_loader_path)
                torch.save(test_loader, test_loader_path)


            for batch_idx, data in enumerate(train_loader):

                print("Epoch: {}, Cell: {}, Batch: {}".format(epoch,cell,batch_idx))
                data = data.to(device)
                encoded, decoded, mu, log_var = model(data)
                # loss, recons_loss, KL_div  = model.loss_function(decoded, data, mu, log_var, beta)
                loss_list = model.loss_function(decoded, data, mu, log_var, args.beta)
                loss = loss_list[0]
                recons_loss = loss_list[1]
                KL_div = loss_list[2]
                L1_reg = loss_list[3]
                print("3 loss: ",loss, recons_loss, KL_div)
                optimizer.zero_grad()
                loss.backward()

                
                

                # ################################################
                optimizer.step()
                # print("data.size(0): ",data.size(0))
                total_loss += loss.item()*data.size(0)
                total_recons_loss += recons_loss.item()*data.size(0)
                total_KL_div += KL_div.item()*data.size(0)
                
                total_L1_reg += L1_reg*data.size(0)
                
                ############# Save model output ################

                data_size = int(data.size(0))
                data = data.cpu().detach().numpy()
                encoded = encoded.cpu().detach().numpy()
                decoded = decoded.cpu().detach().numpy()
                mu = mu.cpu().detach().numpy()
                log_var = log_var.cpu().detach().numpy()
                loss = loss.cpu().detach().numpy()
                recons_loss = recons_loss.cpu().detach().numpy()
                KL_div = KL_div.cpu().detach().numpy()

                cell_save_dict[batch_count] = {}
                cell_save_dict[batch_count]["cell"] = cell_print
                # json_data = json.dumps(cell_print)
                cell_save_dict[batch_count]["batch_idx"] = batch_idx
                # json_data = json.dumps(batch_idx)
                # cell_save_dict[batch_count]["data"] = data.tolist()
                # json_data = json.dumps(data.tolist())
                cell_save_dict[batch_count]["encoded"] = encoded.tolist()
                # json_data = json.dumps(encoded.tolist())
                cell_save_dict[batch_count]["decoded"] = decoded.tolist()
                # json_data = json.dumps(decoded.tolist())
                cell_save_dict[batch_count]["mu"] = mu.tolist()
                # json_data = json.dumps(mu.tolist())
                cell_save_dict[batch_count]["log_var"] = log_var.tolist()
                # json_data = json.dumps(log_var.tolist())
                cell_save_dict[batch_count]["loss"] = loss.tolist()
                # json_data = json.dumps(loss.tolist())
                cell_save_dict[batch_count]["recons_loss"] = recons_loss.tolist()
                # json_data = json.dumps(recons_loss.tolist())
                cell_save_dict[batch_count]["KL_div"] = KL_div.tolist()
                # json_data = json.dumps(KL_div.tolist())

                if reg == "L1":
                    cell_save_dict[batch_count]["L1_reg"] = L1_reg.cpu().detach().numpy().tolist()

                this_cell_all_loss.append([batch_count,cell,batch_idx,loss,recons_loss,KL_div,L1_reg,data_size])
                
                batch_count +=1
            
            num_data += len(train_loader.dataset)
            print("num data",num_data)

    
            # ============Save model output, loss for each batch in one file==========
            os.makedirs(os.path.join(save_path,"epoch_dump"), exist_ok=True)
            with open(os.path.join(save_path,"epoch_dump","epoch_{}_cell_{}_train.json".format(epoch,cell_print)), 'w') as fp:
                json.dump(cell_save_dict, fp)

            os.makedirs(os.path.join(save_path,"train_batch_loss"), exist_ok=True)
            with open(os.path.join(save_path,"train_batch_loss","epoch_{}_cell_{}_train_loss.txt".format(epoch,cell_print)), 'w') as fp:
                header = ["batch_count","cell","batch_idx","loss","recons_loss","KL_div","L1_reg","data_size"]
                fp.write('\t'.join(header) + '\n')
                for row in this_cell_all_loss:
                    fp.write('\t'.join(map(str, row)) + '\n')

        #=============Saving Epoch loss, model, print run time====================
        epoch_loss = total_loss / num_data
        epoch_recons_loss = total_recons_loss / num_data
        epoch_KL_div_loss = total_KL_div / num_data
        epoch_L1_reg = total_L1_reg / num_data
        
        loss_list.append([epoch,epoch_loss,epoch_recons_loss,epoch_KL_div_loss,epoch_L1_reg])

        print("saving model.pt")
        torch.save(model.state_dict(),os.path.join(save_path,"model_{}.pt".format(epoch)))

        end_time = time.time()
        epoch_duration = end_time - start_time
        hours = int(epoch_duration // 3600)
        minutes = int((epoch_duration % 3600) // 60)
        seconds = epoch_duration % 60
        print(f"Epoch {epoch+1}/{args.num_epochs} completed in {hours}h:{minutes}m:{seconds:.2f}s.")
        print(f"loss={epoch_loss}, recon_loss={epoch_recons_loss}, KL_div={epoch_recons_loss}")

        
        print("save epoch trainning loss info")
        os.makedirs(os.path.join(save_path,"train_epoch_loss"), exist_ok=True)
        loss_file = os.path.join(save_path,"train_epoch_loss","loss_{}.txt".format(epoch))
        header = "epoch,loss,recons_loss,KL_div"
        with open(loss_file, 'w') as file:
            file.write(header + '\n')  
            line = ','.join(map(str, [epoch,epoch_loss,epoch_recons_loss,epoch_KL_div_loss,epoch_L1_reg]))
            file.write(line + '\n')


        # ===================== Validate Model after epoch =================================
        if args.valid:
            valid_total_loss = 0.0
            valid_total_recons_loss = 0.0
            valid_total_KL_div = 0.0
            valid_total_L1_reg = 0.0

            valid_num_data = 0
            # debug_path = os.path.join(save_path,"debug")
            # os.makedirs(debug_path, exist_ok=True)
            valid_batch_count = 0
            valid_epoch_save_dict = {}
            valid_this_epoch_all_loss = []

            print("========= Epoch {} Validation ===========".format(epoch))
            for cell in cell_types:
                valid_loader_path = "dataloders/{}_valid_{}.pkl".format(cell_print,args.norm)
                valid_loader = torch.load(valid_loader_path)
                
                for batch_idx, data in enumerate(valid_loader):
                    print("Cell: {}, Batch: {}".format(epoch,cell,batch_idx))
                    data = data.to(device)
                    encoded, decoded, mu, log_var = model(data)
                    loss, recons_loss, KL_div,L1_reg = model.loss_function(decoded, data, mu, log_var, args.beta)

                    print("3 loss: ",loss, recons_loss, KL_div)

                    print("data.size(0): ",data.size(0))
                    
                    valid_total_loss += loss.item()*data.size(0)
                    valid_total_recons_loss += recons_loss.item()*data.size(0)
                    valid_total_KL_div += KL_div.item()*data.size(0)
                    valid_total_L1_reg += L1_reg.item()*data.size(0)
                    
                    ############# Save model output ################

                    data_size = int(data.size(0))
                    data = data.cpu().detach().numpy()
                    encoded = encoded.cpu().detach().numpy()
                    decoded = decoded.cpu().detach().numpy()
                    loss = loss.cpu().detach().numpy()
                    recons_loss = recons_loss.cpu().detach().numpy()
                    KL_div = KL_div.cpu().detach().numpy()
                    L1_reg = L1_reg.cpu().detach().numpy()

                    valid_epoch_save_dict[valid_batch_count] = {}
                    valid_epoch_save_dict[valid_batch_count]["cell"] = cell_print
                    json_data = json.dumps(cell_print)
                    valid_epoch_save_dict[valid_batch_count]["batch_idx"] = batch_idx
                    json_data = json.dumps(batch_idx)
                    # valid_epoch_save_dict[valid_batch_count]["data"] = data.tolist()
                    json_data = json.dumps(data.tolist())
                    valid_epoch_save_dict[valid_batch_count]["encoded"] = encoded.tolist()
                    json_data = json.dumps(encoded.tolist())
                    valid_epoch_save_dict[valid_batch_count]["decoded"] = decoded.tolist()
                    json_data = json.dumps(decoded.tolist())
                    valid_epoch_save_dict[valid_batch_count]["mu"] = mu.tolist()
                    json_data = json.dumps(mu.tolist())
                    valid_epoch_save_dict[valid_batch_count]["log_var"] = log_var.tolist()
                    json_data = json.dumps(log_var.tolist())
                    valid_epoch_save_dict[valid_batch_count]["loss"] = loss.tolist()
                    json_data = json.dumps(loss.tolist())
                    valid_epoch_save_dict[valid_batch_count]["recons_loss"] = recons_loss.tolist()
                    # json_data = json.dumps(recons_loss.tolist())
                    valid_epoch_save_dict[valid_batch_count]["KL_div"] = KL_div.tolist()
                    # json_data = json.dumps(KL_div.tolist())
                    valid_epoch_save_dict[valid_batch_count]["L1_reg"] = L1_reg.tolist()

                    

                
                    valid_this_epoch_all_loss.append([valid_batch_count,cell,batch_idx,loss,recons_loss,KL_div,L1_reg,data_size])
                    valid_batch_count +=1
                
                valid_num_data+=len(train_loader.dataset)

            valid_epoch_loss = valid_total_loss / valid_num_data
            valid_epoch_recons_loss = valid_total_recons_loss / valid_num_data
            valid_epoch_KL_div_loss = valid_total_KL_div / valid_num_data
            valid_epoch_L1_reg = valid_total_L1_reg / valid_num_data
            valid_loss_list.append([epoch,valid_epoch_loss,valid_epoch_recons_loss,valid_epoch_KL_div_loss,valid_epoch_L1_reg])

            #============= Saving Validation loss ====================
            os.makedirs(os.path.join(save_path,"epoch_dump"), exist_ok=True)
            with open(os.path.join(save_path,"epoch_dump","epoch_{}_valid.json".format(epoch)), 'w') as fp:
                json.dump(valid_epoch_save_dict, fp)

            os.makedirs(os.path.join(save_path,"valid_batch_loss"), exist_ok=True)
            with open(os.path.join(save_path,"valid_batch_loss","epoch_{}_valid_loss.txt".format(epoch)), 'w') as fp:
                header = ["batch_count","cell","batch_idx","loss","recons_loss","KL_div","L1_reg","data_size"]
                fp.write('\t'.join(header) + '\n')
                for row in valid_this_epoch_all_loss:
                    fp.write('\t'.join(map(str, row)) + '\n')
                
            header = "epoch,loss,recons_loss,KL_div"
            os.makedirs(os.path.join(save_path,"valid_epoch_loss"), exist_ok=True)
            loss_file = os.path.join(save_path,"valid_epoch_loss","loss_{}.txt".format(epoch))
            print("save epoch loss info")
            with open(loss_file, 'w') as file:
                file.write(header + '\n')  
                line = ','.join(map(str, [epoch,valid_epoch_loss,valid_epoch_recons_loss,valid_epoch_KL_div_loss,valid_epoch_L1_reg]))
                file.write(line + '\n')

    return model,loss_list

