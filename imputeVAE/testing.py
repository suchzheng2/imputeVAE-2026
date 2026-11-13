import os
import json
import numpy as np
import torch
import anndata as ad

from .utils import device, setup_device, load_args_from_json
from .models import VAE_L1, VAE_JSD, VAE_NB
from .data import create_dataloaders, create_dataloaders_by_cell_type, MyConcatDataset
from .visualization import (plot_umap_after_train_by_celltype, plot_umap_after_test,
                            plot_umap_after_test_mixed_modality, plot_umap_for_both_modaility)

def test_model_on_whole_dataset(save_path, args):
    #load the model
    model_path = os.path.join(args.load_trained_model_path)
    # load the model arguments
    model_args_path = os.path.join(args.load_trained_model_path, 'arguments.json')
    model_args = load_args_from_json(model_args_path)
    model = VAE_JSD(
        input_size=model_args["num_feature"],
        num_internal=model_args["num_internal"],
        num_latent=model_args["num_latent"]
    )   
    if args.load_trained_model: 
        trained_model_path = os.path.join(args.load_trained_model_path, "models", "model.pt")
        model.load_state_dict(torch.load(trained_model_path, map_location=device))
        model = model.to(device)  # Ensure model is on the correct device
        print(f"Loaded trained model from {args.load_trained_model}")
    #load the data
    if args.use_raw_data:
        adata = load_raw_data("10X_nuclei_v2_v3_AIBS")
    else:
        integrated_dp = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ABC_results"),\
            args.data_path,\
            "integrated.h5ad")
        adata = ad.read_h5ad(integrated_dp)
    # show adata info
    print(adata.info())
    train_loaders, valid_loaders, test_loaders = create_dataloaders_by_cell_type(adata,args)

    cell_type_save_dict = {}
    for cell_type, train_loader in train_loaders.items():
        print("testing on train data of cell type: ", cell_type)
        cell_type_save_dict[cell_type] = {} 
        for batch_idx, batch in enumerate(train_loader):
            data = batch['data'].to(device)
            encoded, decoded, mu, log_var = model(data)
            loss, recons_loss, js_div = model.loss_function(decoded, data, mu, log_var, args)
            cell_type_save_dict[cell_type][batch_idx] = {}
            cell_type_save_dict[cell_type][batch_idx]["batch_idx"] = batch_idx
            cell_type_save_dict[cell_type][batch_idx]["cell_type"] = batch["cell_type_original"].cpu().detach().numpy().tolist()
            cell_type_save_dict[cell_type][batch_idx]["meta_batch_id"] = batch["meta_batch_id_original"].cpu().detach().numpy().tolist()
            cell_type_save_dict[cell_type][batch_idx]["encoded"] = encoded.cpu().detach().numpy().tolist()
            cell_type_save_dict[cell_type][batch_idx]["decoded"] = decoded.cpu().detach().numpy().tolist()
            cell_type_save_dict[cell_type][batch_idx]["mu"] = mu.cpu().detach().numpy().tolist()
            cell_type_save_dict[cell_type][batch_idx]["log_var"] = log_var.cpu().detach().numpy().tolist()
            cell_type_save_dict[cell_type][batch_idx]["loss"] = loss.cpu().detach().numpy().tolist()
            cell_type_save_dict[cell_type][batch_idx]["recons_loss"] = recons_loss.cpu().detach().numpy().tolist()
            cell_type_save_dict[cell_type][batch_idx]["JSD_div"] = js_div.cpu().detach().numpy().tolist()

    with open(os.path.join(save_path,"test_on_whole_dataset_results.json"), 'w') as fp:
        json.dump(cell_type_save_dict, fp)

    #make umap plot
    plot_umap_after_train_by_celltype(save_path,feature="encoded",data_split= "train", epoch=0,file_path=os.path.join(save_path,"test_on_whole_dataset_results.json"))
    plot_umap_after_train_by_celltype(save_path,feature="decoded",data_split= "train", epoch=0,file_path=os.path.join(save_path,"test_on_whole_dataset_results.json"))
    return


def test_nuclei_model_with_cell_data(save_path, args):
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

    print(f"Device at start of function: {device}")
    print(f"testing nuclei model with cell data")
    # Load the JSON file
    json_path = os.path.join(args.test_model_run, 'arguments.json')
    with open(json_path, 'r') as f:
        args_dict = json.load(f)
    nuclei_model = VAE_JSD(
        input_size=args_dict["num_feature"],
        num_internal=args_dict["num_internal"],
        num_latent=args_dict["num_latent"]
    )
    last_eppch = len(os.listdir(os.path.join(args.test_model_run, "models"))) - 1
    model_path = os.path.join(args.test_model_run, "models", "model_{}.pt".format(last_eppch))
    nuclei_model.load_state_dict(torch.load(model_path))
    nuclei_model.to(device)
    nuclei_model.eval()

    integrated_dp = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ABC_results"),\
        args.data_path,\
        "integrated.h5ad")
    adata = ad.read_h5ad(integrated_dp)
    train_loaders, valid_loaders, test_loaders = create_dataloaders_by_cell_type(adata,args)

    train_save_dict = {}

    with torch.no_grad():
        for cell_type, train_loader in train_loaders.items():
            print("testing on train data of cell type: ", cell_type)
            train_save_dict[cell_type] = {}
            for batch_idx, batch in enumerate(train_loader):
                data = batch['data'].to(device)
                encoded, decoded, mu, log_var = nuclei_model(data)
                loss, recons_loss, js_div = nuclei_model.loss_function(decoded, data, mu, log_var, args)

                encoded_np = encoded.cpu().numpy()
                decoded_np = decoded.cpu().numpy()
                mu_np = mu.cpu().numpy()
                log_var_np = log_var.cpu().numpy()
                loss_np = loss.cpu().numpy()
                recons_loss_np = recons_loss.cpu().numpy()
                js_div_np = js_div.cpu().numpy()
                print("loss: ",loss, "recons_loss: ",recons_loss, "js_div: ",js_div)

                celltypes = batch["cell_type"].cpu().detach().numpy()
                meta_batch_id = batch["batch_id"].cpu().detach().numpy()

                original_cell_types = []
                for cell_code in batch["cell_type"].cpu().numpy():
                    original_cell_types.append(train_loader.dataset.get_original_value('cell_type', cell_code))

                original_batch_ids = []
                if train_loader.dataset.has_batch_id:
                    for batch_code in batch["batch_id"].cpu().numpy():
                        original_batch_ids.append(train_loader.dataset.get_original_value('batch_id', batch_code))

                train_save_dict[cell_type][batch_idx] = {} 
                train_save_dict[cell_type][batch_idx]["batch_idx"] = batch_idx
                train_save_dict[cell_type][batch_idx]["cell_type"] = celltypes.tolist()
                train_save_dict[cell_type][batch_idx]["cell_type_original"] = original_cell_types
                train_save_dict[cell_type][batch_idx]["meta_batch_id"] = meta_batch_id.tolist()
                train_save_dict[cell_type][batch_idx]["meta_batch_id_original"] = original_batch_ids
                train_save_dict[cell_type][batch_idx]["encoded"] = encoded_np.tolist()
                train_save_dict[cell_type][batch_idx]["decoded"] = decoded_np.tolist()
                train_save_dict[cell_type][batch_idx]["mu"] = mu_np.tolist()
                train_save_dict[cell_type][batch_idx]["log_var"] = log_var_np.tolist()
                train_save_dict[cell_type][batch_idx]["loss"] = loss_np.tolist()
                train_save_dict[cell_type][batch_idx]["recons_loss"] = recons_loss_np.tolist()
                train_save_dict[cell_type][batch_idx]["JSD_div"] = js_div_np.tolist()

        with open(os.path.join(save_path,"test_on_cell_trainning_data_results.json"), 'w') as fp:
            json.dump(train_save_dict, fp)

        #make umap plot
        print("save umap plot")
        plot_umap_after_train_by_celltype(save_path,feature="encoded",data_split= "train", epoch=0,file_path=os.path.join(save_path,"test_on_cell_trainning_data_results.json"))
        plot_umap_after_train_by_celltype(save_path,feature="decoded",data_split= "train", epoch=0,file_path=os.path.join(save_path,"test_on_cell_trainning_data_results.json"))
        
        #nuclei file is the last training epoch save dict
        #list file with format epoch_{}_train.json
        epoch_dump_path = os.path.join(args.test_model_run,"epoch_dump")
        print(epoch_dump_path)
        train_files =[f for f in os.listdir(os.path.join(args.test_model_run,"epoch_dump")) if f.endswith("train.json")]
        print(train_files)
        #get the int number in the file name
        train_files_num = [int(f.split("_")[1]) for f in train_files]
        print(train_files_num)
        #get the max number
        epoch = max(train_files_num)
        print(epoch)
        nuclei_file = os.path.join(args.test_model_run,"epoch_dump","epoch_{}_train.json".format(epoch))

        cell_file = os.path.join(save_path,"test_on_cell_trainning_data_results.json")
        print("nuclei_file: ",nuclei_file)
        print("cell_file: ",cell_file)
        
        plot_umap_for_both_modaility(save_path,nuclei_file,cell_file,feature="encoded")
        plot_umap_for_both_modaility(save_path,nuclei_file,cell_file,feature="decoded")
    return nuclei_model, train_save_dict


def test_model_ABC(save_path, args):
    
    #load model arguments
    model_args_path = os.path.join(args.test_model_run, 'arguments.json')
    model_run_arguments = load_args_from_json(model_args_path)

    # arg_list = []
    # for key, value in saved_args_dict.items():
    #     cli_key = f'--{key.replace("_", "-")}'  # e.g., learning_rate -> --learning-rate
    #     arg_list.extend([cli_key, str(value)])

    # args = parser.parse_args(arg_list)

    if model_run_arguments["model"] == "vae_js":
            model = VAE_JSD(input_size = model_run_arguments["num_feature"], \
             num_internal = model_run_arguments["num_internal"], \
            num_latent = model_run_arguments["num_latent"])
    else:
        model = VAE_L1(input_size = model_run_arguments["num_feature"], \
            num_internal = model_run_arguments["num_internal"], \
            num_latent = model_run_arguments["num_latent"],\
            l1_coef = model_run_arguments["l1_coef"])

    # model = VAE_L1(input_size = model_run_arguments["num_feature"], \
    #         num_internal = model_run_arguments["num_internal"], \
    #         num_latent = model_run_arguments["num_latent"],\
    #         l1_coef = model_run_arguments["l1_coef"])


    model_dir = os.path.join(args.test_model_run,"models")
    model_files = sorted([f for f in os.listdir(model_dir) if f.startswith('model_') and f.endswith('.pt')])
    latest_model_path = os.path.join(model_dir, model_files[-1])
    model.load_state_dict(torch.load(latest_model_path))
    model.to(device)
    model.eval()


    # Load test data
    # test_data_loader_path = args.test_data
    # if os.path.exists(test_data_loader_path):
    #     test_loader = torch.load(test_data_loader_path)
    # else:
    #     raise FileNotFoundError(f"Test data loader not found at: {test_data_loader_path}")

    integrated_dp = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ABC_results"),\
        args.data_path,\
        "integrated.h5ad")
    adata = ad.read_h5ad(integrated_dp)
    _, _, test_loader = create_dataloaders(adata,args)

    total_loss, total_recons_loss, total_JSD_div, total_L1_reg = 0.0, 0.0, 0.0, 0.0
    num_data = 0
    test_save_dict = {}
        # Testing loop
    with torch.no_grad():

        for batch_idx, batch in enumerate(test_loader):
            # print("Epoch: {}, Batch: {}".format(epoch,batch_idx))
            data = batch['data']
            # obs = batch['obs']
            # var = batch['var']
            print("data shape:")
            print(data.shape)
            data = data.to(device)
            encoded, decoded, mu, log_var = model(data)
            # print(encoded, decoded, mu, log_var)
            # loss, recons_loss, KL_div  = model.loss_function(decoded, data, mu, log_var, beta)
            all_loss = model.loss_function(decoded, data, mu, log_var, args)
            print("all loss:",all_loss)
            loss = all_loss[0]
            recons_loss = all_loss[1]
            KL_div = all_loss[2]
            # L1_reg = all_loss[3]
            print("3 loss: ",loss, recons_loss, KL_div)
           
            # print("data.size(0): ",data.size(0))
            total_loss += loss.item()*data.size(0)
            total_recons_loss += recons_loss.item()*data.size(0)
            total_JSD_div += JSD_div.item()*data.size(0)
            print("total_JSD_div:",total_JSD_div)
            # total_L1_reg += L1_reg.item()*data.size(0)

            
            # cell_save_dict[batch_count]["cell"] = cell_print
            data_size = int(data.size(0))


            # data = data.cpu().detach().numpy()
            encoded = encoded.cpu().detach().numpy()
            decoded = decoded.cpu().detach().numpy()
            mu = mu.cpu().detach().numpy()
            log_var = log_var.cpu().detach().numpy()
            loss = loss.cpu().detach().numpy()
            recons_loss = recons_loss.cpu().detach().numpy()
            JSD_div = JSD_div.cpu().detach().numpy()
            # L1_reg = L1_reg.cpu().detach().numpy()

            num_data += data.size(0)

            print(f"Batch {batch_idx}: Loss = {loss.item()}, Recon Loss = {recons_loss.item()}, KL Div = {KL_div.item()}")

            celltypes = batch["cell_type"].cpu().detach().numpy()
            meta_batch_id = batch["batch_id"].cpu().detach().numpy()

            original_cell_types = []
            for cell_code in batch["cell_type"].cpu().numpy():
                original_cell_types.append(test_loader.dataset.get_original_value('cell_type', cell_code))

            # Get the original batch IDs if they exist
            original_batch_ids = []
            if test_loader.dataset.has_batch_id:
                for batch_code in batch["batch_id"].cpu().numpy():
                    original_batch_ids.append(test_loader.dataset.get_original_value('batch_id', batch_code))
                
            test_save_dict[batch_idx] = {}
            test_save_dict[batch_idx]["batch_idx"] = batch_idx
            test_save_dict[batch_idx]["cell_type"] = celltypes.tolist() 
            test_save_dict[batch_idx]["cell_type_original"] = original_cell_types
            test_save_dict[batch_idx]["meta_batch_id"] = meta_batch_id.tolist() 
            test_save_dict[batch_idx]["meta_batch_id_original"] = original_batch_ids
            test_save_dict[batch_idx]["encoded"] = encoded.tolist()    
            test_save_dict[batch_idx]["decoded"] = decoded.tolist()        
            test_save_dict[batch_idx]["mu"] = mu.tolist()
            test_save_dict[batch_idx]["log_var"] = log_var.tolist()
            test_save_dict[batch_idx]["loss"] = loss.tolist()
            test_save_dict[batch_idx]["recons_loss"] = recons_loss.tolist()
            test_save_dict[batch_idx]["JSD_div"] = JSD_div.tolist()
            # test_save_dict[batch_idx]["L1_reg"] = L1_reg.tolist()

            

            # exit()
    with open(os.path.join(save_path,"test_results.json"), 'w') as fp:
        json.dump(test_save_dict, fp)
    
    avg_loss = total_loss / num_data
    avg_recons_loss = total_recons_loss / num_data
    avg_JSD_div = total_JSD_div / num_data
    # avg_L1_reg = total_L1_reg / num_data
    print("avg_JSD_div:",avg_JSD_div)

    print("Testing Complete")
    print(f"Avg Loss: {avg_loss}, Avg Recon Loss: {avg_recons_loss}, Avg JSD Div: {avg_JSD_div}")

    loss_file = os.path.join(save_path,"avg_loss.txt")
    print("save loss info")
    line_header = ["loss","recons_loss","JSD_div"]
    with open(loss_file, 'w') as file:
        # file.write(header + '\n')
        line_header_print = '\t'.join(map(str, line_header))  
        file.write(line_header_print + '\n')

        line_print = '\t'.join(map(str, [avg_loss, avg_recons_loss, avg_JSD_div]))  
        file.write(line_print + '\n')

    
        print("------------------------ Model Evaluation =====================")
        # os.makedirs(fig_save_dir,exist_ok=True)
        plot_umap_after_test(save_path,feature="encoded",data_split= "test")
        plot_umap_after_test(save_path,feature="decoded",data_split= "test")


    return avg_loss, avg_recons_loss, avg_JSD_div
    
def test_model_ABC_combined(save_path, args):
    # Load both sets of model arguments
    nuclei_args_path = os.path.join(args.nuclei_model_run, 'arguments.json')
    cell_args_path = os.path.join(args.cell_model_run, 'arguments.json')
    nuclei_args = load_args_from_json(nuclei_args_path)
    cell_args = load_args_from_json(cell_args_path)

    # Load both models
    def load_vae_jsd(model_args, model_run_dir):
        model = VAE_JSD(
            input_size=model_args["num_feature"],
            num_internal=model_args["num_internal"],
            num_latent=model_args["num_latent"]
        )
        model_path = sorted([
            f for f in os.listdir(os.path.join(model_run_dir, "models"))
            if f.startswith("model_") and f.endswith(".pt")
        ])[-1]
        full_model_path = os.path.join(model_run_dir, "models", model_path)
        model.load_state_dict(torch.load(full_model_path, map_location=device))
        model.to(device)
        model.eval()
        return model

    nuclei_model = load_vae_jsd(nuclei_args, args.nuclei_model_run)
    cell_model = load_vae_jsd(cell_args, args.cell_model_run)

    # Build combined model
    combined_model = VAE_JSD(
        input_size=nuclei_args["num_feature"],
        num_internal=nuclei_args["num_internal"],
        num_latent=nuclei_args["num_latent"]
    )
    combined_model.encoder = nuclei_model.encoder
    combined_model.mu = nuclei_model.mu
    combined_model.log_var = nuclei_model.log_var
    combined_model.decoder = cell_model.decoder
    combined_model.to(device)
    combined_model.eval()

    # Load test dataset from nuclei
    integrated_dp = os.path.join(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ABC_results"),
        args.data_path,
        "integrated.h5ad"
    )
    adata = ad.read_h5ad(integrated_dp)
    _, _, test_loader = create_dataloaders(adata, args)

    total_loss, total_recons_loss, total_JS_div = 0.0, 0.0, 0.0
    num_data = 0
    test_save_dict = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            data = batch['data'].to(device)
            encoded, decoded, mu, log_var = combined_model(data)
            loss, recons_loss, js_div = combined_model.loss_function(decoded, data, mu, log_var, args)

            total_loss += loss.item() * data.size(0)
            total_recons_loss += recons_loss.item() * data.size(0)
            total_JS_div += js_div.item() * data.size(0)
            num_data += data.size(0)

            encoded_np = encoded.cpu().numpy()
            decoded_np = decoded.cpu().numpy()
            mu_np = mu.cpu().numpy()
            log_var_np = log_var.cpu().numpy()

            original_cell_types = [
                test_loader.dataset.get_original_value('cell_type', x.item())
                for x in batch["cell_type"]
            ]
            original_batch_ids = [
                test_loader.dataset.get_original_value('batch_id', x.item())
                for x in batch["batch_id"]
            ]

            test_save_dict[batch_idx] = {
                "batch_idx": batch_idx,
                "cell_type": batch["cell_type"].cpu().tolist(),
                "cell_type_original": original_cell_types,
                "meta_batch_id": batch["batch_id"].cpu().tolist(),
                "meta_batch_id_original": original_batch_ids,
                "encoded": encoded_np.tolist(),
                "decoded": decoded_np.tolist(),
                "mu": mu_np.tolist(),
                "log_var": log_var_np.tolist(),
                "loss": loss.item(),
                "recons_loss": recons_loss.item(),
                "JS_div": js_div.item()
            }

            print(f"Batch {batch_idx}: Loss = {loss.item()}, Recon Loss = {recons_loss.item()}, JS Div = {js_div.item()}")

    # Save test results
    with open(os.path.join(save_path, "test_results.json"), 'w') as f:
        json.dump(test_save_dict, f)

    # Report average loss
    avg_loss = total_loss / num_data
    avg_recons_loss = total_recons_loss / num_data
    avg_JS_div = total_JS_div / num_data

    print("Testing Complete")
    print(f"Avg Loss: {avg_loss}, Avg Recon Loss: {avg_recons_loss}, Avg JS Div: {avg_JS_div}")

    with open(os.path.join(save_path, "avg_loss_combined.txt"), 'w') as f:
        f.write("loss\trecons_loss\tJS_div\n")
        f.write(f"{avg_loss}\t{avg_recons_loss}\t{avg_JS_div}\n")

    # Plot UMAP visualizations
    plot_umap_after_test(save_path, feature="encoded", data_split="test")
    plot_umap_after_test(save_path, feature="decoded", data_split="test")

    return avg_loss, avg_recons_loss, avg_JS_div

def load_mixed_dataset(args):
    integrated_dp = os.path.join(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ABC_results"),
        "10X_nuclei_v2_v3_AIBS_g10k",
        "integrated.h5ad"
    )
    adata = ad.read_h5ad(integrated_dp)
    args.batch= "n2A"
    _, _, test_loader_1 = create_dataloaders(adata, args)

    integrated_dp = os.path.join(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ABC_results"),
        "10X_cells_v2_v3_AIBS_g10k",
        "integrated.h5ad"
    )
    adata = ad.read_h5ad(integrated_dp)
    args.batch= "c2A"
    _, _, test_loader_2 = create_dataloaders(adata, args)

    dataset1 = test_loader_1.dataset
    dataset2 = test_loader_2.dataset

    combined_dataset = MyConcatDataset([dataset1, dataset2])

    combined_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)
    return combined_loader

def test_model_ABC_combined_mixed_modality(save_path, args):

    print("test_model_ABC_combined_mixed_modality")
    # Load both sets of model arguments
    # Load model args
    def load_args_from_json(path):
        with open(path, 'r') as f:
            return json.load(f)

    nuclei_args = load_args_from_json(os.path.join(args.nuclei_model_run, 'arguments.json'))
    cell_args = load_args_from_json(os.path.join(args.cell_model_run, 'arguments.json'))


    # Load both models
    def load_vae_jsd(model_args, model_run_dir):
        model = VAE_JSD(
            input_size=model_args["num_feature"],
            num_internal=model_args["num_internal"],
            num_latent=model_args["num_latent"]
        )
        model_path = sorted([
            f for f in os.listdir(os.path.join(model_run_dir, "models"))
            if f.startswith("model_") and f.endswith(".pt")
        ])[-1]
        full_model_path = os.path.join(model_run_dir, "models", model_path)
        model.load_state_dict(torch.load(full_model_path, map_location=device))
        model.to(device)
        model.eval()
        return model

    nuclei_model = load_vae_jsd(nuclei_args, args.nuclei_model_run)
    cell_model = load_vae_jsd(cell_args, args.cell_model_run)

    # Build combined model
    combined_model = VAE_JSD(
        input_size=nuclei_args["num_feature"],
        num_internal=nuclei_args["num_internal"],
        num_latent=nuclei_args["num_latent"]
    )
    combined_model.encoder = nuclei_model.encoder
    combined_model.mu = nuclei_model.mu
    combined_model.log_var = nuclei_model.log_var
    combined_model.decoder = cell_model.decoder
    combined_model.to(device)
    combined_model.eval()

    # test_loader = load_mixed_dataset(args)

    total_loss, total_recons_loss, total_JS_div = 0.0, 0.0, 0.0
    num_data = 0
    test_save_dict = {}

    batch_idx_count = 0

    #test on nuclei data
    print("================Test on nuclei data===================")
    integrated_dp = os.path.join(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ABC_results"),
        "10X_nuclei_v2_v3_AIBS_g10k",
        "integrated.h5ad"
    )
    adata = ad.read_h5ad(integrated_dp)
    args.batch= "n2A"
    _, _, test_loader = create_dataloaders(adata, args)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            data = batch['data'].to(device)
            encoded, decoded, mu, log_var = combined_model(data)
            loss, recons_loss, js_div = combined_model.loss_function(decoded, data, mu, log_var, args)

            total_loss += loss.item() * data.size(0)
            total_recons_loss += recons_loss.item() * data.size(0)
            total_JS_div += js_div.item() * data.size(0)
            num_data += data.size(0)

            encoded_np = encoded.cpu().numpy()
            decoded_np = decoded.cpu().numpy()
            mu_np = mu.cpu().numpy()
            log_var_np = log_var.cpu().numpy()

            original_cell_types = [
                test_loader.dataset.get_original_value('cell_type', x.item())
                for x in batch["cell_type"]
            ]
            original_batch_ids = [
                test_loader.dataset.get_original_value('batch_id', x.item())
                for x in batch["batch_id"]
            ]

            test_save_dict[batch_idx_count] = {
                "batch_idx_overall":batch_idx_count,
                "modality:":"nuclei",
                "batch_idx": batch_idx,
                "cell_type": batch["cell_type"].cpu().tolist(),
                "cell_type_original": original_cell_types,
                "meta_batch_id": batch["batch_id"].cpu().tolist(),
                "meta_batch_id_original": original_batch_ids,
                "encoded": encoded_np.tolist(),
                "decoded": decoded_np.tolist(),
                "mu": mu_np.tolist(),
                "log_var": log_var_np.tolist(),
                "loss": loss.item(),
                "recons_loss": recons_loss.item(),
                "JS_div": js_div.item()
            }

            batch_idx_count +=1

            print(f"Batch {batch_idx}: Loss = {loss.item()}, Recon Loss = {recons_loss.item()}, JS Div = {js_div.item()}")
    
    
    #test on cell data
    print("================Test on cell data===================")
    integrated_dp = os.path.join(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ABC_results"),
        "10X_cells_v2_v3_AIBS_g10k",
        "integrated.h5ad"
    )
    adata = ad.read_h5ad(integrated_dp)
    args.batch= "c2A"
    _, _, test_loader = create_dataloaders(adata, args)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            data = batch['data'].to(device)
            encoded, decoded, mu, log_var = combined_model(data)
            loss, recons_loss, js_div = combined_model.loss_function(decoded, data, mu, log_var, args)

            total_loss += loss.item() * data.size(0)
            total_recons_loss += recons_loss.item() * data.size(0)
            total_JS_div += js_div.item() * data.size(0)
            num_data += data.size(0)

            encoded_np = encoded.cpu().numpy()
            decoded_np = decoded.cpu().numpy()
            mu_np = mu.cpu().numpy()
            log_var_np = log_var.cpu().numpy()

            original_cell_types = [
                test_loader.dataset.get_original_value('cell_type', x.item())
                for x in batch["cell_type"]
            ]
            original_batch_ids = [
                test_loader.dataset.get_original_value('batch_id', x.item())
                for x in batch["batch_id"]
            ]

            test_save_dict[batch_idx_count] = {
                "batch_idx_overall":batch_idx_count,
                "modality:":"cell",
                "batch_idx": batch_idx,
                "cell_type": batch["cell_type"].cpu().tolist(),
                "cell_type_original": original_cell_types,
                "meta_batch_id": batch["batch_id"].cpu().tolist(),
                "meta_batch_id_original": original_batch_ids,
                "encoded": encoded_np.tolist(),
                "decoded": decoded_np.tolist(),
                "mu": mu_np.tolist(),
                "log_var": log_var_np.tolist(),
                "loss": loss.item(),
                "recons_loss": recons_loss.item(),
                "JS_div": js_div.item()
            }

            batch_idx_count +=1

            print(f"Batch {batch_idx}: Loss = {loss.item()}, Recon Loss = {recons_loss.item()}, JS Div = {js_div.item()}")
    
    # Save test results
    with open(os.path.join(save_path, "test_results.json"), 'w') as f:
        json.dump(test_save_dict, f)

    # Report average loss
    avg_loss = total_loss / num_data
    avg_recons_loss = total_recons_loss / num_data
    avg_JS_div = total_JS_div / num_data

    print("Testing Complete")
    print(f"Avg Loss: {avg_loss}, Avg Recon Loss: {avg_recons_loss}, Avg JS Div: {avg_JS_div}")

    with open(os.path.join(save_path, "avg_loss_combined.txt"), 'w') as f:
        f.write("loss\trecons_loss\tJS_div\n")
        f.write(f"{avg_loss}\t{avg_recons_loss}\t{avg_JS_div}\n")

    # Plot UMAP visualizations
    plot_umap_after_test_mixed_modality(save_path, feature="encoded", data_split="test")
    plot_umap_after_test_mixed_modality(save_path, feature="decoded", data_split="test")

    return avg_loss, avg_recons_loss, avg_JS_div


def evaluate_nuclei_model(model, data_loader, cell_type, save_path, epoch, current_cell_mu, current_cell_log_var, args):
    model.eval()
    total_loss, total_recons_loss, total_JS_div_nuclei, total_JS_div_cell_nuclei, total_KL_div = 0.0, 0.0, 0.0, 0.0, 0.0
    num_data = 0
    valid_epoch_save_dict = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            data = batch['data'].to(device)
            encoded, decoded, mu, log_var = model(data)
            #get current batch cell mu and log_var by batch size    
            start_idx = batch_idx * args.batch_size
            end_idx = start_idx + data.size(0)
            if args.use_all_cell_mu_log_var:
                cell_mu_batch = torch.tensor(current_cell_mu, dtype=torch.float32).to(device)
                cell_log_var_batch = torch.tensor(current_cell_log_var, dtype=torch.float32).to(device)
            else:
                cell_mu_batch = torch.tensor(current_cell_mu[start_idx:end_idx], dtype=torch.float32).to(device)
                cell_log_var_batch = torch.tensor(current_cell_log_var[start_idx:end_idx], dtype=torch.float32).to(device)
            all_loss = model.fine_tune_loss_function(decoded, data, mu, log_var, cell_mu_batch, cell_log_var_batch, args)
            loss, recons_loss, JS_div_nuclei, JS_div_cell_nuclei = all_loss[0], all_loss[1], all_loss[2], all_loss[3]
            
            total_loss += loss.item() * data.size(0)
            total_recons_loss += recons_loss.item() * data.size(0)
            total_JS_div_nuclei += JS_div_nuclei.item() * data.size(0)
            total_JS_div_cell_nuclei += JS_div_cell_nuclei.item() * data.size(0)
            num_data += data.size(0)

            encoded_np = encoded.cpu().numpy()
            decoded_np = decoded.cpu().numpy()
            mu_np = mu.cpu().numpy()
            log_var_np = log_var.cpu().numpy()

            celltypes = batch["cell_type"].cpu().detach().numpy()
            meta_batch_id = batch["batch_id"].cpu().detach().numpy()

            original_cell_types = []
            for cell_code in batch["cell_type"].cpu().numpy():
                original_cell_types.append(data_loader.dataset.get_original_value('cell_type', cell_code))

            original_batch_ids = []
            if data_loader.dataset.has_batch_id:
                for batch_code in batch["batch_id"].cpu().numpy():
                    original_batch_ids.append(data_loader.dataset.get_original_value('batch_id', batch_code))
            
            batch_results = {}
            batch_results["loss"] = loss.item()
            batch_results["recons_loss"] = recons_loss.item()
            batch_results["JS_div_nuclei"] = JS_div_nuclei.item()
            batch_results["JS_div_cell_nuclei"] = JS_div_cell_nuclei.item()
            batch_results["encoded"] = encoded_np.tolist()
            batch_results["decoded"] = decoded_np.tolist()
            batch_results["mu"] = mu_np.tolist()
            batch_results["log_var"] = log_var_np.tolist()
            batch_results["cell_type"] = celltypes.tolist()
            batch_results["meta_batch_id"] = meta_batch_id.tolist()
            batch_results["cell_type_original"] = original_cell_types
            batch_results["meta_batch_id_original"] = original_batch_ids

            valid_epoch_save_dict[batch_idx] = batch_results

            if args.debug and batch_idx == 1:
                break

    
    model.train()

    return valid_epoch_save_dict, total_loss, total_recons_loss, total_JS_div_nuclei, total_JS_div_cell_nuclei

def combine_model(nuclei_model, cell_model, nuclei_args):
    combined_model = VAE_JSD(
        input_size=nuclei_args["num_feature"],
        num_internal=nuclei_args["num_internal"],
        num_latent=nuclei_args["num_latent"]
    )
    combined_model.encoder = nuclei_model.encoder
    combined_model.mu = nuclei_model.mu
    combined_model.log_var = nuclei_model.log_var
    combined_model.decoder = cell_model.decoder
    combined_model.to(device)
    combined_model.eval()
    return combined_model

def evaluate_combined_model(model, data_loader, cell_type, save_path, epoch, args, current_cell_mu, current_cell_log_var):
    model.eval()
    total_loss, total_recons_loss, total_JS_div_nuclei, total_JS_div_cell_nuclei = 0.0, 0.0, 0.0, 0.0
    num_data = 0
    valid_epoch_save_dict = {} 

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            data = batch['data'].to(device) 
            encoded, decoded, mu, log_var = model(data)
            # get next batch of cell mu and log_var from cell model for this cell type by batch size
            start_idx = batch_idx * args.batch_size
            end_idx = start_idx + data.size(0)
            if args.use_all_cell_mu_log_var:
                cell_mu_batch = torch.tensor(current_cell_mu, dtype=torch.float32).to(device)
                cell_log_var_batch = torch.tensor(current_cell_log_var, dtype=torch.float32).to(device)
            else:
                cell_mu_batch = torch.tensor(current_cell_mu[start_idx:end_idx], dtype=torch.float32).to(device)
                cell_log_var_batch = torch.tensor(current_cell_log_var[start_idx:end_idx], dtype=torch.float32).to(device)

            all_loss = model.fine_tune_loss_function(decoded, data, mu, log_var, cell_mu_batch, cell_log_var_batch, args)
            loss, recons_loss, JS_div_nuclei, JS_div_cell_nuclei = all_loss[0], all_loss[1], all_loss[2], all_loss[3]

            total_loss += loss.item() * data.size(0)
            total_recons_loss += recons_loss.item() * data.size(0)
            total_JS_div_nuclei += JS_div_nuclei.item() * data.size(0)
            total_JS_div_cell_nuclei += JS_div_cell_nuclei.item() * data.size(0)
            num_data += data.size(0)

            encoded_np = encoded.cpu().numpy()
            decoded_np = decoded.cpu().numpy()
            mu_np = mu.cpu().numpy()
            log_var_np = log_var.cpu().numpy()  

            celltypes = batch["cell_type"].cpu().detach().numpy()
            meta_batch_id = batch["batch_id"].cpu().detach().numpy()

            original_cell_types = []
            for cell_code in batch["cell_type"].cpu().numpy():
                original_cell_types.append(data_loader.dataset.get_original_value('cell_type', cell_code))  

            original_batch_ids = []
            if data_loader.dataset.has_batch_id:
                for batch_code in batch["batch_id"].cpu().numpy():
                    original_batch_ids.append(data_loader.dataset.get_original_value('batch_id', batch_code))
            
            batch_results = {}
            batch_results["loss"] = loss.item()
            batch_results["recons_loss"] = recons_loss.item()
            batch_results["JS_div_nuclei"] = JS_div_nuclei.item()
            batch_results["JS_div_cell_nuclei"] = JS_div_cell_nuclei.item()
            batch_results["encoded"] = encoded_np.tolist()  
            batch_results["decoded"] = decoded_np.tolist()
            batch_results["mu"] = mu_np.tolist()
            batch_results["log_var"] = log_var_np.tolist()
            batch_results["cell_type"] = celltypes.tolist()
            batch_results["meta_batch_id"] = meta_batch_id.tolist()
            batch_results["cell_type_original"] = original_cell_types
            batch_results["meta_batch_id_original"] = original_batch_ids


            valid_epoch_save_dict[batch_idx] = batch_results

            if args.debug and batch_idx == 1:
                break

    model.train()

    return valid_epoch_save_dict, total_loss, total_recons_loss, total_JS_div_nuclei, total_JS_div_cell_nuclei

def load_vae_jsd(model_args, model_run_dir):
    model = VAE_JSD(
        input_size=model_args["num_feature"],
        num_internal=model_args["num_internal"],
        num_latent=model_args["num_latent"]
    )
    
    # model_path = sorted([
    #     f for f in os.listdir(os.path.join(model_run_dir, "models"))
    #     if f.startswith("model") and f.endswith(".pt")
    # ])[-1]
    full_model_path = os.path.join(model_run_dir, "models", "model.pt")
    
    # Always load to CPU first, then move to device if possible
    try:
        state_dict = torch.load(full_model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"Model loaded successfully from {full_model_path}")
        
        # Try to move to GPU, fall back to CPU if needed
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear cache first
                model = model.to(device)
                print(f"Model moved to {device}")
            else:
                print("CUDA not available, using CPU")
                
        except RuntimeError as e:
            print(f"Failed to move model to GPU: {e}")
            print("Using CPU instead")
            model = model.to('cpu')
            
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    model.eval()
    return model
