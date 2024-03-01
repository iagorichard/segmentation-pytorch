import os

def iterate_dict(dictx):
    for chave, valor in dictx.items():
        yield {chave: valor}

def create_checkpoints_folders(infos_dict, models_list):
    base_path = os.path.join(infos_dict["checkpoints_folder"], 
                             infos_dict["dataset_name"],
                             infos_dict["subset"],
                             infos_dict["task"],
                             infos_dict["learning"])

    # Create base directory and subsequent folders
    os.makedirs(base_path, exist_ok=True)

    # Create subfolders for each model in models_list
    for model in models_list:
        model_path = os.path.join(base_path, model)
        for folder in ['best', 'last']:
            folder_path = os.path.join(model_path, folder)
            os.makedirs(folder_path, exist_ok=True)

def get_checkpoint_folders(infos_dict, model_name):
    base_path = os.path.join(infos_dict["checkpoints_folder"], 
                             infos_dict["dataset_name"],
                             infos_dict["subset"],
                             infos_dict["task"],
                             infos_dict["learning"],
                             model_name)
    
    return base_path, os.path.join(base_path, "last"), os.path.join(base_path, "best")
