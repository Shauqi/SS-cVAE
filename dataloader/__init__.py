from .brca_loader import BRCA_BIN_Paired_File_Loader, BRCA_BIN_File_Loader, BRCA_TIL_VS_Other_File_Loader, BRCA_GAN_File_Loader, BRCA_MTL_File_Loader

def get_datasets(config):
    model_name = config['CVAE_MODEL_TRAIN']['model_name']
    train_dir = f"{config['PROJECT_DIR']}{config['CVAE_MODEL_TRAIN']['train_dir']}"
    val_dir = f"{config['PROJECT_DIR']}{config['CVAE_MODEL_TRAIN']['val_dir']}"
    test_dir = f"{config['PROJECT_DIR']}{config['CVAE_MODEL_TRAIN']['test_dir']}"

    if model_name == 'chc_vae' or model_name == 'ch_vae' or model_name == 'ss_cvae_ablation' or model_name == 'ss_cvae' or model_name == 'resnet_cvae' or model_name == 'ss_cvae_ablation':
        train_ds = BRCA_BIN_Paired_File_Loader(train_dir)
        valid_ds = BRCA_BIN_File_Loader(val_dir)
        test_ds = BRCA_BIN_File_Loader(test_dir, shuffle=False)
    elif model_name == 'ss_cvae_one_stage_ablation':
        train_ds = BRCA_MTL_File_Loader(train_dir)
        valid_ds = BRCA_MTL_File_Loader(val_dir)
        test_ds = BRCA_BIN_File_Loader(test_dir, shuffle=False)
    else:
        train_ds = BRCA_BIN_File_Loader(train_dir)
        valid_ds = BRCA_BIN_File_Loader(val_dir)
        test_ds = BRCA_BIN_File_Loader(test_dir, shuffle=False)

    return train_ds, valid_ds, test_ds

def get_til_vs_other_datasets(config):
    train_dir = f"{config['PROJECT_DIR']}{config['CONTRASTIVE_MODEL_TRAIN']['train_dir']}"
    val_dir = f"{config['PROJECT_DIR']}{config['CONTRASTIVE_MODEL_TRAIN']['val_dir']}"
    test_dir = f"{config['PROJECT_DIR']}{config['CONTRASTIVE_MODEL_TRAIN']['test_dir']}"

    train_ds = BRCA_TIL_VS_Other_File_Loader(train_dir)
    valid_ds = BRCA_TIL_VS_Other_File_Loader(val_dir, shuffle=False)
    test_ds = BRCA_TIL_VS_Other_File_Loader(test_dir, shuffle=False)

    return train_ds, valid_ds, test_ds