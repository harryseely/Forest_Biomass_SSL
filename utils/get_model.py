
from models.ocnn_lightning import OCNNLightning

def get_model(cfg):

    if cfg['task'] == 'finetune':
        assert cfg['model_ckpt_fpath'] is not None, "No model checkpoint specified for finetuning."
    elif cfg['task'] == 'scratch':
        assert cfg['model_ckpt_fpath'] is None, "Model checkpoint should not be specified for training from scratch."

    #Instantiate model
    model = OCNNLightning(cfg)

    #Load checkpoint if one is specified for finetuning
    if cfg['task'] == 'finetune':
        assert cfg['model_ckpt_fpath'] is not None, "No model checkpoint specified for finetuning."

        try:
            model.load_checkpoint(cfg['model_ckpt_fpath'])

        except Exception as e:
            print(f"Error loading pretraining model checkpoint: {e}")
            raise e
    
    elif cfg['task'] == 'scratch':
        assert cfg['model_ckpt_fpath'] is None, "Model checkpoint should not be specified for training from scratch."
    
    return model


