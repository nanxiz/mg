import os
from pydantic import BaseModel
import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from pathlib import Path
import pytorch_lightning as pl
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
import runpod
from typing import Union

class MotionRequest(BaseModel):
    text: str

def load_cfg_from_file(file_path):
    cfg = OmegaConf.load(file_path)
    return cfg

def setup_model():
    cfg = load_cfg_from_file('./config_mix.yaml')
    cfg.FOLDER = 'cache'
    output_dir = Path(cfg.FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(cfg.SEED_VALUE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = snapshot_download(repo_id="bill-jiang/MotionGPT-base")
    datamodule = build_data(cfg, phase="test")
    model = build_model(cfg, datamodule)
    state_dict = torch.load(f'{model_path}/motiongpt_s3_h3d.tar', map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    return model, device

model, device = setup_model()

def inference(event) -> Union[str, dict]:
    request = MotionRequest(**event)
    input_text = request.text
    
    motion_uploaded = {
        "feats": None,
        "joints": None,
        "motion_video": None,
        "motion_lengths": 0,
        "motion_token": None,
        "motion_token_string": '',
        "motion_token_length": 0,
    }
    motion_length, motion_token_string = motion_uploaded["motion_lengths"], motion_uploaded["motion_token_string"]
    prompt = model.lm.placeholder_fulfill(input_text, motion_length, motion_token_string, "")
    batch = {
            "length": [motion_length],
            "text": [prompt],
    }
    
    outputs = model(batch, task="t2m")
    
    joints = outputs['joints'].cpu().numpy().tolist() if outputs['joints'].is_cuda else outputs['joints'].numpy().tolist()
    length = outputs['length']
    
    return {"joints": joints, "length": length}

runpod.serverless.start({"handler": inference})
