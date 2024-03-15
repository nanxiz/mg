from flask import Flask, request, jsonify
from pathlib import Path
import pytorch_lightning as pl
import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model

app = Flask(__name__)

def load_cfg_from_file(file_path):
    cfg = OmegaConf.load(file_path)
    return cfg

def setup_model():
    cfg = load_cfg_from_file('./config_mix.yaml')
    cfg.FOLDER = 'cache'
    output_dir = Path(cfg.FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(cfg.SEED_VALUE)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_path = snapshot_download(repo_id="bill-jiang/MotionGPT-base")
    
    datamodule = build_data(cfg, phase="test")
    model = build_model(cfg, datamodule)
    state_dict = torch.load(f'{model_path}/motiongpt_s3_h3d.tar', map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    return model, device

model, device = setup_model()

@app.route('/generate_motion', methods=['POST'])
def generate_motion():
    data = request.json
    input_text = data['text']
    
    # Assuming `model` and `device` are already set up
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
    
    motion_output = outputs
    
    return jsonify(motion_output)

if __name__ == '__main__':
    app.run(debug=True, port=9880)
