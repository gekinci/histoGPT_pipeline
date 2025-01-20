import textwrap
import requests
import h5py

from histogpt.helpers.patching import main, PatchingConfigs
from histogpt.helpers.inference import generate

import torch
device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def download_file_from_url(fname, url):
    response = requests.get(url)
    with open(fname, mode="wb") as file:
        file.write(response.content)


def run_ctranspath(img_folder, file_ending, save_patches):
    """
    Runs the CTranspath model to generate embeddings for the patches of the WSIs in the
    configured folder with the specified file ending

    Args:
        img_folder: The folder that contains the WSIs
        file_ending: File ending of WSIs
        save_patches: Whether to save the patches for debugging

    """

    configs = PatchingConfigs()
    configs.slide_path = img_folder
    configs.save_path = '../save_folder'
    configs.file_extension = file_ending
    configs.save_patch_images = save_patches
    configs.model_path = '../ctranspath.pth'
    configs.patch_size = 256
    configs.white_thresh = [170, 185, 175]
    configs.edge_threshold = 2
    configs.resolution_in_mpp = 0.0
    configs.downscaling_factor = 4.0
    configs.batch_size = 16

    main(configs)


def run_histogpt(h5_file, model, tokenizer, prompt):
    """
    Runs the HistoGPT model to generate the clinical reports from the embeddings

    Args:
        h5_file: Embeddings file
        model: HistoGPT model
        tokenizer: BioGPT tokenizer
        prompt: Prompt text

    Returns:
        str: Diagnosis text

    """
    with h5py.File(h5_file, 'r') as f:
        features = f['feats'][:]
        features = torch.tensor(features).unsqueeze(0).to(device)

        output = generate(
            model=model,
            prompt=prompt,
            image=features,
            length=256,
            top_k=40,
            top_p=0.95,
            temp=0.7,
            device=device
        )

        decoded = tokenizer.decode(output[0, 1:])
        diagnosis_text = textwrap.fill(decoded, width=64)

        return diagnosis_text
