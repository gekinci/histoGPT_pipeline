from dagster import In, Nothing, op, job, Config
import os, glob
import subprocess
import sys
import pandas as pd

from transformers import BioGptTokenizer, BioGptConfig
import torch
device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from ml_pipeline.helpers import run_ctranspath, run_histogpt, download_file_from_url

try:
    from histogpt.models import HistoGPTForCausalLM, PerceiverResamplerConfig
except ImportError:
    repo = "https://github.com/marrlab/HistoGPT"
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"git+{repo}"])
    try:
        from histogpt.models import HistoGPTForCausalLM, PerceiverResamplerConfig
    except ImportError:
        pass


class MyOpConfig(Config):
    folder_path: str = "../"
    file_ending: str = ".ndpi"
    save_patches: bool = True


@op
def get_models(context):
    ctrans_path = "../ctranspath.pth"
    histo_path = "../histogpt-1b-6k-pruned.pth"
    input_path = "../2023-03-06%2023.51.44.ndpi"


    if not os.path.isfile(histo_path):
        context.log.info(f"Input image cannot be found. Downloading...")
        download_file_from_url(input_path,
                               url="https://huggingface.co/marr-peng-lab/histogpt/resolve/main/2023-03-06%2023.51.44.ndpi")
        context.log.info(f"Done.")
    else:
        pass

    if not os.path.isfile(ctrans_path):
        context.log.info(f"CTranspath model cannot be found. Downloading...")
        download_file_from_url(ctrans_path,
                               url="https://huggingface.co/marr-peng-lab/histogpt/resolve/main/ctranspath.pth")
        context.log.info(f"Done.")
    else:
        pass

    if not os.path.isfile(histo_path):
        context.log.info(f"HistoGPT model cannot be found. Downloading...")
        download_file_from_url(histo_path,
                               url="https://huggingface.co/marr-peng-lab/histogpt/resolve/main/histogpt-1b-6k-pruned.pth")
        context.log.info(f"Done.")
    else:
        pass


@op(ins={"start": In(Nothing)})
def generate_and_save_embeddings(context, config:MyOpConfig):
    try:
        os.mkdir('../save_folder')
    except Exception:
        pass

    image_path = config.folder_path
    file_ending = config.file_ending
    save_patches = config.save_patches
    context.log.info(f"Scanning folder: {image_path}")
    context.log.info(f"Looking for files with ending: {file_ending}")
    context.log.info(f"Save patches: {save_patches}")

    context.log.info(f"Generating embeddings...")
    run_ctranspath(image_path, file_ending, save_patches)
    embeddings_folder = "../save_folder/h5_files/256px_ctranspath_0.0mpp_4.0xdown_normal"
    return embeddings_folder


@op
def generate_and_save_clinical_reports(context, embeddings_path):
    txt_folder = "../save_folder/txt_files"
    try:
        os.mkdir(txt_folder)
    except Exception:
        pass

    context.log.info(f"Configuring histogpt and loading weights...")
    histogpt_model = HistoGPTForCausalLM(BioGptConfig(), PerceiverResamplerConfig())
    histogpt_model = histogpt_model.to(device)

    PATH = '../histogpt-1b-6k-pruned.pth'
    state_dict = torch.load(PATH, map_location=device)
    histogpt_model.load_state_dict(state_dict, strict=True)

    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

    prompt = 'Final diagnosis:'
    prompt = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    for h5_file in glob.glob(f"{embeddings_path}/*.h5"):
        img_name = h5_file.split("/")[-1][:-3]
        txt_file = f"{txt_folder}/wsi_{img_name}.txt"

        context.log.info(f"Generating diagnosis for image: {img_name}...")
        diagnosis = run_histogpt(h5_file, histogpt_model, tokenizer, prompt)

        context.log.info(f"Saving diagnosis for image into .txt: {txt_file}...")
        with open(txt_file, "w") as file:
            file.write(diagnosis)

    context.log.info(f"All clinical reports have been saved into {txt_folder}.")
    return txt_folder


@op
def aggregate_and_save_txt_to_csv(context, txt_path):
    context.log.info(f"Aggregating .txt files into a dataframe...")
    results_list = []
    for txt_file in glob.glob(f"{txt_path}/*.txt"):
        img_name = txt_file.split("/")[-1][4:-4]
        with open(txt_file, "r") as file:
            content = file.read()
        row_dict = {'Image':img_name, 'Diagnosis':content}
        results_list.append(row_dict)

    df_results = pd.DataFrame(results_list)

    context.log.info(f"Saving results as result.csv...")
    df_results.to_csv("../save_folder/result.csv")


@job(tags={"ecs/cpu": "4096", "ecs/memory": "32384"})
def ml_pipeline():
    embeddings_folder = generate_and_save_embeddings(start=get_models())
    txt_folder = generate_and_save_clinical_reports(embeddings_folder)
    aggregate_and_save_txt_to_csv(txt_folder)


if __name__ == "__main__":
    ml_pipeline.execute_in_process()
