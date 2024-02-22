import os
import uuid
import time
import yaml  # type: ignore
import execution
import folder_paths
from torchvision.datasets.utils import download_url  # type: ignore


def run_workflow(workflow_file: str, new_base_path: str | None):
    with open(workflow_file, "r") as f:
        workflow = yaml.safe_load(f)
        valid = execution.validate_prompt(workflow)

        if valid[0]:
            if new_base_path:
                adjust_folder_names_and_paths(new_base_path)

            download_missing_models(workflow)

            prompt_id = str(uuid.uuid4())
            outputs_to_execute = valid[2]

            execution_start_time = time.perf_counter()
            e = execution.PromptExecutor()

            e.execute(workflow, prompt_id, {}, outputs_to_execute)

            current_time = time.perf_counter()
            execution_time = current_time - execution_start_time

            print("Prompt executed in {:.2f} seconds".format(execution_time))
        else:
            print("invalid prompt:", valid[1])
            print("node errors:", valid[3])


def download_missing_models(workflow, extra_models: list[str] = []):

    # Find required models for workflow

    models = find_used_models(workflow)
    models.extend(extra_models)

    print(f"Required models for workflow: {models}")

    # Check if models already exist anywhere in the folder paths

    models_to_download = []
    skip_checking_categories = ["custom_nodes"]

    for model_name in models:
        exists = any(
            os.path.exists(os.path.join(folder, model_name))
            for category in folder_paths.folder_names_and_paths
            if category not in skip_checking_categories
            for folder in folder_paths.folder_names_and_paths[category][0]
        )

        if not exists:
            models_to_download.append(model_name)

    if len(models_to_download) == 0:
        return

    print(f"Models not download yet: {models_to_download}")

    # Download models

    this_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(this_dir, "model-list.json"), "r") as f:
        model_list = yaml.safe_load(f)["models"]

        entries = []
        for model_entry in model_list:
            for model_name in models_to_download:
                if model_entry["name"] == model_name:
                    entries.append(model_entry)

        missing_models = list(
            set(models_to_download) - set([entry["name"] for entry in entries])
        )

        if len(missing_models) > 0:
            raise Exception(
                f"Could not find download info for required models: {missing_models}"
            )

        for entry in entries:
            model_dir = get_model_dir(entry)
            filepath = os.path.join(model_dir, entry["name"])
            print(f"Downloading {entry['name']} --> {filepath}")

            download_url(entry["url"], model_dir, filename=entry["filename"])


# This function finds all the models used in a workflow by scanning the
# inputs of each node and checking it against a list of filetypes
def find_used_models(workflow) -> list[str]:
    scan_for_filetypes = folder_paths.supported_pt_extensions
    models = []

    for node_id in workflow:
        node = workflow[node_id]
        if "inputs" in node:
            for input in node["inputs"]:
                value = node["inputs"][input]
                if isinstance(value, str) and value.endswith(tuple(scan_for_filetypes)):
                    models.append(value)

    return models


# Taken from ComfyUI Manager
# extracts the directory in which the model should be saved
def get_model_dir(entry):
    if entry["save_path"] != "default":
        if ".." in entry["save_path"] or entry["save_path"].startswith("/"):
            print(
                f"[WARN] '{entry['save_path']}' is not allowed path. So it will be saved into 'models/etc'."
            )
            dir = os.path.join(folder_paths.base_path, "models/etc")
        else:
            if entry["save_path"].startswith("custom_nodes"):
                dir = os.path.join(
                    folder_paths.folder_names_and_paths["custom_nodes"][0][0],
                    "..",
                    entry["save_path"],
                )
            else:
                dir = os.path.join(folder_paths.models_dir, entry["save_path"])
    else:
        model_type = entry["type"]
        if model_type == "checkpoints":
            dir = folder_paths.folder_names_and_paths["checkpoints"][0][0]
        elif model_type == "unclip":
            dir = folder_paths.folder_names_and_paths["checkpoints"][0][0]
        elif model_type == "VAE":
            dir = folder_paths.folder_names_and_paths["vae"][0][0]
        elif model_type == "lora":
            dir = folder_paths.folder_names_and_paths["loras"][0][0]
        elif model_type == "T2I-Adapter":
            dir = folder_paths.folder_names_and_paths["controlnet"][0][0]
        elif model_type == "T2I-Style":
            dir = folder_paths.folder_names_and_paths["controlnet"][0][0]
        elif model_type == "controlnet":
            dir = folder_paths.folder_names_and_paths["controlnet"][0][0]
        elif model_type == "clip_vision":
            dir = folder_paths.folder_names_and_paths["clip_vision"][0][0]
        elif model_type == "gligen":
            dir = folder_paths.folder_names_and_paths["gligen"][0][0]
        elif model_type == "upscale":
            dir = folder_paths.folder_names_and_paths["upscale_models"][0][0]
        elif model_type == "embeddings":
            dir = folder_paths.folder_names_and_paths["embeddings"][0][0]
        else:
            dir = os.path.join(folder_paths.base_path, "models/etc")

    return dir


# We do this because ComfyUI Manager uses the first path in the `folder_names_and_paths` dictionary
# to save models and custom nodes
def adjust_folder_names_and_paths(new_base_path: str):
    if folder_paths.base_path == new_base_path:
        return

    for category in folder_paths.folder_names_and_paths:
        # for every category, replace the base path with the new base path
        for i, folder in enumerate(folder_paths.folder_names_and_paths[category][0]):
            # skip output folders added in the main script
            if folder.startswith(f"{folder_paths.base_path}/output"):
                continue

            folder_paths.folder_names_and_paths[category][0][i] = folder.replace(
                folder_paths.base_path, new_base_path
            )

        # add back in original custom nodes folder for ComfyUI Manager installation
        if category == "custom_nodes":
            folder_paths.folder_names_and_paths[category][0].append(
                os.path.join(folder_paths.base_path, "custom_nodes")
            )

    folder_paths.base_path = new_base_path
    print(folder_paths.folder_names_and_paths, "\n")
    return
