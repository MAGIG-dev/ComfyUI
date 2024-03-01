import os
import git
import sys
import uuid
import time
import yaml  # type: ignore
import nodes
import zipfile
import execution
import subprocess
import folder_paths
import urllib.request
from torchvision.datasets.utils import download_url  # type: ignore


def run_workflow(workflow_file: str, extra_models: list[dict] = []):
    with open(workflow_file, "r") as f:
        workflow = yaml.safe_load(f)

        if not is_api_workflow(workflow):
            raise Exception(
                "Workflow is in the wrong format. Please use the API format."
            )

        # Randomize seed
        for node in workflow.values():
            if "input" in node:
                keys = ["seed", "noise_seed"]
                for key in keys:
                    if key in node["input"]:
                        new_seed = uuid.uuid4().int
                        print(
                            f"Randomizing {key} to {new_seed} for node {node['class_type']}"
                        )
                        node["input"][key] = new_seed

        install_missing_nodes(workflow)
        download_missing_models(workflow, extra_models)

        valid = execution.validate_prompt(workflow)

        if valid[0]:
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


def is_api_workflow(workflow) -> bool:
    return all("class_type" in node for node in workflow.values())


def install_missing_nodes(workflow):
    # Find missing nodes for workflow

    missing_nodes = find_missing_nodes(workflow)

    if len(missing_nodes) == 0:
        return

    print(f"Missing nodes for workflow: {missing_nodes}")

    # Install missing nodes

    this_dir = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(this_dir, "custom-node-list.json"), "r") as a:
        with open(os.path.join(this_dir, "extension-node-map.json"), "r") as b:
            package_db = yaml.safe_load(a)["custom_nodes"]
            package_node_map = yaml.safe_load(b)

            packages_to_download = []
            for url in package_node_map:
                for node_name in missing_nodes:
                    if node_name in package_node_map[url][0]:
                        package_name = package_node_map[url][1]["title_aux"]
                        if package_name not in packages_to_download:
                            packages_to_download.append(package_name)

            print(f"Packages to install: {packages_to_download}")

            entries = []
            for entry in package_db:
                if entry["title"] in packages_to_download:
                    entries.append(entry)

            packages_not_found = list(
                set(packages_to_download) - set([entry["title"] for entry in entries])
            )

            if len(packages_not_found) > 0:
                raise Exception(
                    f"Could not find download info for missing nodes: {packages_not_found}"
                )

            for entry in entries:
                install_type = entry["install_type"]

                if install_type == "unzip":
                    unzip_install(entry["files"])

                if install_type == "copy":
                    copy_install(entry["files"])

                if install_type == "git-clone":
                    gitclone_install(entry["files"])

                if "pip" in entry:
                    for pname in entry["pip"]:
                        install_cmd = [sys.executable, "-m", "pip", "install", pname]
                        try_install_script(install_cmd)

            nodes.load_custom_nodes()


def find_missing_nodes(workflow):
    missing_nodes = []

    for node in workflow.values():
        type = node["class_type"]
        all_node_types = nodes.NODE_CLASS_MAPPINGS.keys()
        if type not in all_node_types and type not in missing_nodes:
            missing_nodes.append(type)

    return missing_nodes


def unzip_install(files: list[str]):
    temp_filename = "manager-temp.zip"
    for url in files:
        if url.endswith("/"):
            url = url[:-1]
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            }

            req = urllib.request.Request(url, headers=headers)
            response = urllib.request.urlopen(req)
            data = response.read()

            with open(temp_filename, "wb") as f:
                f.write(data)

            with zipfile.ZipFile(temp_filename, "r") as zip_ref:
                zip_ref.extractall(
                    folder_paths.folder_names_and_paths["custom_nodes"][0][0]
                )

            os.remove(temp_filename)
        except Exception as e:
            print(f"Install(unzip) error: {url}", file=sys.stderr)
            raise e

    print("Installation was successful.")


def copy_install(files: list[str]):
    for url in files:
        if url.endswith("/"):
            url = url[:-1]
        try:
            if url.endswith(".py"):
                download_url(
                    url, folder_paths.folder_names_and_paths["custom_nodes"][0][0]
                )

        except Exception as e:
            print(f"Install(copy) error: {url}", file=sys.stderr)
            raise e

    print("Installation was successful.")


def gitclone_install(files):
    for url in files:
        if url.endswith("/"):
            url = url[:-1]
        try:
            repo_name = os.path.splitext(os.path.basename(url))[0]
            repo_path = os.path.join(
                folder_paths.folder_names_and_paths["custom_nodes"][0][0], repo_name
            )

            # Clone the repository from the remote URL
            repo = git.Repo.clone_from(url, repo_path, recursive=True)
            repo.git.clear_cache()
            repo.close()

            print(f"Cloned repository {url} to {repo_path}")

            execute_install_script(repo_path)

        except Exception as e:
            print(f"Install(git-clone) error: {url}", file=sys.stderr)
            raise e


def execute_install_script(repo_path):
    install_script_path = os.path.join(repo_path, "install.py")
    requirements_path = os.path.join(repo_path, "requirements.txt")

    if os.path.exists(requirements_path):
        with open(requirements_path, "r") as requirements_file:
            for line in requirements_file:
                package_name = line.strip()
                if package_name:
                    install_cmd = [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        package_name,
                    ]
                    if package_name.strip() != "":
                        try_install_script(install_cmd, repo_path)

    if os.path.exists(install_script_path):
        install_cmd = [sys.executable, "install.py"]
        try_install_script(install_cmd, repo_path)


def try_install_script(cmd, cwd="."):
    subprocess.run(args=cmd, cwd=cwd, check=True)


def download_missing_models(workflow, extra_models: list[dict] = []):
    if not os.path.exists(folder_paths.models_dir):
        os.makedirs(folder_paths.models_dir)

    # Find required models for workflow

    used_models = find_used_models(workflow)
    for entry in extra_models:
        if entry["filename"] not in used_models:
            used_models.append(entry["filename"])

    print(f"Required models for workflow: {used_models}")

    # Check if models already exist anywhere in the folder paths

    models_to_download = []

    for model_name in used_models:
        exists = any(
            os.path.exists(os.path.join(folder_paths.models_dir, folder, model_name))
            for folder in os.listdir(folder_paths.models_dir)
        )

        if not exists and folder_paths.comfy_path is not folder_paths.base_path:
            comfy_models_dir = os.path.join(folder_paths.comfy_path, "models")
            exists = any(
                os.path.exists(os.path.join(comfy_models_dir, folder, model_name))
                for folder in os.listdir(comfy_models_dir)
            )

        if not exists:
            models_to_download.append(model_name)

    if len(models_to_download) == 0:
        return

    print(f"Models not download yet: {models_to_download}")

    # Download models

    this_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(this_dir, "model-list.json"), "r") as f:
        model_db = yaml.safe_load(f)["models"]
        model_db.extend(extra_models)

        entries = []
        for entry in model_db:
            if entry["filename"] in models_to_download:
                entries.append(entry)

        models_not_found = list(
            set(models_to_download) - set([entry["filename"] for entry in entries])
        )

        if len(models_not_found) > 0:
            raise Exception(
                f"Could not find download info for required models: {models_not_found}"
            )

        for entry in entries:
            model_dir = get_model_dir(entry)
            download_url(entry["url"], model_dir, filename=entry["filename"])


# This function finds all the models used in a workflow by scanning the inputs
# of each node and checking it against a list of filetypes
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


# extracts the directory in which the model should be saved
def get_model_dir(entry):
    if entry["save_path"] != "default":
        if ".." in entry["save_path"] or entry["save_path"].startswith("/"):
            print(
                f"[WARN] '{entry['save_path']}' is not allowed path. So it will be saved into 'models/etc'."
            )
            dir = os.path.join(folder_paths.models_dir, "etc")
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
            dir = os.path.join(folder_paths.models_dir, "etc")

    return dir


# We do this because the get_model_dir function uses the first path in the
# `folder_names_and_paths` dictionary to save models and custom nodes
def adjust_folder_names_and_paths(new_base_path: str):
    if folder_paths.base_path == new_base_path:
        return

    for category in folder_paths.folder_names_and_paths:
        if category == "custom_nodes":
            continue

        # for every category, replace the base path with the new base path
        for i, folder in enumerate(folder_paths.folder_names_and_paths[category][0]):
            # skip output folders added in the main script
            if folder.startswith(f"{folder_paths.base_path}/output"):
                continue

            folder_paths.folder_names_and_paths[category][0][i] = folder.replace(
                folder_paths.base_path, new_base_path
            )

        # add back in the original base path
        folder_paths.folder_names_and_paths[category][0].append(
            os.path.join(folder_paths.base_path, category)
        )

    print(f"Switching base_path from {folder_paths.base_path} to {new_base_path}")
    folder_paths.base_path = new_base_path
    folder_paths.models_dir = os.path.join(new_base_path, "models")

    print(f"{folder_paths.folder_names_and_paths}")

    return
