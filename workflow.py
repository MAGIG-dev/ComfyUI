import os
import uuid
import time
import yaml
import execution
import folder_paths


def run_workflow(workflow_file: str):
    with open(workflow_file, "r") as f:
        workflow = yaml.safe_load(f)
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

    print(folder_paths.folder_names_and_paths, "\n")
    return
