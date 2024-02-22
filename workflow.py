from server import PromptServer
from comfy.cli_args import args
from main import (
    cleanup_temp,
    load_extra_path_config,
    init_custom_nodes,
    cuda_malloc_warning,
)

import os
import uuid
import time
import yaml
import execution
import itertools
import folder_paths


def run_workflow_only():
    _server = PromptServer()

    if args.temp_directory:
        temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
        print(f"Setting temp directory to: {temp_dir}")
        folder_paths.set_temp_directory(temp_dir)
    cleanup_temp()

    extra_model_paths_config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml"
    )
    if os.path.isfile(extra_model_paths_config_path):
        load_extra_path_config(extra_model_paths_config_path)

    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):
            load_extra_path_config(config_path)

    init_custom_nodes()
    cuda_malloc_warning()

    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        print(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)

    # These are the default folders that checkpoints, clip and vae models will be saved to when using CheckpointSave, etc.. nodes
    folder_paths.add_model_folder_path(
        "checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints")
    )
    folder_paths.add_model_folder_path(
        "clip", os.path.join(folder_paths.get_output_directory(), "clip")
    )
    folder_paths.add_model_folder_path(
        "vae", os.path.join(folder_paths.get_output_directory(), "vae")
    )

    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        print(f"Setting input directory to: {input_dir}")
        folder_paths.set_input_directory(input_dir)

    with open(args.workflow, "r") as f:
        workflow = yaml.safe_load(f)
        valid = execution.validate_prompt(workflow)

        if valid[0]:
            prompt_id = str(uuid.uuid4())
            outputs_to_execute = valid[2]
            run_workflow(workflow, prompt_id, {}, outputs_to_execute)
        else:
            print("invalid prompt:", valid[1])
            print("node errors:", valid[3])

    cleanup_temp()


def run_workflow(prompt, prompt_id, extra_data={}, execute_outputs=[]):
    execution_start_time = time.perf_counter()
    e = execution.PromptExecutor()

    e.execute(prompt, prompt_id, extra_data, execute_outputs)

    current_time = time.perf_counter()
    execution_time = current_time - execution_start_time

    print("Prompt executed in {:.2f} seconds".format(execution_time))
