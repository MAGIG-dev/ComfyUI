import uuid
import time
import yaml
import execution


def run_workflow(workflow_file: str):
    with open(workflow_file, "r") as f:
        workflow = yaml.safe_load(f)
        valid = execution.validate_prompt(workflow)

        if valid[0]:
            alter_base_path()

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


def alter_base_path():
    base_path = "/home/ubuntu/projects/comfy_ui_cache"
    # TODO: implement
    return
