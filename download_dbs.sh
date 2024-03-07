#/bin/bash

set -exo pipefail

# Clone ComfyUI Manager repository to a tmp folder
tmp_folder="/tmp/comfyui_manager"
repo_url="https://github.com/ltdrdata/ComfyUI-Manager.git"

git clone $repo_url $tmp_folder

# Copy the database files to the this folder
custom_nodes_db="custom-node-list.json"
extenstion_node_map="extension-node-map.json"
models_db="model-list.json"

cp $tmp_folder/$custom_nodes_db .
cp $tmp_folder/$extenstion_node_map .
cp $tmp_folder/$models_db .

# Remove the tmp folder
rm -rf $tmp_folder
