import os
import shutil

# Define model prefixes (folder names)
model_prefixes = ["irra_cuhk", "irra_icfg", "irra_rstp"]
types = ["image_encoder", "text_encoder"]

# Base repository directory
base_repo = "/home/jayn2u/IRRA/model_repository"
if not os.path.exists(base_repo):
    os.makedirs(base_repo)

# Configuration templates for config.pbtxt (Assuming 512 dim based on ViT-B/16)
config_templates = {
    "image_encoder": """name: "{model_name}"
platform: "tensorrt_plan"
max_batch_size: 32
input [
  {{
    name: "image"
    data_type: TYPE_FP16
    dims: [ 3, 256, 256 ]
  }}
]
output [
  {{
    name: "image_embeddings"
    data_type: TYPE_FP32
    dims: [ 512 ]
  }}
]
""",
    "text_encoder": """name: "{model_name}"
platform: "tensorrt_plan"
max_batch_size: 32
input [
  {{
    name: "text"
    data_type: TYPE_INT64
    dims: [ 77 ]
  }}
]
output [
  {{
    name: "text_embeddings"
    data_type: TYPE_FP32
    dims: [ 512 ]
  }}
]
"""
}

def setup_model(model_prefix, model_type):
    model_name = f"{model_prefix}_{model_type}"
    
    # Path to the source .plan file
    # Example: irra_cuhk/irra_cuhk_image_encoder.plan
    src_file_name = f"{model_name}.plan"
    src_path = os.path.join("/home/jayn2u/IRRA", model_prefix, src_file_name)
    
    if not os.path.exists(src_path):
        print(f"Skipping {model_name}: Source file not found at {src_path}")
        return

    # Destination directory structure: model_repository/<model_name>/1/
    model_dir = os.path.join(base_repo, model_name)
    version_dir = os.path.join(model_dir, "1")
    
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(version_dir, exist_ok=True)
    
    # Destination file path - Keep explicit name
    dst_file_name = f"{model_name}.plan"
    dst_path = os.path.join(version_dir, dst_file_name)
    
    # Copy the file
    print(f"Copying {src_path} to {dst_path}...")
    shutil.copy2(src_path, dst_path)
    
    # Create config.pbtxt with default_model_filename
    config_content = config_templates[model_type].format(model_name=model_name)
    # Add default_model_filename line
    config_content = f'default_model_filename: "{dst_file_name}"\n' + config_content
    
    config_path = os.path.join(model_dir, "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"Created config for {model_name} with default_model_filename")

def main():
    print(f"Setting up models in {base_repo}...")
    for prefix in model_prefixes:
        for model_type in types:
            setup_model(prefix, model_type)
    print("Done.")

if __name__ == "__main__":
    main()
