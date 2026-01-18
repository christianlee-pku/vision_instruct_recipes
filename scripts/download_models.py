import os
from huggingface_hub import snapshot_download

def download_models_locally():
    # 创建本地缓存目录 / Create local cache directory
    base_dir = os.path.join(os.getcwd(), "model_cache")
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"📂 Models will be downloaded to: {base_dir}")

    # 1. 下载 CLIP - 下载所有文件，不使用 ignore_patterns (除了非PyTorch权重)
    clip_id = "openai/clip-vit-large-patch14-336"
    clip_local_path = os.path.join(base_dir, "clip-vit-large-patch14-336")
    print(f"⬇️ Downloading CLIP to {clip_local_path}...")
    snapshot_download(
        repo_id=clip_id, 
        local_dir=clip_local_path, 
        ignore_patterns=["*.h5", "*.tf", "*.msgpack"], # 仅排除 TF/Jax/Flax
        resume_download=True
    )
    
    # 2. 下载 LLM (SmolLM)
    llm_id = "HuggingFaceTB/SmolLM-135M"
    llm_local_path = os.path.join(base_dir, "SmolLM-135M")
    print(f"⬇️ Downloading LLM to {llm_local_path}...")
    snapshot_download(
        repo_id=llm_id, 
        local_dir=llm_local_path,
        ignore_patterns=["*.h5", "*.tf", "*.msgpack"],
        resume_download=True
    )

    print("\n✅ Download Complete!")
    
    # 将路径写入环境变量文件
    with open("model_paths.env", "w") as f:
        f.write(f"LOCAL_CLIP_PATH={clip_local_path}\n")
        f.write(f"LOCAL_LLM_PATH={llm_local_path}\n")

if __name__ == "__main__":
    download_models_locally()
