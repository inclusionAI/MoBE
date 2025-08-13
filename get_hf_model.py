import argparse
import os
import torch
from safetensors.torch import load_file as torch_load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Merge MoBE experts into a MoE model.")
    parser.add_argument("--base_model", required=True,
                        help="Path to the base model checkpoint (HuggingFace format).")
    parser.add_argument("--mobe_dir", required=True,
                        help="Directory that contains MoBE weights.")
    parser.add_argument("--save_dir", required=True,
                        help="Where to save the merged model.")
    parser.add_argument("--start_layer", type=int, default=0,
                        help="Start layer index for MoBE conversion.")
    parser.add_argument("--end_layer", type=int, default=94,
                        help="End layer index for MoBE conversion.")
    parser.add_argument("--num_experts", type=int, default=128,
                        help="Total number of experts per layer")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Data type for weights")
    parser.add_argument("--grouped_experts", action="store_true",
                        help="If set, experts are stored in multiple grouped files like "
                             "model_layers_0_mlp_gate_proj_group0.safetensors")
    return parser.parse_args()


def load_safetensors(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return torch_load_file(file_path)


def load_grouped_tensors(mobe_dir: str, layer_idx: int, proj_name: str, num_experts: int) -> dict:
    """
    Load grouped .safetensors files for a given layer and projection (gate/up).
    Files are expected to be named like:
        model_layers_{i}_mlp_{proj_name}_group{k}.safetensors
    Returns a merged dict with keys: f'experts_{j}_{proj_name}_weight'
    """
    pattern = re.compile(rf"model_layers_{layer_idx}_mlp_{proj_name}_group(\d+)\.safetensors$")
    group_files = []
    for f in os.listdir(mobe_dir):
        match = pattern.match(f)
        if match:
            group_files.append((int(match.group(1)), os.path.join(mobe_dir, f)))
    group_files.sort()

    if not group_files:
        raise FileNotFoundError(f"No grouped files found for layer {layer_idx}, proj {proj_name}")

    merged_dict = {}
    expert_idx = 0
    for _, file_path in group_files:
        tensors = load_safetensors(file_path)
        # Sort tensors by expert index to ensure order
        sorted_items = sorted(tensors.items(), key=lambda x: int(x[0].split('_')[1]))
        for k, v in sorted_items:
            new_key = f"experts_{expert_idx}_{proj_name}_weight"
            merged_dict[new_key] = v
            expert_idx += 1

    if expert_idx != num_experts:
        raise ValueError(f"Expected {num_experts} experts for layer {layer_idx}, "
                         f"but loaded {expert_idx} from grouped files.")

    return merged_dict


def main():
    args = parse_args()

    dtype_map = {"float16": torch.float16,
                 "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading base model from {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    state_dict = model.state_dict()

    for i in tqdm(range(args.start_layer, args.end_layer), desc="Merging experts"):
        if args.grouped_experts:
            # Load grouped files
            gate_dict = load_grouped_tensors(args.mobe_dir, i, "gate_proj", args.num_experts)
            up_dict = load_grouped_tensors(args.mobe_dir, i, "up_proj", args.num_experts)
        else:
            # Load single-file per layer
            gate_path = os.path.join(args.mobe_dir, f"model_layers_{i}_mlp_gate_proj_weight.safetensors")
            up_path = os.path.join(args.mobe_dir, f"model_layers_{i}_mlp_up_proj_weight.safetensors")
            gate_dict = load_safetensors(gate_path)
            up_dict = load_safetensors(up_path)

        # Merge experts
        for j in range(args.num_experts):
            gate_key = f'experts_{j}_gate_proj_weight'
            up_key = f'experts_{j}_up_proj_weight'

            state_dict[f'model.layers.{i}.mlp.experts.{j}.gate_proj.weight'] = gate_dict[gate_key]
            state_dict[f'model.layers.{i}.mlp.experts.{j}.up_proj.weight'] = up_dict[up_key]

    print("Loading merged state dict ...")
    model.load_state_dict(state_dict)
    print(f"Saving merged model to {args.save_dir}")
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print("Done.")


if __name__ == "__main__":
    main()