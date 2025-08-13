import os
import json
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from models.modeling_deepseek_v3_mobe import DeepseekV3MoBEForCausalLM 
from models.modeling_qwen3_mobe import Qwen3MoBEForCausalLM
from models.modeling_kimi_k2_mobe import KimiK2MoBEForCausalLM
from models.modeling_bailing_mobe import BailingMoBEForCausalLM
from safetensors import safe_open
from safetensors.torch import load_file as torch_load_file
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Convert MoE to MoBE format with B/W/A parameters.")

    # Model paths
    parser.add_argument('--base_model', type=str, required=True,
                        help='Path to the base model checkpoint (HuggingFace format).')
    parser.add_argument('--mobe_dir', type=str, required=True,
                        help='Directory containing WAB .safetensors files')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to save the converted MoBE model')

    # Model config
    parser.add_argument('--num_B', type=int, default=32,
                        help='Number of basis matrices in MoBE')
    parser.add_argument('--num_experts', type=int, default=128,
                        help='Total number of experts per layer')
    parser.add_argument('--start_layer', type=int, default=0,
                        help='Start layer index for MoBE conversion.')
    parser.add_argument('--end_layer', type=int, default=94,
                        help='End layer index for MoBE conversion')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['bfloat16', 'float16', 'float32'],
                        help='Data type for weights')
    parser.add_argument("--activation", type=str, choices=["silu", "tanh"], default="silu")
    parser.add_argument("--grouped_experts", action="store_true",
                        help="If set, experts are stored in multiple grouped files like "
                             "model_layers_0_mlp_gate_proj_group0_WAB.pth")

    return parser.parse_args()


def get_layer_proj_dict(index_path, base_dir, layer_i, matrix_type):
    with open(index_path, 'r') as f:
        index_data = json.load(f)
    layer_dict = {}
    for weight_name, safetensor_file in index_data['weight_map'].items():
        if f"model.layers.{layer_i}.mlp.experts" in weight_name and weight_name.endswith(f"{matrix_type}.weight"):
            safetensor_path = f"{base_dir}/{safetensor_file}"
            with safe_open(safetensor_path, framework="pt") as f:
                layer_dict[weight_name] = f.get_tensor(weight_name)
    return layer_dict


def main(args):
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32
    }
    dtype = dtype_map[args.dtype]

    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)

    config.num_B = args.num_B
    config.activation = args.activation

    if "Qwen3" in args.base_model:
        model = Qwen3MoBEForCausalLM.from_pretrained(
            args.base_model,
            ignore_mismatched_sizes=True,
            config=config,
            device_map='auto',
            torch_dtype=dtype
        )
    elif "DeepSeek" in args.base_model:
        model = DeepseekV3MoBEForCausalLM.from_pretrained(
            args.base_model,
            ignore_mismatched_sizes=True,
            config=config,
            device_map='auto',
            torch_dtype=dtype
        )
    elif "Kimi" in args.base_model:
        model = KimiK2MoBEForCausalLM.from_pretrained(
            args.base_model,
            ignore_mismatched_sizes=True,
            config=config,
            torch_dtype=dtype,
            device_map='auto',
            trust_remote_code=True
        )
    elif "Ling" in args.base_model:
        model = BailingMoBEForCausalLM.from_pretrained(
            args.base_model,
            ignore_mismatched_sizes=True,
            config=config,
            torch_dtype=dtype,
            device_map='auto',
            trust_remote_code=True
        )

    state_dict = model.state_dict()

    index_path = os.path.join(args.base_model, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Model index file not found: {index_path}")

    for i in tqdm(range(args.start_layer, args.end_layer), desc="Processing layers"):
        if args.grouped_experts:
            up_weight_dict = get_layer_proj_dict(index_path, args.base_model, i, ".up_proj")
            gate_weight_dict = get_layer_proj_dict(index_path, args.base_model, i, ".gate_proj")

            up_tensors = [
                up_weight_dict[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"].to(dtype)
                for j in range(args.num_experts)
            ]
            up_std = torch.stack(up_tensors).std()

            gate_tensors = [
                gate_weight_dict[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"].to(dtype)
                for j in range(args.num_experts)
            ]
            gate_std = torch.stack(gate_tensors).std()

            for g in range(2):
                gate_file = os.path.join(args.mobe_dir, f"model_layers_{i}_mlp_gate_proj_group{g}_WAB.pth")
                up_file = os.path.join(args.mobe_dir, f"model_layers_{i}_mlp_up_proj_group{g}_WAB.pth")

                for f_path in [gate_file, up_file]:
                    if not os.path.exists(f_path):
                        raise FileNotFoundError(f"Parameter file not found: {f_path}")

                gate_dict = torch.load(gate_file, map_location='cpu')
                up_dict = torch.load(up_file, map_location='cpu')

                state_dict[f'model.layers.{i}.mlp.B_up_{g}'] = up_dict['B_params']
                state_dict[f'model.layers.{i}.mlp.B_gate_{g}'] = gate_dict['B_params']
                state_dict[f'model.layers.{i}.mlp.W_up_{g}'] = torch.softmax(up_dict['w_params'], dim=1)
                state_dict[f'model.layers.{i}.mlp.W_gate_{g}'] = torch.softmax(gate_dict['w_params'], dim=1)

                for j in range(args.num_experts // 2):
                    state_dict[f'model.layers.{i}.mlp.experts.{j+(args.num_experts//2)*g}.gate_proj.weight'] = gate_dict['A_params'][j] * gate_std
                    state_dict[f'model.layers.{i}.mlp.experts.{j+(args.num_experts//2)*g}.up_proj.weight'] = up_dict['A_params'][j] * up_std
        
        else:
            gate_file = os.path.join(args.mobe_dir, f"model_layers_{i}_mlp_gate_proj_WAB.pth")
            up_file = os.path.join(args.mobe_dir, f"model_layers_{i}_mlp_up_proj_WAB.pth")

            for f_path in [gate_file, up_file]:
                if not os.path.exists(f_path):
                    raise FileNotFoundError(f"Parameter file not found: {f_path}")

            gate_dict = torch.load(gate_file, map_location='cpu')
            up_dict = torch.load(up_file, map_location='cpu')

            up_weight_dict = get_layer_proj_dict(index_path, args.base_model, i, ".up_proj")
            gate_weight_dict = get_layer_proj_dict(index_path, args.base_model, i, ".gate_proj")

            up_tensors = [
                up_weight_dict[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"].to(dtype)
                for j in range(args.num_experts)
            ]
            up_std = torch.stack(up_tensors).std()

            gate_tensors = [
                gate_weight_dict[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"].to(dtype)
                for j in range(args.num_experts)
            ]
            gate_std = torch.stack(gate_tensors).std()

            state_dict[f'model.layers.{i}.mlp.B_up'] = up_dict['B_params']
            state_dict[f'model.layers.{i}.mlp.B_gate'] = gate_dict['B_params']
            state_dict[f'model.layers.{i}.mlp.W_up'] = torch.softmax(up_dict['w_params'], dim=1)
            state_dict[f'model.layers.{i}.mlp.W_gate'] = torch.softmax(gate_dict['w_params'], dim=1)

            for j in range(args.num_experts):
                state_dict[f'model.layers.{i}.mlp.experts.{j}.gate_proj.weight'] = gate_dict['A_params'][j] * gate_std
                state_dict[f'model.layers.{i}.mlp.experts.{j}.up_proj.weight'] = up_dict['A_params'][j] * up_std

    model.load_state_dict(state_dict)
    model.save_pretrained(args.save_dir)
    config.save_pretrained(args.save_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.save_dir)
    print(f"✅ MoBE model saved to {args.save_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)