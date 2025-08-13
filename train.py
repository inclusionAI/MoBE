import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from safetensors import safe_open
import os
from safetensors.torch import save_file
import copy
import argparse
import gc

class MoBE(nn.Module):
    def __init__(self, initial_A, initial_B, initial_W, activation='silu'):
        super().__init__()
        self.A_params = nn.Parameter(initial_A)
        self.B_params = nn.Parameter(initial_B)
        self.w_params = nn.Parameter(initial_W)

        if activation not in ('silu', 'tanh'):
            raise ValueError("activation must be 'silu' or 'tanh'")
        self.activation = activation

    def forward(self, batch_indices):
        A = self.A_params[batch_indices]
        w = self.w_params[batch_indices]
        w = torch.softmax(w, dim=1)
        weighted_B = torch.einsum('bi,ijk->bjk', w, self.B_params)

        if self.activation == 'silu':
            activation = nn.functional.silu(weighted_B)
        else:  
            activation = torch.tanh(weighted_B)

        hat_Z = A @ activation
        return hat_Z


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


def parse_args():
    parser = argparse.ArgumentParser(description="MoBE Training")
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    parser.add_argument("--num_hidden_layers", type=int, default=94)
    parser.add_argument("--num_matrices", type=int, default=128)
    parser.add_argument("--rows_per_matrix", type=int, default=1536)
    parser.add_argument("--cols", type=int, default=4096)

    parser.add_argument("--num_epochs", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.07)

    parser.add_argument("--num_B", type=int, default=32)
    parser.add_argument("--truncation", type=int, default=1536)
    parser.add_argument("--start_layer", type=int, default=0)
    parser.add_argument("--end_layer", type=int, default=94)

    parser.add_argument("--matrix_type", type=str, choices=["gate_proj", "up_proj"], default="gate_proj")

    parser.add_argument("--activation", type=str, choices=["silu", "tanh"], default="silu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.save_path, exist_ok=True)

    num_group = args.num_B
    k = args.truncation
    num_matrices_per_group = args.num_matrices // num_group

    for n in range(args.start_layer, args.end_layer):
        best_real_loss = float('inf')
        best_model_state = None
        print(f'layer: {n}')

        state_dict = get_layer_proj_dict(args.index_path, args.base_dir, layer_i=n, matrix_type=args.matrix_type)
        target_list = []
        for i in range(args.num_matrices):
            target_list.append(state_dict[f"model.layers.{n}.mlp.experts.{i}.{args.matrix_type}.weight"].to(torch.float16).to(device))
        target = torch.stack(target_list)
        global_target_std = target.std()

        initial_A, initial_B, initial_W = None, None, None
        for group_i in range(num_group):
            W = target[group_i*num_matrices_per_group:(group_i+1)*num_matrices_per_group].view(-1, args.cols)
            W_float32 = W.to(torch.float32)
            U, S, Vt = torch.linalg.svd(W_float32, full_matrices=False)
            U_k = U[:, :k]
            S_k = S[:k]
            Vt_k = Vt[:k, :]

            group_A = U_k @ torch.diag(S_k)
            group_B = Vt_k.unsqueeze(0)
            group_W = torch.zeros(num_matrices_per_group, num_group, device=device)
            group_W[:, group_i] = 1

            if initial_B is None:
                initial_B = group_B
            else:
                initial_B = torch.cat((initial_B, group_B), dim=0)

            for weight_i in range(num_matrices_per_group):
                start_r = weight_i * args.rows_per_matrix
                end_r = (weight_i + 1) * args.rows_per_matrix
                single_A = group_A[start_r:end_r, :]
                single_W = group_W[start_r:end_r, :]

                initial_A = single_A.unsqueeze(0) if initial_A is None else torch.cat((initial_A, single_A.unsqueeze(0)), dim=0)
                initial_W = single_W if initial_W is None else torch.cat((initial_W, single_W), dim=0)

        model = MoBE(initial_A, initial_B, initial_W, activation=args.activation).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        grad_accum = args.num_batches

        for epoch in range(args.num_epochs):
            epoch_loss, epoch_real_loss, epoch_real_mae = 0.0, 0.0, 0.0
            optimizer.zero_grad()
            for batch_idx in range(args.num_batches):
                start_idx = batch_idx * args.batch_size
                end_idx = min((batch_idx + 1) * args.batch_size, args.num_matrices)
                indices = torch.arange(start_idx, end_idx, device=device)
                if len(indices) == 0:
                    continue

                outputs = model(indices)
                batch_target = target[indices]

                Z_scaled = batch_target / global_target_std
                Z_hat_unscaled = outputs * global_target_std

                loss = F.mse_loss(outputs, Z_scaled.to(torch.float32))
                real_loss = F.mse_loss(Z_hat_unscaled, batch_target)
                real_mae = F.l1_loss(Z_hat_unscaled, batch_target)

                (loss / grad_accum).backward()
                epoch_loss += loss.item() * len(indices)
                epoch_real_loss += real_loss.item() * len(indices)
                epoch_real_mae += real_mae.item() * len(indices)

            optimizer.step()
            epoch_loss /= args.num_matrices
            epoch_real_loss /= args.num_matrices
            epoch_real_mae /= args.num_matrices

            if (epoch + 1) % 200 == 0 and epoch_real_loss < best_real_loss:
                best_real_loss = epoch_real_loss
                best_model_state = copy.deepcopy(model.state_dict())

            if (epoch + 1) % 200 == 0:
                print(f"Epoch {epoch+1}, Scaled Loss: {epoch_loss:.10f}, Real MSE Loss: {epoch_real_loss:.10f}, Real MAE Loss: {epoch_real_mae:.10f}")

        model.load_state_dict(best_model_state)

        model.eval()
        reconstructed_dict = {}
        with torch.no_grad():
            indices = torch.arange(0, args.num_matrices, device=device)
            outputs = model(indices)
            Z_hat_unscaled = outputs * global_target_std
            for weight_i in range(args.num_matrices):
                key = f'experts_{weight_i}_{args.matrix_type}_weight'
                reconstructed_dict[key] = Z_hat_unscaled[weight_i, :]

        output_path = f'{args.save_path}/model_layers_{n}_mlp_{args.matrix_type}_weight.safetensors'
        save_file(reconstructed_dict, output_path)

        out_file = f'{args.save_path}/model_layers_{n}_mlp_{args.matrix_type}_WAB.pth'
        torch.save(model.state_dict(), out_file)
        print(f"Saved best model weights for layer {n} at {out_file} with loss {best_real_loss:.10f}")

        model_params = sum(p.numel() for p in model.parameters())
        compression_ratio = model_params / (args.num_matrices * args.rows_per_matrix * args.cols)
        print(f'Layer: {n}, Compression Ratio: {compression_ratio:.5f}')

        del target, target_list, state_dict, model, optimizer, model_params
        if 'initial_A' in globals():
            del initial_A
        if 'initial_B' in globals():
            del initial_B
        if 'initial_W' in globals():
            del initial_W

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()