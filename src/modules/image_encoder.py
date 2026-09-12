from .projection import Resampler
import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPVisionModelWithProjection

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders.lora import LoraLoaderMixin

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / torch.sqrt(torch.tensor(self.weight.size(1), dtype=torch.float32))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):

        x = x.to(dtype=self.weight.dtype)
        adj = adj.to(dtype=self.weight.dtype)

        support = torch.matmul(x, self.weight)
        output = torch.bmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output

class NoiseFusionNetwork(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=16):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels * 2, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)

        self.gate_conv = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)

        nn.init.constant_(self.gate_conv.bias, -3.0)

    def forward(self, original_noise, updated_noise):

        x = torch.cat([original_noise, updated_noise], dim=1)

        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))

        gate = torch.sigmoid(self.gate_conv(x))
        fused_noise = (1 - gate) * original_noise + gate * updated_noise
        return fused_noise, gate

class SubjectNodeFusionNetwork(nn.Module):
    def __init__(self, hidden_dim=1280, hidden_channels=256):
        super().__init__()

        self.conv1 = nn.Conv1d(hidden_dim * 2, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)

        self.gate_conv = nn.Conv1d(hidden_channels, hidden_dim, kernel_size=3, padding=1)

        nn.init.constant_(self.gate_conv.bias, -3.0)

    def forward(self, original_nodes, updated_nodes):

        x = torch.cat([original_nodes, updated_nodes], dim=-1).transpose(1, 2)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))

        gate = torch.sigmoid(self.gate_conv(x)).transpose(1, 2)

        fused_nodes = (1 - gate) * original_nodes + gate * updated_nodes

        return fused_nodes, gate

class NoiseSubjectCrossAttention(nn.Module):

    def __init__(self, hidden_dim, attention_temperature=0.05, num_heads=8, bidirectional=True, enable_intra_type_attention=True, intra_attention_type="local_window"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_temperature = attention_temperature
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.bidirectional = bidirectional
        self.enable_intra_type_attention = enable_intra_type_attention
        self.intra_attention_type = intra_attention_type

        if self.enable_intra_type_attention and not self.bidirectional:
            self.bidirectional = True

        if self.intra_attention_type == "conv":
            self.noise_intra_conv = nn.Sequential(

                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.GELU(),

                nn.Conv2d(hidden_dim, 9, kernel_size=3, padding=1)
            )

            for m in self.noise_intra_conv.modules():
                if isinstance(m, nn.Conv2d):
                    if m.out_channels == 9:

                        nn.init.zeros_(m.weight)
                        nn.init.constant_(m.bias, 0.0)

                        m.bias.data[4] = 3.0
                    else:
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

        self.q_proj_noise = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj_subj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj_subj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj_noise = nn.Linear(hidden_dim, hidden_dim)

        if self.bidirectional:
            self.q_proj_subj = nn.Linear(hidden_dim, hidden_dim)
            self.k_proj_noise = nn.Linear(hidden_dim, hidden_dim)
            self.v_proj_noise = nn.Linear(hidden_dim, hidden_dim)
            self.out_proj_subj = nn.Linear(hidden_dim, hidden_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, noise_nodes, subject_nodes, return_adj_matrix=False, scale_info=None):
        bsz, num_noise, _ = noise_nodes.shape
        _, num_subject, _ = subject_nodes.shape

        q_noise = self.q_proj_noise(noise_nodes)
        k_subj = self.k_proj_subj(subject_nodes)
        v_subj = self.v_proj_subj(subject_nodes)

        q_noise = q_noise.view(bsz, num_noise, self.num_heads, self.head_dim).transpose(1, 2)
        k_subj = k_subj.view(bsz, num_subject, self.num_heads, self.head_dim).transpose(1, 2)
        v_subj = v_subj.view(bsz, num_subject, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores_noise2subj = torch.matmul(q_noise, k_subj.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=q_noise.dtype))
        attn_scores_noise2subj = attn_scores_noise2subj / self.attention_temperature
        attn_weights_noise2subj = F.softmax(attn_scores_noise2subj, dim=-1)

        attn_output_noise = torch.matmul(attn_weights_noise2subj, v_subj)
        attn_output_noise = attn_output_noise.transpose(1, 2).contiguous().view(bsz, num_noise, self.hidden_dim)

        avg_attn_noise2subj = attn_weights_noise2subj.mean(dim=1)

        if self.bidirectional:

            q_subj = self.q_proj_subj(subject_nodes)
            k_noise = self.k_proj_noise(noise_nodes)
            v_noise = self.v_proj_noise(noise_nodes)

            q_subj = q_subj.view(bsz, num_subject, self.num_heads, self.head_dim).transpose(1, 2)
            k_noise = k_noise.view(bsz, num_noise, self.num_heads, self.head_dim).transpose(1, 2)
            v_noise = v_noise.view(bsz, num_noise, self.num_heads, self.head_dim).transpose(1, 2)

            attn_scores_subj2noise = torch.matmul(q_subj, k_noise.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=q_subj.dtype))
            attn_scores_subj2noise = attn_scores_subj2noise / self.attention_temperature
            attn_weights_subj2noise = F.softmax(attn_scores_subj2noise, dim=-1)

            attn_output_subj = torch.matmul(attn_weights_subj2noise, v_noise)
            attn_output_subj = attn_output_subj.transpose(1, 2).contiguous().view(bsz, num_subject, self.hidden_dim)

            avg_attn_subj2noise = attn_weights_subj2noise.mean(dim=1)
        else:

            attn_output_subj = None
            updated_subject = subject_nodes
            avg_attn_subj2noise = None

        avg_attn_noise2noise = None
        attn_output_noise_intra = None
        if self.enable_intra_type_attention:
            if scale_info is not None:
                num_noise_patches_list, ds_shapes = scale_info

                if self.intra_attention_type == "conv":

                    noise_scale_list = torch.split(noise_nodes, num_noise_patches_list, dim=1)
                    avg_attn_noise2noise_list = []

                    for scale_nodes, (ds_h, ds_w) in zip(noise_scale_list, ds_shapes):
                        bsz, num_patches, hidden_dim = scale_nodes.shape

                        scale_2d = scale_nodes.view(bsz, ds_h, ds_w, hidden_dim).permute(0, 3, 1, 2)

                        conv_weights_2d = self.noise_intra_conv(scale_2d)

                        conv_weights_2d = F.softmax(conv_weights_2d, dim=1)

                        scale_weights = conv_weights_2d.permute(0, 2, 3, 1).flatten(1, 2)

                        y_indices = torch.arange(ds_h, device=scale_nodes.device).view(-1, 1, 1)
                        x_indices = torch.arange(ds_w, device=scale_nodes.device).view(1, -1, 1)
                        dy = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1], device=scale_nodes.device).view(1, 1, -1)
                        dx = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1], device=scale_nodes.device).view(1, 1, -1)

                        y_neighbors = torch.clamp(y_indices + dy, 0, ds_h - 1)
                        x_neighbors = torch.clamp(x_indices + dx, 0, ds_w - 1)
                        neighbor_indices = (y_neighbors * ds_w + x_neighbors).flatten(0, 1)

                        scale_attn_full = torch.zeros(bsz, num_patches, num_patches, device=scale_nodes.device, dtype=scale_weights.dtype)
                        batch_idx = torch.arange(bsz, device=scale_nodes.device).view(-1, 1, 1)
                        patch_idx = torch.arange(num_patches, device=scale_nodes.device).view(1, -1, 1)
                        scale_attn_full[batch_idx, patch_idx, neighbor_indices.unsqueeze(0)] = scale_weights
                        scale_attn_avg = scale_attn_full
                        avg_attn_noise2noise_list.append(scale_attn_avg)

                    avg_attn_noise2noise = torch.zeros(bsz, num_noise, num_noise, device=noise_nodes.device, dtype=noise_nodes.dtype)
                    current_idx = 0
                    for scale_idx, scale_attn in enumerate(avg_attn_noise2noise_list):
                        if scale_attn is None:
                            continue
                        num_patches = num_noise_patches_list[scale_idx]
                        avg_attn_noise2noise[:, current_idx:current_idx+num_patches, current_idx:current_idx+num_patches] = scale_attn
                        current_idx += num_patches

                    attn_output_noise_intra = torch.zeros_like(noise_nodes, device=noise_nodes.device, dtype=noise_nodes.dtype)

                else:

                    noise_scale_list = torch.split(noise_nodes, num_noise_patches_list, dim=1)
                    attn_output_noise_intra_list = []
                    avg_attn_noise2noise_list = []

                    for scale_nodes, (ds_h, ds_w) in zip(noise_scale_list, ds_shapes):
                        bsz, num_patches, hidden_dim = scale_nodes.shape

                        q = self.q_proj_noise(scale_nodes)
                        k = self.k_proj_noise(scale_nodes)
                        v = self.v_proj_noise(scale_nodes)

                        if ds_h <= 32 and ds_w <= 32:

                            q = q.view(bsz, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
                            k = k.view(bsz, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
                            v = v.view(bsz, num_patches, self.num_heads, self.head_dim).transpose(1, 2)

                            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=q.dtype))
                            attn_scores = attn_scores / self.attention_temperature
                            attn_weights = F.softmax(attn_scores, dim=-1)

                            scale_output = torch.matmul(attn_weights, v)
                            scale_output = scale_output.transpose(1, 2).contiguous().view(bsz, num_patches, hidden_dim)

                            scale_attn_avg = attn_weights.mean(dim=1) if return_adj_matrix else None
                        else:

                            if self.intra_attention_type == "local_window":

                                scale_nodes_2d = scale_nodes.view(bsz, ds_h, ds_w, hidden_dim)
                                q_2d = q.view(bsz, ds_h, ds_w, hidden_dim)
                                k_2d = k.view(bsz, ds_h, ds_w, hidden_dim)
                                v_2d = v.view(bsz, ds_h, ds_w, hidden_dim)

                                q_unfold = F.unfold(q_2d.permute(0, 3, 1, 2), kernel_size=3, padding=1).view(bsz, hidden_dim, 9, num_patches)
                                k_unfold = F.unfold(k_2d.permute(0, 3, 1, 2), kernel_size=3, padding=1).view(bsz, hidden_dim, 9, num_patches)
                                v_unfold = F.unfold(v_2d.permute(0, 3, 1, 2), kernel_size=3, padding=1).view(bsz, hidden_dim, 9, num_patches)

                                q_unfold = q_unfold.permute(0, 3, 2, 1)
                                k_unfold = k_unfold.permute(0, 3, 2, 1)
                                v_unfold = v_unfold.permute(0, 3, 2, 1)
                                q_unfold = q_unfold.reshape(bsz, num_patches, 9, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
                                k_unfold = k_unfold.reshape(bsz, num_patches, 9, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
                                v_unfold = v_unfold.reshape(bsz, num_patches, 9, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

                                q_reshaped = q.view(bsz, num_patches, self.num_heads, self.head_dim).permute(0, 2, 1, 3).unsqueeze(3)
                                attn_scores = torch.matmul(q_reshaped, k_unfold.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=q.dtype))
                                attn_scores = attn_scores / self.attention_temperature
                                attn_weights = F.softmax(attn_scores, dim=-1)

                                scale_output = torch.matmul(attn_weights, v_unfold).squeeze(3)
                                scale_output = scale_output.transpose(1, 2).contiguous().view(bsz, num_patches, hidden_dim)

                                scale_attn_avg = None
                                if return_adj_matrix:

                                    scale_attn_avg = attn_weights.mean(dim=1).squeeze(2)

                                    y_indices = torch.arange(ds_h, device=scale_nodes.device).view(-1, 1, 1)
                                    x_indices = torch.arange(ds_w, device=scale_nodes.device).view(1, -1, 1)
                                    dy = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1], device=scale_nodes.device).view(1, 1, -1)
                                    dx = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1], device=scale_nodes.device).view(1, 1, -1)

                                    y_neighbors = torch.clamp(y_indices + dy, 0, ds_h - 1)
                                    x_neighbors = torch.clamp(x_indices + dx, 0, ds_w - 1)
                                    neighbor_indices = (y_neighbors * ds_w + x_neighbors).flatten(0, 1)

                                    scale_attn_full = torch.zeros(bsz, num_patches, num_patches, device=scale_nodes.device, dtype=scale_attn_avg.dtype)
                                    batch_idx = torch.arange(bsz, device=scale_nodes.device).view(-1, 1, 1)
                                    patch_idx = torch.arange(num_patches, device=scale_nodes.device).view(1, -1, 1)
                                    scale_attn_full[batch_idx, patch_idx, neighbor_indices.unsqueeze(0)] = scale_attn_avg
                                    scale_attn_avg = scale_attn_full

                            elif self.intra_attention_type == "axial":

                                q_2d = q.view(bsz, ds_h, ds_w, hidden_dim)
                                k_2d = k.view(bsz, ds_h, ds_w, hidden_dim)
                                v_2d = v.view(bsz, ds_h, ds_w, hidden_dim)

                                q_row = q_2d.reshape(bsz * ds_h, ds_w, hidden_dim)
                                k_row = k_2d.reshape(bsz * ds_h, ds_w, hidden_dim)
                                v_row = v_2d.reshape(bsz * ds_h, ds_w, hidden_dim)

                                q_row = q_row.view(bsz * ds_h, ds_w, self.num_heads, self.head_dim).transpose(1, 2)
                                k_row = k_row.view(bsz * ds_h, ds_w, self.num_heads, self.head_dim).transpose(1, 2)
                                v_row = v_row.view(bsz * ds_h, ds_w, self.num_heads, self.head_dim).transpose(1, 2)

                                attn_scores_row = torch.matmul(q_row, k_row.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=q.dtype))
                                attn_scores_row = attn_scores_row / self.attention_temperature
                                attn_weights_row = F.softmax(attn_scores_row, dim=-1)
                                row_output = torch.matmul(attn_weights_row, v_row)
                                row_output = row_output.transpose(1, 2).contiguous().view(bsz, ds_h, ds_w, hidden_dim)

                                q_col = q_2d.permute(0, 2, 1, 3).reshape(bsz * ds_w, ds_h, hidden_dim)
                                k_col = k_2d.permute(0, 2, 1, 3).reshape(bsz * ds_w, ds_h, hidden_dim)
                                v_col = v_2d.permute(0, 2, 1, 3).reshape(bsz * ds_w, ds_h, hidden_dim)

                                q_col = q_col.view(bsz * ds_w, ds_h, self.num_heads, self.head_dim).transpose(1, 2)
                                k_col = k_col.view(bsz * ds_w, ds_h, self.num_heads, self.head_dim).transpose(1, 2)
                                v_col = v_col.view(bsz * ds_w, ds_h, self.num_heads, self.head_dim).transpose(1, 2)

                                attn_scores_col = torch.matmul(q_col, k_col.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=q.dtype))
                                attn_scores_col = attn_scores_col / self.attention_temperature
                                attn_weights_col = F.softmax(attn_scores_col, dim=-1)
                                col_output = torch.matmul(attn_weights_col, v_col)
                                col_output = col_output.transpose(1, 2).contiguous().view(bsz, ds_w, ds_h, hidden_dim).permute(0, 2, 1, 3)

                                scale_output = (row_output + col_output).flatten(1, 2)

                                scale_attn_avg = None
                                if return_adj_matrix:

                                    attn_weights_row_avg = attn_weights_row.mean(dim=1).view(bsz, ds_h, ds_w, ds_w)
                                    attn_weights_col_avg = attn_weights_col.mean(dim=1).view(bsz, ds_w, ds_h, ds_h).permute(0, 2, 1, 3)

                                    scale_attn_full = torch.zeros(bsz, num_patches, num_patches, device=scale_nodes.device, dtype=q.dtype)
                                    for b in range(bsz):
                                        for y in range(ds_h):
                                            for x in range(ds_w):
                                                idx = y * ds_w + x

                                                row_weights = attn_weights_row_avg[b, y, x].to(scale_attn_full.dtype)
                                                scale_attn_full[b, idx, y*ds_w : (y+1)*ds_w] = row_weights

                                                col_indices = torch.arange(ds_h, device=scale_nodes.device) * ds_w + x
                                                col_weights = attn_weights_col_avg[b, y, x].to(scale_attn_full.dtype)
                                                scale_attn_full[b, idx, col_indices] = torch.maximum(
                                                    scale_attn_full[b, idx, col_indices],
                                                    col_weights
                                                )
                                    scale_attn_avg = scale_attn_full

                        attn_output_noise_intra_list.append(scale_output)
                        avg_attn_noise2noise_list.append(scale_attn_avg)

                    if self.intra_attention_type != "conv":

                        attn_output_noise_intra = torch.cat(attn_output_noise_intra_list, dim=1)

                        if return_adj_matrix:
                            avg_attn_noise2noise = torch.zeros(bsz, num_noise, num_noise, device=noise_nodes.device, dtype=noise_nodes.dtype)
                            current_idx = 0
                            for scale_idx, scale_attn in enumerate(avg_attn_noise2noise_list):
                                if scale_attn is None:
                                    continue
                                num_patches = num_noise_patches_list[scale_idx]
                                avg_attn_noise2noise[:, current_idx:current_idx+num_patches, current_idx:current_idx+num_patches] = scale_attn
                                current_idx += num_patches
            else:

                    q_noise_intra = self.q_proj_noise(noise_nodes)
                    k_noise_intra = self.k_proj_noise(noise_nodes)
                    v_noise_intra = self.v_proj_noise(noise_nodes)

                    q_noise_intra = q_noise_intra.view(bsz, num_noise, self.num_heads, self.head_dim).transpose(1, 2)
                    k_noise_intra = k_noise_intra.view(bsz, num_noise, self.num_heads, self.head_dim).transpose(1, 2)
                    v_noise_intra = v_noise_intra.view(bsz, num_noise, self.num_heads, self.head_dim).transpose(1, 2)

                    attn_scores_noise2noise = torch.matmul(q_noise_intra, k_noise_intra.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=q_noise_intra.dtype))
                    attn_scores_noise2noise = attn_scores_noise2noise / self.attention_temperature
                    attn_weights_noise2noise = F.softmax(attn_scores_noise2noise, dim=-1)

                    attn_output_noise_intra = torch.matmul(attn_weights_noise2noise, v_noise_intra)
                    attn_output_noise_intra = attn_output_noise_intra.transpose(1, 2).contiguous().view(bsz, num_noise, self.hidden_dim)

                    avg_attn_noise2noise = attn_weights_noise2noise.mean(dim=1)

        avg_attn_subj2subj = None
        attn_output_subj_intra = None
        if self.enable_intra_type_attention:

            q_subj_intra = self.q_proj_subj(subject_nodes)
            k_subj_intra = self.k_proj_subj(subject_nodes)
            v_subj_intra = self.v_proj_subj(subject_nodes)

            q_subj_intra = q_subj_intra.view(bsz, num_subject, self.num_heads, self.head_dim).transpose(1, 2)
            k_subj_intra = k_subj_intra.view(bsz, num_subject, self.num_heads, self.head_dim).transpose(1, 2)
            v_subj_intra = v_subj_intra.view(bsz, num_subject, self.num_heads, self.head_dim).transpose(1, 2)

            attn_scores_subj2subj = torch.matmul(q_subj_intra, k_subj_intra.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=q_subj_intra.dtype))
            attn_scores_subj2subj = attn_scores_subj2subj / self.attention_temperature
            attn_weights_subj2subj = F.softmax(attn_scores_subj2subj, dim=-1)

            attn_output_subj_intra = torch.matmul(attn_weights_subj2subj, v_subj_intra)
            attn_output_subj_intra = attn_output_subj_intra.transpose(1, 2).contiguous().view(bsz, num_subject, self.hidden_dim)

            avg_attn_subj2subj = attn_weights_subj2subj.mean(dim=1)

        if self.enable_intra_type_attention:
            attn_output_noise = attn_output_noise + attn_output_noise_intra
        updated_noise = self.out_proj_noise(attn_output_noise) + noise_nodes

        if self.bidirectional:
            if self.enable_intra_type_attention:
                attn_output_subj = attn_output_subj + attn_output_subj_intra
            updated_subject = self.out_proj_subj(attn_output_subj) + subject_nodes

        if return_adj_matrix:
            total_nodes = num_noise + num_subject
            adj_matrix = torch.zeros(bsz, total_nodes, total_nodes, device=noise_nodes.device, dtype=noise_nodes.dtype)

            if self.bidirectional:

                combined_attn = (avg_attn_noise2subj + avg_attn_subj2noise.transpose(1, 2)) / 2

                adj_matrix[:, :num_noise, num_noise:] = combined_attn
                adj_matrix[:, num_noise:, :num_noise] = combined_attn.transpose(1, 2)
            else:

                adj_matrix[:, :num_noise, num_noise:] = avg_attn_noise2subj
                adj_matrix[:, num_noise:, :num_noise] = avg_attn_noise2subj.transpose(1, 2)

            if self.enable_intra_type_attention:

                if avg_attn_noise2noise is not None:
                    adj_matrix[:, :num_noise, :num_noise] = avg_attn_noise2noise

                if avg_attn_subj2subj is not None:
                    adj_matrix[:, num_noise:, num_noise:] = avg_attn_subj2subj

            adj_matrix = adj_matrix + torch.eye(total_nodes, device=adj_matrix.device).unsqueeze(0)
            return adj_matrix

        else:
            attn_weights = {
                "noise_to_subject": avg_attn_noise2subj,
                "subject_to_noise": avg_attn_subj2noise
            }
            return updated_noise, updated_subject, attn_weights

class FineGrainedNoiseFusionNetwork(nn.Module):
    def __init__(self, in_channels=4, num_scales=4, hidden_channels=32):
        super().__init__()
        self.num_scales = num_scales

        self.conv_in = nn.Conv2d(in_channels * num_scales, hidden_channels, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, hidden_channels // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 4, hidden_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv_out = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, noise_list):

        x = torch.cat(noise_list, dim=1)

        x = F.gelu(self.conv_in(x))

        residual = x
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = self.conv3(x)

        attn = self.channel_attention(x)
        x = x * attn

        x = x + residual
        x = F.gelu(x)

        fused_noise = self.conv_out(x)

        return fused_noise

class ImageEncoder(ModelMixin, ConfigMixin):
    def __init__(
        self,
        clip_model_name_or_path,
        dim,
        depth,
        dim_head,
        heads,
        num_queries,
        output_dim,
        ff_mult,
        latent_init_mode,
        phrase_embeddings_dim,
        num_dummy_tokens=4,
        cross_attention_dim=2048,

        gcn_num_layers: int = 2,
        gcn_hidden_dim: int = None,
        max_supported_subjects: int = 4,
        adj_mode: str = "dynamic_cosine",

        adj_temperature: float = 0.1,
        skip_connection: bool = True,

        noise_gcn_num_layers: int = 2,
        noise_gcn_hidden_dim: int = None,
        noise_adj_temperature: float = 0.05,
        noise_proj_bottleneck_dim: int = None,
        use_cross_attention: bool = False,
        cross_attention_heads: int = 8,
        bidirectional_attention: bool = True,
        enable_intra_type_attention: bool = True,
        intra_attention_type: str = "local_window",
    ):
        super().__init__()

        self.clip_model = CLIPVisionModelWithProjection.from_pretrained(
            clip_model_name_or_path
        )
        self.hidden_size = self.clip_model.config.hidden_size
        self.gcn_hidden_dim = gcn_hidden_dim or self.hidden_size
        self.adj_mode = adj_mode
        self.adj_temperature = adj_temperature
        self.skip_connection = skip_connection
        self.max_tokens_per_subject = 257
        self.max_total_nodes = max_supported_subjects * self.max_tokens_per_subject

        self.noise_gcn_num_layers = noise_gcn_num_layers
        self.noise_gcn_hidden_dim = noise_gcn_hidden_dim or self.hidden_size
        self.noise_adj_temperature = noise_adj_temperature
        self.use_cross_attention = use_cross_attention
        self.cross_attention_heads = cross_attention_heads
        self.bidirectional_attention = bidirectional_attention
        self.enable_intra_type_attention = enable_intra_type_attention
        self.intra_attention_type = intra_attention_type

        if self.use_cross_attention:
            self.cross_attention = NoiseSubjectCrossAttention(
                hidden_dim=self.hidden_size,
                attention_temperature=self.noise_adj_temperature,
                num_heads=self.cross_attention_heads,
                bidirectional=self.bidirectional_attention,
                enable_intra_type_attention=self.enable_intra_type_attention,
                intra_attention_type=self.intra_attention_type
            )

        self.resampler = Resampler(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            num_queries=num_queries,
            embedding_dim=self.hidden_size,
            output_dim=output_dim,
            ff_mult=ff_mult,
            latent_init_mode=latent_init_mode,
            phrase_embeddings_dim=phrase_embeddings_dim
        )

        self.dummy_image_tokens = nn.Parameter(torch.randn(1, num_dummy_tokens, cross_attention_dim))

        if adj_mode == "learnable":
            self.learnable_adj = nn.Parameter(torch.ones(self.max_total_nodes, self.max_total_nodes) * 0.5)

        self.gcn_layers = nn.ModuleList()

        self.gcn_layers.append(GCNLayer(self.hidden_size, self.gcn_hidden_dim))

        for _ in range(gcn_num_layers - 2):
            self.gcn_layers.append(GCNLayer(self.gcn_hidden_dim, self.gcn_hidden_dim))

        if gcn_num_layers >= 1:
            self.gcn_layers.append(GCNLayer(self.gcn_hidden_dim, self.hidden_size))

        self.output_norm = nn.LayerNorm(self.hidden_size) if skip_connection else nn.Identity()

        self.noise_proj_bottleneck_dim = noise_proj_bottleneck_dim or (self.hidden_size // 4)
        self.noise_proj = nn.Sequential(
            nn.Linear(4, self.noise_proj_bottleneck_dim),
            nn.GELU(),
            nn.LayerNorm(self.noise_proj_bottleneck_dim),
            nn.Linear(self.noise_proj_bottleneck_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

        self.noise_gcn_layers = nn.ModuleList()

        self.noise_gcn_layers.append(GCNLayer(self.hidden_size, self.noise_gcn_hidden_dim))

        for _ in range(noise_gcn_num_layers - 2):
            self.noise_gcn_layers.append(GCNLayer(self.noise_gcn_hidden_dim, self.noise_gcn_hidden_dim))

        if noise_gcn_num_layers >= 1:
            self.noise_gcn_layers.append(GCNLayer(self.noise_gcn_hidden_dim, self.hidden_size))

        self.noise_output_norm = nn.LayerNorm(self.hidden_size) if skip_connection else nn.Identity()

        self.noise_reproj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, 4)
        )

        self.noise_fusion_net = NoiseFusionNetwork(in_channels=4, hidden_channels=16)

        self.fine_grained_fusion_net = FineGrainedNoiseFusionNetwork(
            in_channels=4,
            num_scales=4,
            hidden_channels=32
        )

        self.subject_fusion_net = SubjectNodeFusionNetwork(
            hidden_dim=self.hidden_size,
            hidden_channels=self.hidden_size // 4
        )

    def forward(self, concept_images, concept_images_896=None, grounding_kwargs=None, return_intermediate=False, noisy_model_input=None, run_full_pipeline=False):

        bsz, n, c, h, w = concept_images.shape
        concept_images = concept_images.reshape(-1, c, h, w)
        total_nodes = n * self.max_tokens_per_subject

        concept_embeddings = self.clip_model(concept_images, output_hidden_states=True).hidden_states[-2]

        original_embeddings = concept_embeddings.clone()

        cls_tokens = concept_embeddings[:, 0, :].view(bsz, n, self.hidden_size)

        node_features = concept_embeddings.view(bsz, total_nodes, self.hidden_size)

        if self.adj_mode == "dynamic_cosine":

            normed_nodes = F.normalize(node_features, p=2, dim=-1)

            adj = torch.bmm(normed_nodes, normed_nodes.transpose(1, 2))

            adj = F.softmax(adj / self.adj_temperature, dim=-1)
            adj = adj.to(node_features.dtype)
        elif self.adj_mode == "learnable":

            adj = self.learnable_adj[:total_nodes, :total_nodes]

            adj = torch.sigmoid(adj)

            adj = adj.unsqueeze(0).repeat(bsz, 1, 1)
            adj = adj.to(node_features.dtype)
        else:
            raise ValueError(f"Unsupported adj_mode: {self.adj_mode}")

        adj = adj + torch.eye(total_nodes, device=adj.device).unsqueeze(0)
        deg = torch.diag_embed(torch.sum(adj, dim=-1) ** (-0.5))
        adj_norm = torch.bmm(torch.bmm(deg, adj), deg)

        x = node_features
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, adj_norm)
            if i != len(self.gcn_layers) - 1:
                x = F.gelu(x)

        if self.skip_connection:
            x = x + node_features
        updated_node_features = self.output_norm(x)

        updated_cls_tokens = updated_node_features.view(bsz, n, self.max_tokens_per_subject, self.hidden_size)[:, :, 0, :]

        first_gcn_updated_cls = updated_cls_tokens
        first_gcn_updated_all_tokens = updated_node_features
        num_subjects = n

        if return_intermediate:
            return {
                "original_cls": cls_tokens,
                "first_gcn_updated_cls": first_gcn_updated_cls,
                "first_gcn_updated_all_tokens": first_gcn_updated_all_tokens,
                "num_subjects": num_subjects,
                "tokens_per_subject": self.max_tokens_per_subject
            }

        if run_full_pipeline and noisy_model_input is not None:

            updated_noisy_input, updated_subject_nodes, gate = self.noise_cls_gcn_forward(
                noisy_model_input,
                first_gcn_updated_all_tokens
            )

            fused_subject_nodes, subject_gate = self.subject_fusion_net(
                first_gcn_updated_all_tokens,
                updated_subject_nodes
            )

            image_prompt_embeds = self.resample_with_updated_cls(
                fused_subject_nodes,
                num_subjects,
                grounding_kwargs
            )
            return updated_noisy_input, image_prompt_embeds, gate

        concept_embeddings = first_gcn_updated_all_tokens.view(bsz * num_subjects, self.max_tokens_per_subject, self.hidden_size)

        image_prompt_embeds = self.resampler(concept_embeddings, grounding_kwargs)
        image_prompt_embeds = image_prompt_embeds.view(bsz, -1, image_prompt_embeds.shape[-2], image_prompt_embeds.shape[-1])
        image_prompt_embeds = image_prompt_embeds.view(bsz, image_prompt_embeds.shape[-3] * image_prompt_embeds.shape[-2],
                                                        image_prompt_embeds.shape[-1])
        dummy_image_tokens = self.dummy_image_tokens.repeat(bsz, 1, 1)
        image_prompt_embeds = torch.cat([dummy_image_tokens, image_prompt_embeds], dim=1)

        return image_prompt_embeds

    def resample_with_updated_cls(self, fused_subject_nodes, num_subjects, grounding_kwargs=None):

        bsz = fused_subject_nodes.shape[0]
        tokens_per_subject = self.max_tokens_per_subject

        concept_embeddings = fused_subject_nodes.view(bsz * num_subjects, tokens_per_subject, self.hidden_size)

        resampler_dtype = self.resampler.proj_in.weight.dtype
        concept_embeddings = concept_embeddings.to(dtype=resampler_dtype)
        if grounding_kwargs is not None:
            for k, v in grounding_kwargs.items():
                if isinstance(v, torch.Tensor):
                    grounding_kwargs[k] = v.to(dtype=resampler_dtype)

        image_prompt_embeds = self.resampler(concept_embeddings, grounding_kwargs)
        image_prompt_embeds = image_prompt_embeds.view(bsz, -1, image_prompt_embeds.shape[-2], image_prompt_embeds.shape[-1])
        image_prompt_embeds = image_prompt_embeds.view(bsz, image_prompt_embeds.shape[-3] * image_prompt_embeds.shape[-2],
                                                        image_prompt_embeds.shape[-1])
        dummy_image_tokens = self.dummy_image_tokens.repeat(bsz, 1, 1)
        image_prompt_embeds = torch.cat([dummy_image_tokens, image_prompt_embeds], dim=1)

        return image_prompt_embeds

    def noise_cls_gcn_forward(self, noise, all_subject_tokens):

        bsz, noise_channels, noise_h, noise_w = noise.shape
        total_subject_nodes = all_subject_tokens.shape[1]

        original_noise_dtype = noise.dtype
        target_dtype = self.noise_proj[0].weight.dtype
        noise = noise.to(dtype=target_dtype)
        all_subject_tokens = all_subject_tokens.to(dtype=target_dtype)
        original_noise = noise.clone()

        scale_factors = [0.57, 0.5, 0.25, 0.125]
        downsampled_noises = []
        num_noise_patches_list = []
        ds_shapes = []

        for scale in scale_factors:
            ds_noise = F.interpolate(noise, scale_factor=scale, mode='bilinear', align_corners=False)
            _, ds_c, ds_h, ds_w = ds_noise.shape
            num_patches = ds_h * ds_w
            downsampled_noises.append(ds_noise)
            num_noise_patches_list.append(num_patches)
            ds_shapes.append((ds_h, ds_w))

        total_noise_nodes = sum(num_noise_patches_list)

        noise_proj_list = []
        for ds_noise, num_patches in zip(downsampled_noises, num_noise_patches_list):

            ds_noise_patches = ds_noise.permute(0, 2, 3, 1).reshape(bsz, num_patches, noise_channels)

            ds_noise_proj = self.noise_proj(ds_noise_patches)
            noise_proj_list.append(ds_noise_proj)

        all_noise_proj = torch.cat(noise_proj_list, dim=1)

        joint_nodes = torch.cat([all_noise_proj, all_subject_tokens], dim=1)
        total_joint_nodes = joint_nodes.shape[1]

        if self.use_cross_attention:

            adj = self.cross_attention(
                all_noise_proj,
                all_subject_tokens,
                return_adj_matrix=True,
                scale_info=(num_noise_patches_list, ds_shapes)
            )
        else:

            normed_nodes = F.normalize(joint_nodes, p=2, dim=-1)
            adj = torch.bmm(normed_nodes, normed_nodes.transpose(1, 2))
            adj = F.softmax(adj / self.noise_adj_temperature, dim=-1)
            adj = adj.to(joint_nodes.dtype)

        adj = adj + torch.eye(total_joint_nodes, device=adj.device).unsqueeze(0)
        deg = torch.diag_embed(torch.sum(adj, dim=-1) ** (-0.5))
        adj_norm = torch.bmm(torch.bmm(deg, adj), deg)

        x = joint_nodes
        for i, gcn_layer in enumerate(self.noise_gcn_layers):
            x = gcn_layer(x, adj_norm)
            if i != len(self.noise_gcn_layers) - 1:
                x = F.gelu(x)

        if self.skip_connection:
            x = x + joint_nodes
        if hasattr(self.noise_output_norm, 'weight'):
            x = x.to(dtype=self.noise_output_norm.weight.dtype)
        updated_joint_nodes = self.noise_output_norm(x)

        updated_noise_proj = updated_joint_nodes[:, :total_noise_nodes, :]
        updated_subject_nodes = updated_joint_nodes[:, total_noise_nodes:, :]

        updated_noise_proj_list = []
        current_idx = 0
        for num_patches in num_noise_patches_list:
            scale_proj = updated_noise_proj[:, current_idx:current_idx + num_patches, :]
            updated_noise_proj_list.append(scale_proj)
            current_idx += num_patches

        updated_noise_list = []
        for i, (scale_proj, (ds_h, ds_w)) in enumerate(zip(updated_noise_proj_list, ds_shapes)):

            ds_updated_noise_patches = self.noise_reproj(scale_proj)

            ds_updated_noise = ds_updated_noise_patches.view(bsz, ds_h, ds_w, 4).permute(0, 3, 1, 2)

            upsampled_noise = F.interpolate(ds_updated_noise, size=(noise_h, noise_w), mode='bilinear', align_corners=False)
            updated_noise_list.append(upsampled_noise)

        fine_grained_fused_noise = self.fine_grained_fusion_net(updated_noise_list)

        final_fused_noise, gate = self.noise_fusion_net(original_noise, fine_grained_fused_noise)

        final_fused_noise = final_fused_noise.to(dtype=original_noise_dtype)
        updated_subject_nodes = updated_subject_nodes.to(dtype=original_noise_dtype)
        gate = gate.to(dtype=original_noise_dtype)

        return final_fused_noise, updated_subject_nodes, gate
