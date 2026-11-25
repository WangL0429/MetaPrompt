# clip/modules/missing_aware_prompt.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MissingAwarePrompt(nn.Module):

    def __init__(
        self,
        prompt_dim: int,
        num_missing_types: int = 3,   # 0: 完整, 1: 文本缺, 2: 图像缺
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_missing_types = num_missing_types
        cond_dim = num_missing_types + 2 + 1  # 6

        self.mlp_delta = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prompt_dim),
        )

        self.mlp_gate = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        P_mod: torch.Tensor,      
        missing_type,             
        vis_ratio_img,            
        vis_ratio_txt,            
        missing_ratio_sample,    
    ) -> torch.Tensor:
        device = P_mod.device
        L, D = P_mod.shape

        # ---- 1. one-hot(missing_type) ----
        mt_onehot = torch.zeros(self.num_missing_types, device=device)
        if torch.is_tensor(missing_type):
            mt_idx = int(missing_type.item())
        else:
            mt_idx = int(missing_type)
        if 0 <= mt_idx < self.num_missing_types:
            mt_onehot[mt_idx] = 1.0

        # ---- 2. 其它标量特征 ----
        def _to_scalar_tensor(x):
            if torch.is_tensor(x):
                return torch.tensor(float(x.item()), device=device).view(1)
            else:
                return torch.tensor(float(x), device=device).view(1)

        vr_img = _to_scalar_tensor(vis_ratio_img)        # [1]
        vr_txt = _to_scalar_tensor(vis_ratio_txt)        # [1]
        mrs = _to_scalar_tensor(missing_ratio_sample)    # [1]

        cond = torch.cat([mt_onehot, vr_img, vr_txt, mrs], dim=0)  # [cond_dim]

        # ---- 3. 生成 delta 和 gate ----
        delta = self.mlp_delta(cond)          # [D]
        gate = torch.sigmoid(self.mlp_gate(cond))[0]  # 标量

        delta = delta.unsqueeze(0).expand(L, D)       # [L, D]
        P_out = P_mod + gate * delta
        return P_out
