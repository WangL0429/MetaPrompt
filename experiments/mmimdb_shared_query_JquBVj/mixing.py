import torch

def mix_query(query, static_prompt, matrix):
    # Fold in parameter precision even when the encoder itself uses autocast.
    with torch.autocast(query.device.type, enabled=False):
        rms_ratio=(query.square().mean()/static_prompt.square().mean().clamp_min(1e-12)).sqrt()
        return query+(matrix@(static_prompt*rms_ratio)).unsqueeze(1)
