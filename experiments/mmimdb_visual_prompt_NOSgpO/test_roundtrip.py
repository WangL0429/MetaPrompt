import torch
from clip.modules.vision_transformer_prompts import ResidualAttentionBlock
m=ResidualAttentionBlock(16,1,prompt_length=4)
assert "prompts_dynamic_image" in dict(m.named_parameters()), "missing trainable visual prompt"
t=ResidualAttentionBlock(16,1,prompt_length=4,text_layer=True)
assert not any("prompts_dynamic_" in n for n,_ in t.named_parameters())
saved=m.state_dict()
assert all(n in saved for n in m.dynamic_names)
clone=ResidualAttentionBlock(16,1,prompt_length=4)
clone.load_state_dict(saved,strict=True)
for n in m.dynamic_names:
    assert torch.equal(getattr(m,n),getattr(clone,n))
old={k:v for k,v in saved.items() if k not in m.dynamic_names}
clone.load_state_dict(old,strict=True)
del saved[m.dynamic_names[0]]
try:
    clone.load_state_dict(saved,strict=True)
except RuntimeError:
    pass
else:
    raise AssertionError("A damaged v2 checkpoint passed strict loading")
print("Verified visual parameter, text buffers, v1 compatibility and v2 strict round-trip")
