import unittest
from types import SimpleNamespace
import torch
from clip.modules.clip_meta_prompt_module import MultiModalPromptLearner
from clip.modules.vision_transformer_prompts import ResidualAttentionBlock


class SharedDeepTest(unittest.TestCase):
    def test_both_modalities_use_the_same_residual_and_old_weights_load(self):
        args=(4,3,SimpleNamespace(dtype=torch.float32))
        old=MultiModalPromptLearner(*args)
        new=MultiModalPromptLearner(*args,shared_deep_prompt=True)
        new.load_state_dict(old.state_dict(),strict=True)
        states=[0,1,2]
        base=old(states)
        changed=new(states)
        for original,actual in zip(base,changed):
            torch.testing.assert_close(actual[0],original[0])
            self.assertFalse(actual[1].any())
        with torch.no_grad():
            new.shared_deep_residual.normal_(std=.1)
        image,text=new(states)
        for out,projection in [(image,new.common_prompt_projection_image),
                               (text,new.common_prompt_projection_text)]:
            for i,p in enumerate([new.common_prompt_complete,new.common_prompt_image,new.common_prompt_text]):
                expected=(projection(p+new.shared_deep_residual)-projection(p))/2
                torch.testing.assert_close(out[1][i],expected)
                torch.testing.assert_close(out[2][i],expected)
            g,=torch.autograd.grad(out[1].sum(),new.shared_deep_residual,retain_graph=True)
            self.assertGreater(g.norm().item(),0)
        saved=new.state_dict()
        del saved['shared_deep_residual']
        with self.assertRaises(RuntimeError):
            new.load_state_dict(saved,strict=True)

    def test_deeper_block_adds_to_static_positions_only(self):
        block=ResidualAttentionBlock(16,2,prompt_length=4,i=1,prompt_depth=3)
        block.use_shared_deep_prompt=True
        x=torch.randn(11,2,16)
        update=torch.randn(2,2,16)
        captured=[]
        hook=block.ln_1.register_forward_pre_hook(lambda module,args:captured.append(args[0]))
        try:
            y=block([x,[update]*3,1,[0,1]])[0]
        finally:
            hook.remove()
        self.assertEqual(y.shape,x.shape)
        torch.testing.assert_close(captured[0][:2],x[:2]+update.permute(1,0,2))
        torch.testing.assert_close(captured[0][2:],x[2:])

