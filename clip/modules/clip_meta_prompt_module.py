from contextlib import nullcontext
import torch
import torch.nn as nn
import pytorch_lightning as pl
import clip.modules.vision_transformer_prompts as vit
import math
import torch.nn.functional as F 
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from clip.modules import clip_utils, heads, objectives, clip
import copy
from torch.func import functional_call

def load_clip_to_cpu(backbone_name, prompt_length, prompt_depth):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu")#.eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = vit.build_model(state_dict or model.state_dict(), prompt_length, prompt_depth)

    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.prompt_length = clip_model.prompt_length

    def forward(self, tokenized_texts, all_prompts_text, missing_type):
        x = self.token_embedding(tokenized_texts).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, all_prompts_text, 0, missing_type]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0][self.prompt_length:]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # B, L_text, C = x.shape
        # eot_idx = torch.full((B,), L_text - 1, dtype=torch.long, device=x.device)  # 全是 L_text-1
        # x = x[torch.arange(B, device=x.device), eot_idx] @ self.text_projection
        x = x[torch.arange(x.shape[0]), tokenized_texts.argmax(dim=-1)] @ self.text_projection

        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MultiModalPromptLearner(nn.Module):
    def __init__(self, prompt_length, prompt_depth, clip_model, shared_deep_prompt=False):
        super().__init__()
        dtype = clip_model.dtype
        prompt_length_half = prompt_length//2 # use half length for generating static prompts, and the other for generating dynamic prompts
        # Default is 1, which is compound shallow prompting
        self.prompt_depth = prompt_depth  # max=12, but will create 11 such shared prompts
        self.common_prompt_complete = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        self.common_prompt_image = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        self.common_prompt_text = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        # Also make corresponding projection layers, for each prompt
        embed_dim_text = 512
        embed_dim_image = 768
        r = 16
        self.common_prompt_projection_image = nn.Sequential(
                nn.Linear(embed_dim_text, embed_dim_text//r),
                nn.GELU(),
                # nn.Dropout(0.1),
                nn.Linear(embed_dim_text//r, embed_dim_image),
                )
        self.common_prompt_projection_text = nn.Sequential(
                nn.Linear(embed_dim_text, embed_dim_text//r),
                nn.GELU(),
                # nn.Dropout(0.1),
                nn.Linear(embed_dim_text//r, embed_dim_text),
                )

        self._version = 2 if shared_deep_prompt else 1
        if shared_deep_prompt and prompt_depth < 2:
            raise ValueError("Shared deep prompts require at least two injection layers")
        self.register_parameter("shared_deep_residual", nn.Parameter(torch.zeros_like(
            self.common_prompt_complete)) if shared_deep_prompt else None)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if self.shared_deep_residual is not None and local_metadata.get("version", 0) < 2:
            state_dict.setdefault(prefix + "shared_deep_residual", torch.zeros_like(self.shared_deep_residual))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def _shared_deep_prompts(self, missing_type):
        states = missing_type.tolist() if torch.is_tensor(missing_type) else list(missing_type)
        banks = (self.common_prompt_complete, self.common_prompt_image, self.common_prompt_text)
        unique = {state: banks[state] for state in dict.fromkeys(states)}

        def project(projection):
            first = {state: projection(prompt) for state, prompt in unique.items()}
            residuals = {state: (projection(prompt + self.shared_deep_residual) - first[state])
                                / (self.prompt_depth - 1) for state, prompt in unique.items()}
            shallow = torch.stack([first[state] for state in states])
            deep = torch.stack([residuals[state] for state in states])
            return [shallow] + [deep] * (self.prompt_depth - 1)

        return project(self.common_prompt_projection_image), project(self.common_prompt_projection_text)

    def forward(self, missing_type):
        if self.shared_deep_residual is not None:
            return self._shared_deep_prompts(missing_type)

        all_prompts_image = [[] for _ in range(self.prompt_depth)]
        all_prompts_text  = [[] for _ in range(self.prompt_depth)]

        for i in range(len(missing_type)):
            mt = int(missing_type[i].item()) if torch.is_tensor(missing_type) else int(missing_type[i])
            if mt == 0:   common_prompt = self.common_prompt_complete
            elif mt == 1: common_prompt = self.common_prompt_image
            elif mt == 2: common_prompt = self.common_prompt_text
            else: raise ValueError(mt)

            img_common = self.common_prompt_projection_image(common_prompt)  # [L, 768]
            txt_common = self.common_prompt_projection_text(common_prompt)   # [L, 512]

            for d in range(self.prompt_depth):
                all_prompts_image[d].append(img_common)
                all_prompts_text[d].append(txt_common)

        all_prompts_image = [torch.stack(layer, dim=0) for layer in all_prompts_image]  # [B,L,768]
        all_prompts_text  = [torch.stack(layer, dim=0) for layer in all_prompts_text]   # [B,L,512]

        
        return all_prompts_image, all_prompts_text   
  
class CustomCLIP(nn.Module):
    def __init__(self, prompt_length, prompt_depth, clip_model, shared_deep_prompt=False):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(prompt_length, prompt_depth, clip_model, shared_deep_prompt)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        if shared_deep_prompt:
            for encoder in (self.image_encoder, self.text_encoder):
                blocks = encoder.transformer.resblocks
                if prompt_depth > len(blocks):
                    raise ValueError("Prompt depth exceeds encoder depth")
                for block in list(blocks)[1:prompt_depth]:
                    block.use_shared_deep_prompt = True
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, text, missing_type):
        tokenized_texts = torch.stack([clip.tokenize(tx, context_length=77, truncate=True) for tx in text[0]], 0).to(image.get_device()).squeeze(1)  # extract texts from the first key  # [b, 77]
        #logit_scale = self.logit_scale.exp()

        all_prompts_image, all_prompts_text = self.prompt_learner(missing_type)
        text_features = self.text_encoder(tokenized_texts, all_prompts_text, missing_type)
        image_features = self.image_encoder(image.type(self.dtype), all_prompts_image, missing_type)
        return torch.cat([image_features, text_features], -1)

class CLIPransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        clip_model = load_clip_to_cpu(config['vit'], config['prompt_length'], config['prompt_depth'])

        print("Building custom CLIP")
        hidden_size = 512*2
        self.model = CustomCLIP(config['prompt_length'], config['prompt_depth'], clip_model,
                                config.get('shared_deep_prompt', False))

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
            and not self.hparams.config["finetune_first"]
        ):

            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.model.load_state_dict(state_dict, strict=False)

        if self.hparams.config["loss_names"]["hatememes"] > 0:
            cls_num = self.hparams.config["hatememes_class_num"]
            self.hatememes_classifier = nn.Linear(hidden_size, cls_num)
            self.hatememes_classifier.apply(objectives.init_weights)

            # feat_dim = hidden_size // 2
            
            # self.hatememes_text_head = nn.Linear(feat_dim, cls_num)
            # self.hatememes_image_head = nn.Linear(feat_dim, cls_num)

            # self.hatememes_text_head.apply(objectives.init_weights)
            # self.hatememes_image_head.apply(objectives.init_weights)  
            
        if self.hparams.config["loss_names"]["food101"] > 0:
            cls_num = self.hparams.config["food101_class_num"]
            self.food101_classifier = nn.Linear(hidden_size, cls_num)
            self.food101_classifier.apply(objectives.init_weights)  

            # feat_dim = hidden_size // 2 

            # self.food101_text_head = nn.Linear(feat_dim, cls_num)
            # self.food101_image_head = nn.Linear(feat_dim, cls_num)

            # self.food101_text_head.apply(objectives.init_weights)
            # self.food101_image_head.apply(objectives.init_weights)                
            
        if self.hparams.config["loss_names"]["mmimdb"] > 0:
            cls_num = self.hparams.config["mmimdb_class_num"]
            self.mmimdb_classifier = nn.Linear(hidden_size, cls_num)
            self.mmimdb_classifier.apply(objectives.init_weights) 

            # feat_dim = hidden_size // 2 

            # self.mmimdb_text_head = nn.Linear(feat_dim, cls_num)
            # self.mmimdb_image_head = nn.Linear(feat_dim, cls_num)

            # self.mmimdb_text_head.apply(objectives.init_weights)
            # self.mmimdb_image_head.apply(objectives.init_weights) 
            
        if self.hparams.config["load_path"] != "" and self.hparams.config["finetune_first"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.model.load_state_dict(state_dict, strict=False)            
            print("use pre-finetune model")

        if not self.hparams.config["test_only"]:
            for name, param in self.model.named_parameters():
                if "prompt_learner" not in name and "prompt" not in name and 'ln_final' not in name and 'ln_post' not in name and name.split('.')[-1]!='proj':
                    param.requires_grad_(False)

        clip_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=True)
        self.records = {}


    def infer(
        self,
        batch,
    ):
        text = batch["text"]
        img = batch["image"][0]  # extract the first view (total 1)
        if self.hparams.config["test_only"]:
            self.model.eval()

            if self.hparams.config["loss_names"]["hatememes"] > 0:
                self.hatememes_classifier.eval()
                # self.hatememes_text_head.eval()
                # self.hatememes_image_head.eval()
            if self.hparams.config["loss_names"]["food101"] > 0:
                self.food101_classifier.eval()
                # self.food101_text_head.eval()
                # self.food101_image_head.eval()
            if self.hparams.config["loss_names"]["mmimdb"] > 0:
                self.mmimdb_classifier.eval()
                # self.mmimdb_text_head.eval()
                # self.mmimdb_image_head.eval()

        both_feats = self.model(img, text, batch["missing_type"])  
        feature_dim = both_feats.shape[1]//2
        for idx in range(len(img)):
            if batch["missing_type"][idx] == 0:
                pass       
            elif batch["missing_type"][idx] == 1:  # missing text
                both_feats[idx, feature_dim:].zero_()
            elif batch["missing_type"][idx] == 2:
                both_feats[idx, :feature_dim].zero_()

        text_feats = both_feats[:, :feature_dim]      # [B, 512]
        image_feats = both_feats[:, feature_dim:]     # [B, 512]

        ret = {
            "cls_feats": both_feats,
            "text_feats": text_feats,
            "image_feats": image_feats,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # ret.update(objectives.compute_distance(self, batch, True))

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))
            
        # Binary classification for Hateful Memes
        if "hatememes" in self.current_tasks:
            ret.update(objectives.compute_hatememes(self, batch))
            
        # Multi-label classification for MM-IMDb
        if "mmimdb" in self.current_tasks:
            ret.update(objectives.compute_mmimdb(self, batch))
            
        # Classification for Food101
        if "food101" in self.current_tasks:
            ret.update(objectives.compute_food101(self, batch))              

        return ret

    def configure_optimizers(self):
        return clip_utils.set_schedule(self)  

    # ================= Helper: Sub-batch Indexing =================
    def _index_subbatch(self, batch, indices):
        """
        """
        if not torch.is_tensor(indices):
            indices = torch.as_tensor(indices, device=self.device)
        indices = indices.long()
        idx_list = indices.tolist()

        sub = {}

        # ---------- image ----------
        img = batch["image"]
        if torch.is_tensor(img):
            # [V, B, ...] or [B, ...]
            if img.dim() >= 2 and img.shape[0] <= 8 and img.shape[1] >= len(idx_list):
                sub["image"] = img[:, indices]
            else:
                sub["image"] = img[indices]
        elif isinstance(img, (list, tuple)):
            new_views = []
            for v in img:
                if torch.is_tensor(v):
                    new_views.append(v[indices])
                else:
                    new_views.append([v[i] for i in idx_list])
            sub["image"] = new_views
        else:
            sub["image"] = img

        # ---------- text ----------
        txt = batch["text"]
        if isinstance(txt, (list, tuple)):
            if len(txt) > 0 and isinstance(txt[0], list):
                # [V][B]
                new_txt_views = []
                for view in txt:
                    new_txt_views.append([view[i] for i in idx_list])
                sub["text"] = new_txt_views
            elif len(txt) > 0 and isinstance(txt[0], str):
                # [B] -> wrap to [1][B]
                sub["text"] = [[txt[i] for i in idx_list]]
            else:
                sub["text"] = txt
        else:
            if torch.is_tensor(txt):
                sub["text"] = txt[indices]
            else:
                sub["text"] = txt

        # ---------- label ----------
        lbl = batch["label"]
        if isinstance(lbl, list):
            sub["label"] = [lbl[i] for i in idx_list]
        elif torch.is_tensor(lbl):
            sub["label"] = lbl[indices]
        else:
            sub["label"] = lbl

        # ---------- missing_type ----------
        mt = batch.get("missing_type", None)
        if mt is not None:
            if isinstance(mt, list):
                sub["missing_type"] = [mt[i] for i in idx_list]
            elif torch.is_tensor(mt):
                sub["missing_type"] = mt[indices]
            else:
                sub["missing_type"] = mt

        # ---------- missing_ratio_sample ----------
        mrs = batch.get("missing_ratio_sample", None)
        if mrs is not None:
            if isinstance(mrs, list):
                mrs = torch.stack(mrs, dim=0)
            sub["missing_ratio_sample"] = mrs[indices]

        # Keep other keys
        for k, v in batch.items():
            if k not in sub:
                sub[k] = v

        return sub

    # ================= Helper: Task Building =================
    def _build_tasks_by_quantile(self, batch, num_tasks=3, n_support=8, n_query=8):
        if "missing_ratio_sample" not in batch:
            return []

        ratios = batch["missing_ratio_sample"]
        if isinstance(ratios, list):
            ratios = torch.stack(ratios, dim=0)
        ratios = ratios.view(-1)

        B = ratios.shape[0]
        per_task = n_support + n_query
        max_tasks = B // per_task
        if max_tasks == 0:
            return []

        num_tasks = min(num_tasks, max_tasks)

        _, idx_sorted = torch.sort(ratios)  # low -> high

        tasks = []
        for t in range(num_tasks):
            idx_t = idx_sorted[t * per_task:(t + 1) * per_task]
            support_idx = idx_t[:n_support]
            query_idx   = idx_t[n_support:]
            
            support = self._index_subbatch(batch, support_idx)
            query   = self._index_subbatch(batch, query_idx)
            tasks.append((t, support, query))
        return tasks

    # ================= Helper: Prompt Params =================
    def _prompt_param_names(self):
        names = []
        for n, p in self.model.named_parameters():
            if p.requires_grad and n.startswith("prompt_learner."):
                names.append(n)
        return names

    # ================= Helper: Inference with Params =================
    def _infer_cls_feats(self, batch, params=None):
        text = batch["text"]
        img  = batch["image"][0]  # [B,C,H,W]
        mt   = batch["missing_type"]

        if not torch.is_tensor(mt):
            mt = torch.as_tensor(mt, device=img.device)
        mt = mt.view(-1)

        if params is None:
            both_feats = self.model(img, text, mt)
        else:
            both_feats = functional_call(self.model, params, (img, text, mt))

        feature_dim = both_feats.shape[1] // 2 

        # Vectorized masking of missing modalities
        both_feats = both_feats.clone()
        both_feats[:, feature_dim:] *= (mt != 1).float().unsqueeze(1) # missing text -> zero text part
        both_feats[:, :feature_dim] *= (mt != 2).float().unsqueeze(1) # missing image -> zero image part

        return both_feats

    # ================= Loss: MMIMDB (BCE) =================
    def _mmimdb_bce_loss(self, batch, params=None):
        cls_feats = self._infer_cls_feats(batch, params=params)
        logits = self.mmimdb_classifier(cls_feats)

        labels = batch["label"]
        labels = torch.tensor(labels, device=self.device).float()

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss

    # ================= Loss: Food101 (CrossEntropy) =================
    def _food101_ce_loss(self, batch, params=None):
        cls_feats = self._infer_cls_feats(batch, params=params)
        logits = self.food101_classifier(cls_feats)

        labels = batch["label"]
        # CE Loss expects LongTensor class indices
        
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels, device=self.device)
        labels = labels.long()

        loss = F.cross_entropy(logits, labels)
        return loss
    
    # ================= Loss: Hatememes (CrossEntropy) =================
    def _hatememes_ce_loss(self, batch, params=None):
        cls_feats = self._infer_cls_feats(batch, params=params)
        logits = self.hatememes_classifier(cls_feats)

        labels = batch["label"]
        # CE Loss expects LongTensor class indices
        
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels, device=self.device)
        labels = labels.long()

        loss = F.cross_entropy(logits, labels)
        return loss

    # ================= Helper: Distance Loss with Params =================
    def _calc_dist_loss_internal(self, batch, params=None):
        """
         fast_params to Meta-Learning
        """
        # 1. _infer_cls_feats support params)
        both_feats = self._infer_cls_feats(batch, params=params)
        feature_dim = both_feats.shape[1] // 2
        
        # 2. feature split
        text_feats = both_feats[:, :feature_dim]
        image_feats = both_feats[:, feature_dim:]
        
        # 3. mask (Missing Type)
        missing_type = batch["missing_type"]
        if not torch.is_tensor(missing_type):
            missing_type = torch.as_tensor(missing_type, device=both_feats.device)
        missing_type = missing_type.view(-1)
        
        mask = (missing_type == 0)   # image available
        
        distance_loss = torch.tensor(0.0, device=both_feats.device)
        
        if mask.sum().item() > 0:
            t_feat = text_feats[mask]
            v_feat = image_feats[mask]
            
            t_feat = F.normalize(t_feat, dim=-1)
            v_feat = F.normalize(v_feat, dim=-1)
            
          
            w_t = getattr(self.hparams, "dist_center_text_weight", 0.2)
            w_v = getattr(self.hparams, "dist_center_image_weight", 0.8)
            w_sum = max(w_t + w_v, 1e-6)
            w_t = w_t / w_sum
            w_v = w_v / w_sum

            # Center
            with torch.no_grad():
                feat_center = w_t * t_feat + w_v * v_feat
                feat_center = F.normalize(feat_center, dim=-1)
                feat_center = feat_center.detach()
            
            #  MSE Loss
            dist_img_center = F.mse_loss(v_feat, feat_center)
            dist_txt_center = F.mse_loss(t_feat, feat_center)

            alpha = getattr(self.hparams, "dist_image_weight", 1.0)
            beta  = getattr(self.hparams, "dist_text_weight", 2.0)

            distance_loss = (alpha * dist_img_center) + (beta * dist_txt_center)
            
        return distance_loss
    
    def combined_loss_fn(self, batch, params):
        # 1. Classification Loss
        # ce_loss = self._hatememes_ce_loss(batch, params=params)
        if "food101" in self.current_tasks:
            cla_loss = self._food101_ce_loss(batch, params=params)
        elif "hatememes" in self.current_tasks:
            cla_loss = self._hatememes_ce_loss(batch, params=params)
        elif "mmimdb" in self.current_tasks:
            cla_loss = self._mmimdb_bce_loss(batch, params=params)
        else:
            raise ValueError(f"combined_loss_fn: unsupported task(s): {self.current_tasks}")
        # 2. Distance Loss
        dist_loss = self._calc_dist_loss_internal(batch, params=params)
        
        return cla_loss + dist_loss

    # ================= Inner Loop (Generalized) =================
    def _inner_adapt_prompts(self, support_batch, loss_fn, inner_lr=0.1, inner_steps=1):
        """Adapt prompts with SGD, optionally retaining support curvature for MAML."""
        second_order = self.hparams.config.get("meta_second_order", False)
        prompt_names = self._prompt_param_names()
        fast_params = {name: self.model.get_parameter(name) for name in prompt_names}

        for _ in range(inner_steps):
            # The math SDPA backend supports double backward on the support graph.
            attention = (torch.backends.cuda.sdp_kernel(enable_flash=False,
                         enable_mem_efficient=False, enable_math=True)
                         if second_order else nullcontext())
            with attention:
                loss_s = loss_fn(support_batch, params=fast_params)
                grads = torch.autograd.grad(loss_s, tuple(fast_params.values()),
                                            create_graph=second_order, allow_unused=True)
            fast_params = {
                name: weight if grad is None else weight - inner_lr * grad
                for (name, weight), grad in zip(fast_params.items(), grads)
            }
        return fast_params

    # ================= Training Step =================
    def training_step(self, batch, batch_idx):

        clip_utils.set_task(self)
        
        # Determine batch size for logging
        if isinstance(batch["image"], list):
            bs = batch["image"][0].shape[0]
        else:
            bs = batch["image"].shape[0]

        # ------------------- Food101 Meta -------------------
        use_food101_meta = ("food101" in self.current_tasks and bool(self.hparams.config.get("food101_meta", False)) and "missing_ratio_sample" in batch)

        if use_food101_meta:
            num_tasks = self.hparams.config.get("food101_meta_num_tasks", 3)
            n_support = self.hparams.config.get("food101_meta_n_support", 20)
            n_query   = self.hparams.config.get("food101_meta_n_query", 20)
            inner_lr  = self.hparams.config.get("food101_meta_inner_lr", 0.01)
            inner_steps = self.hparams.config.get("food101_meta_inner_steps", 1)

            tasks = self._build_tasks_by_quantile(batch, num_tasks, n_support, n_query)
            
            if len(tasks) == 0:
                return self._standard_loss_fallback(batch)

            meta_loss = 0.0
            for t, support, query in tasks:
                # 1. Adapt on Support (using CrossEntropy)
                fast_params = self._inner_adapt_prompts(
                    support, 
                    loss_fn=self.combined_loss_fn, 
                    inner_lr=inner_lr, 
                    inner_steps=inner_steps
                )
                # 2. Evaluate on Query (using CrossEntropy)
                loss_q = self._food101_ce_loss(query, params=fast_params)
                meta_loss += loss_q

            meta_loss = meta_loss / len(tasks)

            # Logging
            m_ratios = batch["missing_ratio_sample"]
            m_ratios = torch.stack(m_ratios) if isinstance(m_ratios, list) else m_ratios
            self.log("m_rate", m_ratios.mean(), prog_bar=True, on_step=True, batch_size=bs)
            self.log("meta_loss", meta_loss, prog_bar=True, on_step=True, batch_size=bs)
            return meta_loss
        

        # ------------------- Hatememes Meta -------------------
        use_hatememes_meta = ("hatememes" in self.current_tasks and bool(self.hparams.config.get("hatememes_meta", False)) and "missing_ratio_sample" in batch)

        if use_hatememes_meta:
            num_tasks = self.hparams.config.get("hatememes_meta_num_tasks", 3)
            n_support = self.hparams.config.get("hatememes_meta_n_support", 20)
            n_query   = self.hparams.config.get("hatememes_meta_n_query", 20)
            inner_lr  = self.hparams.config.get("hatememes_meta_inner_lr", 0.01)
            inner_steps = self.hparams.config.get("hatememes_meta_inner_steps", 1)

            tasks = self._build_tasks_by_quantile(batch, num_tasks, n_support, n_query)
            
            if len(tasks) == 0:
                return self._standard_loss_fallback(batch)

            meta_loss = 0.0
            # meta_dist_sum = 0.0
            for t, support, query in tasks:
                # 1. Adapt on Support (using CrossEntropy)
                fast_params = self._inner_adapt_prompts(
                    support, 
                    # loss_fn=self._hatememes_ce_loss, 
                    loss_fn=self.combined_loss_fn,
                    inner_lr=inner_lr, 
                    inner_steps=inner_steps
                )
                # 2. Evaluate on Query (using CrossEntropy)
                # loss_q_main = self._hatememes_ce_loss(query, params=fast_params)
                # loss_q_dist = self._calc_dist_loss_internal(query, params=fast_params)
                # meta_dist_sum += loss_q_dist.detach()

                # loss_q = loss_q_main + loss_q_dist
                # meta_loss += loss_q
                loss_q = self._hatememes_ce_loss(query, params=fast_params)
                meta_loss += loss_q

            meta_loss = meta_loss / len(tasks)
            # avg_dist_loss = meta_dist_sum / len(tasks)

            # Logging
            m_ratios = batch["missing_ratio_sample"]
            m_ratios = torch.stack(m_ratios) if isinstance(m_ratios, list) else m_ratios
            self.log("m_rate", m_ratios.mean(), prog_bar=True, on_step=True, batch_size=bs)
            # self.log("meta/dist_loss", avg_dist_loss, prog_bar=True, on_step=True, batch_size=bs)
            self.log("meta_loss", meta_loss, prog_bar=True, on_step=True, batch_size=bs)
            return meta_loss

        # ------------------- MMIMDB Meta -------------------
        use_mmimdb_meta = ("mmimdb" in self.current_tasks and bool(self.hparams.config.get("mmimdb_meta", False)) and "missing_ratio_sample" in batch)
        
        if use_mmimdb_meta:
            num_tasks = self.hparams.config.get("mmimdb_meta_num_tasks", 3)
            n_support = self.hparams.config.get("mmimdb_meta_n_support", 20)
            n_query   = self.hparams.config.get("mmimdb_meta_n_query", 20)
            inner_lr  = self.hparams.config.get("mmimdb_meta_inner_lr", 0.01)
            inner_steps = self.hparams.config.get("mmimdb_meta_inner_steps", 1)

            tasks = self._build_tasks_by_quantile(batch, num_tasks, n_support, n_query)
            
            if len(tasks) == 0:
                return self._standard_loss_fallback(batch)

            meta_loss = 0.0
            for t, support, query in tasks:
                # 1. Adapt on Support (using BCE)
                fast_params = self._inner_adapt_prompts(
                    support, 
                    loss_fn=self.combined_loss_fn, 
                    inner_lr=inner_lr, 
                    inner_steps=inner_steps
                )
                # 2. Evaluate on Query (using BCE)
                loss_q = self._mmimdb_bce_loss(query, params=fast_params)
                meta_loss += loss_q

            meta_loss = meta_loss / len(tasks)

            # Logging
            m_ratios = batch["missing_ratio_sample"]
            m_ratios = torch.stack(m_ratios) if isinstance(m_ratios, list) else m_ratios
            self.log("m_rate", m_ratios.mean(), prog_bar=True, on_step=True, batch_size=bs)
            self.log("meta_loss", meta_loss, prog_bar=True, on_step=True, batch_size=bs)
            return meta_loss    

        # ------------------- Fallback (Standard Training) -------------------
        return self._standard_loss_fallback(batch)

    def _standard_loss_fallback(self, batch):
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        if "missing_ratio_sample" in batch:
            m_ratios = batch["missing_ratio_sample"]
            m_ratios = torch.stack(m_ratios) if isinstance(m_ratios, list) else m_ratios
            self.log("m_rate", m_ratios.mean(), prog_bar=True, on_step=True)
            self.log("total_loss", total_loss, prog_bar=True, on_step=True)

        return total_loss
    
    def on_train_epoch_end(self):
        clip_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        clip_utils.set_task(self)
        _ = self(batch)  

    def on_validation_epoch_end(self):
        clip_utils.epoch_wrapup(self)

    def on_test_start(self):
        self._test_outputs = []

    def test_step(self, batch, batch_idx):
        clip_utils.set_task(self)
        output = self(batch)
        ret = {}

        if self.hparams.config["loss_names"].get("vqa", 0) > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        self._test_outputs.append(ret)
        return ret

    def on_test_epoch_end(self):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"].get("vqa", 0) > 0:
            objectives.vqa_test_wrapup(self._test_outputs, model_name)

        clip_utils.epoch_wrapup(self)
        self._test_outputs = []

