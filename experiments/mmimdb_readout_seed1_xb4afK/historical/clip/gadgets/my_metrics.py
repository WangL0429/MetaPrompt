import torch
from torchmetrics.functional import f1_score, auroc
# from pytorch_lightning.metrics import Metric
from torchmetrics import Metric

class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        if logits.size(-1)>1:
            preds = logits.argmax(dim=-1)
        else:
            preds = (torch.sigmoid(logits)>0.5).long()
            
        preds = preds[target != -100]
        target = target[target != -100]
        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total

class AUROC(Metric):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, logits: torch.Tensor, target: torch.Tensor):
        logits = logits.detach().to(self.device)
        target = target.detach().to(self.device)
        target = target.view(-1)
        self.logits.append(logits)
        self.targets.append(target)

    def compute(self):
        if isinstance(self.logits, list):
            if len(self.logits) == 0:
                return torch.tensor(float("nan"), device=self.device)

            all_logits = torch.cat(self.logits, dim=0)
            all_targets = torch.cat(self.targets, dim=0).long()
        else:
            all_logits = self.logits
            all_targets = self.targets.long()

        if all_targets.numel() == 0 or all_targets.unique().numel() < 2:
            return torch.tensor(float("nan"), device=self.device)

        if all_logits.ndim > 1 and all_logits.size(-1) > 1:
            probs = torch.softmax(all_logits, dim=-1)
            num_classes = all_logits.size(-1)   
            return auroc(probs, all_targets, task="multiclass", num_classes=num_classes)
        else:
            scores = all_logits.view(-1)
            probs = torch.sigmoid(scores)
            return auroc(probs, all_targets.view(-1), task="binary")


# class AUROC(Metric):
#     def __init__(self, dist_sync_on_step=False):
#         super().__init__(dist_sync_on_step=dist_sync_on_step)
#         # Using lists is okay if we handle the device manually, 
#         # but torchmetrics prefers registered tensors.
#         self.add_state("logits", default=[], dist_reduce_fx="cat")
#         self.add_state("targets", default=[], dist_reduce_fx="cat")

#     def update(self, logits: torch.Tensor, target: torch.Tensor):
#         # 1. Ensure we detach to prevent memory leaks (keep the computation graph out of the state)
#         # 2. Lightning handles moving these to the correct device if they are tensors
#         self.logits.append(logits.detach())
#         self.targets.append(target.detach())

#     def compute(self):
#         # Concatenate lists of tensors into a single tensor
#         all_logits = torch.cat(self.logits, dim=0)
#         all_targets = torch.cat(self.targets, dim=0).long()
    
#         if all_logits.size(-1) > 1:
#             # Multi-class or Multi-label (depending on your setup)
#             all_probs = torch.softmax(all_logits, dim=1)
#             # Ensure num_classes matches your logit shape
#             num_classes = all_logits.size(-1)
#             res = auroc(all_probs, all_targets, task="multiclass", num_classes=num_classes)
#         else:
#             # Binary case
#             all_probs = torch.sigmoid(all_logits).squeeze()
#             res = auroc(all_probs, all_targets, task="binary")
        
#         return res

# class AUROC(Metric):
#     def __init__(self, dist_sync_on_step=False):
#         super().__init__(dist_sync_on_step=dist_sync_on_step)
#         self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
#         self.add_state("logits", default=[], dist_reduce_fx="cat")
#         self.add_state("targets", default=[], dist_reduce_fx="cat")

#     def update(self, logits, target):
#         logits, targets = (
#             logits.detach().to(self.correct.device),
#             target.detach().to(self.correct.device),
#         )
#         self.logits.append(logits)
#         self.targets.append(targets)

#     def compute(self):
#         if isinstance(self.logits, list):
#             all_logits = torch.cat(self.logits)
#             all_targets = torch.cat(self.targets).long()
#         else:
#             all_logits = self.logits
#             all_targets = self.targets.long()

#         # all_targets: [B], 0/1
#         if all_logits.size(-1) > 1:
#             # logits: [B, 2] → 正类概率 [B]
#             all_probs = torch.softmax(all_logits, dim=1)[:, 1]
#         else:
#             # logits: [B] → 正类概率 [B]
#             all_probs = torch.sigmoid(all_logits)

#         AUROC_val = auroc(
#             all_probs,
#             all_targets,
#             task="binary",
#         )
#         return AUROC_val

    
# class F1_Score(Metric):
#     def __init__(self, dist_sync_on_step=False):
#         super().__init__(dist_sync_on_step=dist_sync_on_step)
#         self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
#         self.add_state("logits", default=[], dist_reduce_fx="cat")
#         self.add_state("targets", default=[], dist_reduce_fx="cat")

#     def update(self, logits, target):
#         logits, targets = (
#             logits.detach().to(self.correct.device),
#             target.detach().to(self.correct.device),
#         )
        
#         self.logits.append(logits)
#         self.targets.append(targets)


#     def compute(self, use_sigmoid=True):
#         if type(self.logits) == list:
#             all_logits = torch.cat(self.logits)
#             all_targets = torch.cat(self.targets).long()
#         else:
#             all_logits = self.logits
#             all_targets = self.targets.long()
#         if use_sigmoid:
#             all_logits = torch.sigmoid(all_logits)
#         # Micro F1
#         F1_Micro = f1_score(all_logits, all_targets, task="multilabel", num_labels=23, average='micro')
#         # Macro F1
#         F1_Macro = f1_score(all_logits, all_targets, task="multilabel", num_labels=23, average='macro')
#         # Samples F1
#         F1_Samples = f1_score(all_logits, all_targets, task="multilabel", num_labels=23, average='samples')
#         # Weighted F1
#         F1_Weighted = f1_score(all_logits, all_targets, task="multilabel", num_labels=23, average='weighted')
        
#         return (F1_Micro, F1_Macro, F1_Samples, F1_Weighted)



class F1_Score(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = 23
        self.task = "multilabel"

        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, logits, target):
        logits, targets = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        self.logits.append(logits)
        self.targets.append(targets)

    def compute(self, use_sigmoid=True):

        if len(self.logits) == 0:
            device = self.correct.device 
            zero = torch.tensor(0.0, device=device)
            return (zero, zero, zero, zero)

        if isinstance(self.logits, list):
            all_logits = torch.cat(self.logits)
            all_targets = torch.cat(self.targets).long()
        else:
            all_logits = self.logits
            all_targets = self.targets.long()

        # MM-IMDb 
        if use_sigmoid:
            all_logits = torch.sigmoid(all_logits)
        
        # 1. Micro F1
        F1_Micro = f1_score(all_logits, all_targets, task="multilabel", num_labels=23, average='micro')
        
        # 2. Macro F1
        F1_Macro = f1_score(all_logits, all_targets, task="multilabel", num_labels=23, average='macro')
        
        # 3. Samples F1 or not use
        # F1_Samples = f1_score(all_logits, all_targets, task="multilabel", num_labels=23, average='samples')
        preds = (all_logits >= 0.5).float()
        targets = all_targets.float()
        tp = (preds * targets).sum(dim=1)
        pred_count = preds.sum(dim=1)
        target_count = targets.sum(dim=1)
        f1_per_sample = 2 * tp / (pred_count + target_count + 1e-8)
        F1_Samples = f1_per_sample.mean()

        
        # 4. Weighted F1
        F1_Weighted = f1_score(all_logits, all_targets, task="multilabel", num_labels=23, average='weighted')
        
        return (F1_Micro, F1_Macro, F1_Samples, F1_Weighted)


    
class check(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, logits, target):
        logits, targets = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        
        self.logits.append(logits)
        self.targets.append(targets)


    def compute(self, use_sigmoid=True):
        if type(self.logits) == list:
            all_logits = torch.cat(self.logits).long()
            all_targets = torch.cat(self.targets).long()
        else:
            all_logits = self.logits.long()
            all_targets = self.targets.long()

        mislead = all_logits ^ all_targets
        accuracy = mislead.sum(dim=0)
        return accuracy
        
class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total    
    
class Scalar2(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar, num):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        
        self.scalar += scalar
        self.total += num

    def compute(self):
        return self.scalar / self.total


class VQAScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * target

        self.score += scores.sum()
        self.total += len(logits)

    def compute(self):
        return self.score / self.total
