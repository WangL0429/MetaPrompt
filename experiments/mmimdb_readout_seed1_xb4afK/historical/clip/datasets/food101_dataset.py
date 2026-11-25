from .base_dataset import BaseDataset
import torch
import random, os

class FOOD101Dataset(BaseDataset):
    def __init__(self, *args, split="", missing_info={}, **kwargs):

        self.split_missing_ratio = missing_info.get('ratio', {}).get(split, 0.0)
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["food101_train"]
        elif split == "val":
            names = ["food101_val"]
        elif split == "test":
            names = ["food101_test"] 

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="text",
            remove_duplicate=False,
        )

        self.missing_table = None

        if self.split != 'train':
            # load missing table
            
            missing_ratio = missing_info.get('ratio',{}).get(split,0)
            missing_type = missing_info.get('type',{}).get(split,"both")

            if missing_ratio > 0:
                self._load_or_generate_fixed_table(names[0], missing_info, split)
    
    def _load_or_generate_fixed_table(self, name, missing_info, split):
        """
        Fixed mask tables used only for generating test/validation sets.
        """
        missing_ratio = missing_info['ratio'][split]
        missing_type = missing_info['type'][split]
        mratio = str(missing_ratio).replace('.','')

        both_ratio = missing_info.get('both_ratio', 0.5) 
        
        missing_table_root = missing_info['missing_table_root']
        missing_table_name = f'{name}_missing_{missing_type}_{mratio}.pt'
        missing_table_path = os.path.join(missing_table_root, missing_table_name)
        
        total_num = len(self.table['image'])

        if os.path.exists(missing_table_path):
            self.missing_table = torch.load(missing_table_path)
            if len(self.missing_table) != total_num:
                raise ValueError('Missing table mismatched!')
        else:
            print(f"Generating missing table for {split}...")
            
            missing_table = torch.zeros(total_num)
            missing_index = random.sample(range(total_num), int(total_num*missing_ratio))

            if missing_type == 'text':
                missing_table[missing_index] = 1
            elif missing_type == 'image':
                missing_table[missing_index] = 2
            elif missing_type == 'both':
                # 这里的 both 实际上是混合缺失
                missing_table[missing_index] = 1
                # 再次随机选一部分改为缺图
                image_missing_count = int(len(missing_index) * both_ratio)
                if image_missing_count > 0:
                    missing_index_image = random.sample(missing_index, image_missing_count)
                    missing_table[missing_index_image] = 2
            
            if not os.path.exists(missing_table_root):
                os.makedirs(missing_table_root, exist_ok=True)
                    
            torch.save(missing_table, missing_table_path)
            self.missing_table = missing_table
            
    def __getitem__(self, index):
        planned_missing = isinstance(index, tuple)
        if planned_missing:
            index, current_missing_type, current_missing_prob = index
        image_index, question_index = self.index_mapper[index]

        if planned_missing:
            missing_ratio_sample = torch.tensor([current_missing_prob], dtype=torch.float)
        elif self.split == 'train':
            # =========================================================
            # 基于 Beta 分布的动态缺失率采样
            # =========================================================
            
            # random.betavariate(alpha, beta) --- missing_prob
            current_missing_prob = random.betavariate(3, 2)
            
            # 2. Based on the probability of sampling, determine whether the current sample is missing.
            if random.random() < current_missing_prob:
                # missing_type: 1 - text missing, 2 - image missing
                current_missing_type = random.choice([1, 2])
            else:
                # 0 - complete
                current_missing_type = 0
            
            missing_ratio_sample = torch.tensor([current_missing_prob], dtype=torch.float)
            
        else:
            # test/val - missing_table
            if self.missing_table is not None:
                current_missing_type = int(self.missing_table[image_index].item())
            else:
                current_missing_type = 0 
                
            _ratio_val = 1.0 if current_missing_type != 0 else 0.0
            missing_ratio_sample = torch.tensor([_ratio_val], dtype=torch.float)

        # =========================================================
        #  Masking
        # =========================================================
        
        # 1. image processing
        image_tensor = self.get_image(index)["image"]
        
        if current_missing_type == 2:
            if isinstance(image_tensor, list):
                image_tensor = [torch.zeros_like(img) for img in image_tensor]
            else:
                image_tensor = torch.zeros_like(image_tensor)
            
        # 2. text processing
        if current_missing_type == 1:
            text = ''
        else:
            text = self.get_text(index)["text"]

        labels = self.table["label"][image_index].as_py()

        return {
            "image": image_tensor,
            "text": text,
            "label": labels,
            "missing_type": current_missing_type,
            "missing_ratio_sample": missing_ratio_sample,  # Sampled missing ratio for training
            "missing_ratio_split": self.split_missing_ratio, # Global missing ratio for the current split (evaluation only)
        }