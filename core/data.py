import random
from datasets import load_dataset
from typing import List, Tuple

class DatasetLoader:
    """Handles dataset loading and sampling."""
    
    @staticmethod
    def get_sample_prompts(dataset_name: str, num_samples: int, seed: int = 42) -> Tuple[List[str], List[int]]:
        """Get sample prompts from dataset."""
        print(f"Loading dataset: {dataset_name}")
        
        dataset = load_dataset(dataset_name)
        split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
        
        random.seed(seed)
        indices = random.sample(range(len(dataset[split_name])), num_samples)
        
        # Handle different dataset formats
        samples = []
        for idx in indices:
            item = dataset[split_name][idx]
            if 'instruction' in item:
                samples.append(item['instruction'])
            elif 'text' in item:
                samples.append(item['text'])
            elif 'prompt' in item:
                samples.append(item['prompt'])
            else:
                # Fallback - use first text field
                text_field = next(k for k, v in item.items() if isinstance(v, str))
                samples.append(item[text_field])
        
        return samples, indices