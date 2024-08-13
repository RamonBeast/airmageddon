import torch
import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 
from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch
from utils.logger import Logger
from conf.configuration import Configuration

class Florence():
    model = None

    def __init__(self):
        self.config = Configuration()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        path = os.path.join(self.config.get_config_param('models_folder'), 'Florence-2-large')

        if not os.path.exists(path):
            path = 'microsoft/Florence-2-large'

        with patch("transformers.dynamic_module_utils.get_imports", self._fix_imports):
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True
            
            ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            path,
            trust_remote_code=True
        )

    # Monkey patch imports as CPU inference doesn't support flash attention
    def _fix_imports(self, filename: str | os.PathLike) -> list[str]:
        imports = get_imports(filename)

        if not torch.cuda.is_available() and "flash_attn" in imports:
            imports.remove("flash_attn")

        return imports

    def process_frame(self, task_prompt: str, image: Image=None) -> str:
        if self.model is None:
            Logger.error('Florence model has not been correctly loaded')
            return ''
        
        prompt = task_prompt
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

        Logger.info(f'[M] Florence: {parsed_answer["<MORE_DETAILED_CAPTION>"]}')
        return parsed_answer
