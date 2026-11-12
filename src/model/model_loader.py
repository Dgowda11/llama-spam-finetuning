from transformers import AutoTokenizer

class ModelLoader:

    def __init__(self, model_id):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None

    def load_tokenizer(self):
        # Load and configure tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return self.tokenizer

    # def create_quantization_config(self):
    #     # Configure 4-bit quantization

    # def load_model(self):
    #     # Load pretrained model

    # def get_model(self):
    #     # Return loaded model

    # def get_tokenizer(self):
    #     # Return loaded tokenizer
