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

    def create_quantization_config(self):
        # Configure 4-bit quantization
        from transformers import BitsAndBytesConfig
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16"
        )
        return self.quantization_config

    def load_model(self):
        # Load pretrained model
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=self.quantization_config,
            device_map="auto",
            
        )
        return self.model

    # def get_model(self):
    #     # Return loaded model

    # def get_tokenizer(self):
    #     # Return loaded tokenizer
