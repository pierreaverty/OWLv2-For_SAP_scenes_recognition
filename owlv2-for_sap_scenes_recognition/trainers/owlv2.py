from transformers import Trainer
from losses.owlv2 import OWLv2LossFunction

class OWLv2Trainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["target"]
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        
        outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        
        loss = OWLv2LossFunction()(outputs, labels)

        return (loss, outputs) if return_outputs else loss
