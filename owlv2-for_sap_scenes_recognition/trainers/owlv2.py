from transformers import Trainer

class OWLv2Trainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        
        print(labels)