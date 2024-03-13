import argparse

from PIL import Image

from models.owlv2 import OWLv2, OWLv2ForSapRecognition 

from utils.plotter import ObjectDetectionPlotter
from utils.owlv2 import collate_fn

from data.sap_scenes import SAPDetectionDataset

from trainers.owlv2 import OWLv2Trainer

from transformers import TrainingArguments

train_directory = "/home/omilab-gpu/OWLv2-For_SAP_scenes_recognition/data/train/"
val_directory = "/home/omilab-gpu/OWLv2-For_SAP_scenes_recognition/data/val/"
test_pred_path = "/home/omilab-gpu/OWLv2-For_SAP_scenes_recognition/data/iLoveIMG IMG 6542.jpg"

parser = argparse.ArgumentParser(description="OWLv2 for SAP scenes recognition")

parser.add_argument('--train','-t', action='store_true', help='Set the model in training mode')

training_args = TrainingArguments(
    output_dir="../owlvit-base-patch32_FT_sap_scenes_recognition",
    per_device_train_batch_size=1,
    num_train_epochs=2,
    save_steps=900,
    logging_steps=10,
    learning_rate=1e-5,
    weight_decay=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    dataloader_pin_memory=False,
    gradient_accumulation_steps=1,
    use_cpu=True,
    max_steps=900,
)

args = parser.parse_args()
    
def train():    
    owl = OWLv2()
    
    train_dataset = SAPDetectionDataset(directory=train_directory, processor=owl.processor)
    val_dataset = SAPDetectionDataset(directory=val_directory, processor=owl.processor)

    trainer = OWLv2Trainer(
        model=owl, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=val_dataset, 
        data_collator=collate_fn, 
        tokenizer=owl.processor
    )
    
    trainer.train()
    
    model_path = "../owlvit-base-patch32_FT_sap_scenes_recognition"

    trainer.model.model.save_pretrained(model_path)
    trainer.tokenizer.save_pretrained(model_path)

def predict(): 
    owl = OWLv2ForSapRecognition()
    
    image = Image.open(test_pred_path)
    
    text = [["women1_front"]]
    results = owl.predict(image, text)
    plotter = ObjectDetectionPlotter(results, texts=text, image=image)

    plotter.plot_results("output")
    
if __name__ == "__main__":
    if args.train:
        train()
    else:
        predict()