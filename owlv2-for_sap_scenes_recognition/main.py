from PIL import Image

from models.owlv2 import OWLv2ForSapRecognition 

from utils.plotter import ObjectDetectionPlotter
from utils.owlv2 import collate_fn

from datasets.sap_scenes import SAPDetectionDataset

from trainers.owlv2 import OWLv2Trainer

from transformers import TrainingArguments

train_directory = "/home/omilab-gpu/OWLv2-For_SAP_scenes_recognition/data/train/"
val_directory = "/home/omilab-gpu/OWLv2-For_SAP_scenes_recognition/data/val/"
test_pred_path = "/home/omilab-gpu/OWLv2-For_SAP_scenes_recognition/data/train/images/woman1_front_1.jpg"

training_args = TrainingArguments(
    output_dir="owlvit-base-patch32_FT_sap_scenes_recognition",
    per_device_train_batch_size=1,
    num_train_epochs=2,
    fp16=True,
    save_steps=200,
    logging_steps=50,
    learning_rate=1e-5,
    weight_decay=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    dataloader_pin_memory=False,
    gradient_accumulation_steps=1
)

def main():
    owl = OWLv2ForSapRecognition()
    
    train_dataset = SAPDetectionDataset(directory=train_directory, processor=owl.processor)
    val_dataset = SAPDetectionDataset(directory=val_directory, processor=owl.processor)
    
    trainer = OWLv2Trainer(owl, training_args, train_dataset, val_dataset, data_collator=collate_fn, tokenizer=owl.processor)
    
    trainer.train()

    image = Image.open(test_pred_path)
    
    text = [["women1_front"]]
    results = owl.predict(image, text)
    plotter = ObjectDetectionPlotter(results, texts=text, image=image)

    plotter.plot_results("output")
    
if __name__ == "__main__":
    main()