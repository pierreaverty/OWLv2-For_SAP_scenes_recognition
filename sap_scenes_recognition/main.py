import config
from PIL import Image
from utils.plotter import ObjectDetectionPlotter
from utils.owlv2 import collate_fn
from data.sap_scenes import SAPDetectionDataset
from trainers.owlv2 import OWLv2Trainer
from transformers import TrainingArguments

from models.owlv2 import (
    OWLv2, 
    OWLv2ForSapRecognition 
)

# Create training arguments
training_args = TrainingArguments(
    output_dir=config.OUTPUT_DIRECTORY_PATH,
    per_device_train_batch_size=config.BATCH_SIZE,
    save_steps=config.NB_STEPS,
    logging_steps=config.NB_STEPS/10,
    learning_rate=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY,
    remove_unused_columns=False,
    push_to_hub=False,
    dataloader_pin_memory=False,
    gradient_accumulation_steps=1,
    use_cpu=config.IS_CPU,
    max_steps=config.NB_STEPS,
)

def train():    
    """
    Trains the OWLv2 model using the specified training and validation datasets.

    Args:
        None

    Returns:
        None
    """
    # Create OWLv2 model instance
    owl = OWLv2()
    
    # Create training and validation datasets
    train_dataset = SAPDetectionDataset(directory=config.TRAIN_DIRECTORY_PATH, processor=owl.processor, target_image=config.TARGET if config.IS_IMAGE_QUERY else None)
    val_dataset = SAPDetectionDataset(directory=config.VAL_DIRECTORY_PATH, processor=owl.processor, target_image=config.TARGET if config.IS_IMAGE_QUERY else None)

    # Create OWLv2Trainer instance
    trainer = OWLv2Trainer(
        model=owl, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=val_dataset, 
        data_collator=collate_fn, 
        tokenizer=owl.processor
    )
    
    # Train the model
    trainer.train()
    
    # Save the trained model and tokenizer
    model_path = config.MODEL_PATH
    trainer.model.model.save_pretrained(model_path)
    trainer.tokenizer.save_pretrained(model_path)

def predict(): 
    """
    Runs the prediction process using OWLv2ForSapRecognition model.

    This function loads an image from the specified path, prepares the text input,
    and uses the OWLv2ForSapRecognition model to predict the scene in the image.
    The results are then plotted and saved to the specified output path.

    Args:
        None

    Returns:
        None
    """
    # Create OWLv2ForSapRecognition model instance
    owl = OWLv2ForSapRecognition()
    
    # Load the image
    image = Image.open(config.IMAGE_EVALUATION_PATH)
    
    # Prepare the text input
    text = [[config.TARGET]]
    
    # Predict the scene in the image
    results = owl.predict(image, text)
    
    # Create ObjectDetectionPlotter instance
    plotter = ObjectDetectionPlotter(results, texts=text, image=image)

    # Plot and save the results
    plotter.plot_results(config.OUTPUTS_RESULTS_PATH)
    
if __name__ == "__main__":
    # Check if training or prediction mode
    if config.IS_TRAINING:
        train()
    else:
        predict()