from utils.arguments import args  # Importing the 'args' object from the 'utils.arguments' module

# Assigning values to variables based on the values in 'args' object
TRAIN_DIRECTORY_PATH = args.train_directory  # Path to the training directory
VAL_DIRECTORY_PATH = args.val_directory  # Path to the validation directory
IMAGE_EVALUATION_PATH = args.image_path  # Path to the image evaluation directory
OUTPUT_DIRECTORY_PATH = args.output_dir  # Path to the output directory
OUTPUT_RESULTS_PATH = args.output_results  # Path to the output results directory
MODEL_PATH = args.model_path  # Path to the model directory

TARGET = args.target  # Target variable

BATCH_SIZE = args.batch_size  # Batch size for training
NUM_EPOCHS = args.num_epochs  # Number of training epochs
LEARNING_RATE = args.learning_rate  # Learning rate for the optimizer
WEIGHT_DECAY = args.weight_decay  # Weight decay for regularization
NB_STEPS = args.nb_steps  # Number of steps

IS_IMAGE_QUERY = args.image_query  # Flag indicating if it's an image query
IS_TRAINING = args.train  # Flag indicating if it's a training process
IS_CPU = args.use_cpu  # Flag indicating if CPU should be used