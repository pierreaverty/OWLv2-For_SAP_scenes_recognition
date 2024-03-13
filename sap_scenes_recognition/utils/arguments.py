import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="OWLv2 for SAP scenes recognition")

# Add command line arguments
parser.add_argument('--train', action='store_true', help='Set the model in training mode')
parser.add_argument('--image_query', action='store_true', help='Use image querying for the model')
parser.add_argument('--cpu', action='store_true', help='Use CPU for training')

parser.add_argument('--model_path', type=str, help='The path to the model to use for prediction', required=True)
parser.add_argument('--image_path', type=str, help='The path to the image to use for prediction', required=True)

parser.add_argument('--target', type=str, help='The target label for the image', required=True)

parser.add_argument('--train_directory', type=str, help='The path to the training directory', required=True)
parser.add_argument('--val_directory', type=str, help='The path to the validation directory', required=True)
parser.add_argument('--output_dir', type=str, help='The path to the output directory', required=True)

parser.add_argument('--output_results', type=str, help='The path to the output results')

parser.add_argument('--batch_size', type=int, help='The batch size for training', default=1)
parser.add_argument('--num_epochs', type=int, help='The number of epochs for training', default=2)
parser.add_argument('--learning_rate', type=float, help='The learning rate for training', default=1e-5)
parser.add_argument('--weight_decay', type=float, help='The weight decay for training', default=1e-4)
parser.add_argument('--nb_steps', type=int, help='The number of steps for training', default=300)

# Parse the command line arguments
args = parser.parse_args()

# Check if either --train or --output_results is specified
if not args.train and not args.output_results:
    parser.error('You must specify either --train or --output_results')