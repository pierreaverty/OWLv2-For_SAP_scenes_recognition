import os 
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ObjectDetectionPlotter:
    """
    A class for plotting object detection results.

    Args:
        results (list): List of dictionaries containing detection results.
        texts (list): List of text labels for each detected object.
        image (numpy.ndarray): The image on which to plot the results.

    Attributes:
        results (list): List of dictionaries containing detection results.
        texts (list): List of text labels for each detected object.
        image (numpy.ndarray): The image on which to plot the results.
    """

    def __init__(self, results, texts, image):
        self.results = results
        self.texts = texts
        self.image = image

    def plot_results(self, file_name):
        """
        Plot the object detection results on the image.

        Args:
            file_name (str): The name of the output file.
        """
        # Extract the boxes, scores, and labels from the detection results
        boxes, scores, labels = self.results[0]["boxes"], self.results[0]["scores"], self.results[0]["labels"]
        text = self.texts[0]

        # Generate random colors for each label
        colors = np.random.random((len(text), 3))

        # Create a figure and axis for plotting
        fig, ax = plt.subplots()
        ax.imshow(self.image)

        # Iterate over every box, score, and label
        for box, score, label in zip(boxes, scores, labels):
            # Convert the box coordinates to a list of rounded values
            box = [round(i, 2) for i in box.detach().numpy()]

            # Convert the box coordinates for Matplotlib (x, y, width, length)
            x, y, xmax, ymax = box
            rect = patches.Rectangle((x, y), xmax - x, ymax - y, linewidth=1, edgecolor=colors[label], facecolor='none')

            # Add the rectangle patch to the plot
            ax.add_patch(rect)

            # Add the text label with the score to the plot
            plt.text(x, y - 10, f'{text[label]}: {round(score.item(), 2)}', color='white', fontsize=8, backgroundcolor=colors[label])

            # Print the detection information
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

        # Turn off the axis
        plt.axis('off')

        # Create the results directory if it doesn't exist
        if not os.path.exists('../results'):
            os.makedirs('../results')
            
        # Save the plot as an image file
        plt.savefig(f'../results/{file_name}.png')