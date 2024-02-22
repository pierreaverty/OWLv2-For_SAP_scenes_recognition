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

    def plot_results(self):
        """
        Plot the object detection results on the image.
        """
        boxes, scores, labels = self.results[0]["boxes"], self.results[0]["scores"], self.results[0]["labels"]
        text = self.texts[0]
        colors = np.random.random((len(text), 3))
        fig, ax = plt.subplots()
        ax.imshow(self.image)

        # Iteration on every boxes, scores and labels
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.detach().numpy()]

            # Convertion of the coordinates of the box for Matplotlib (x, y, width, length)
            x, y, xmax, ymax = box
            rect = patches.Rectangle((x, y), xmax - x, ymax - y, linewidth=1, edgecolor=colors[label], facecolor='none')

            ax.add_patch(rect)
            plt.text(x, y - 10, f'{text[label]}: {round(score.item(), 2)}', color='white', fontsize=8, backgroundcolor=colors[label])

            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

        plt.axis('off')
        plt.show()