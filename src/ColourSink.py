import math
import random
from itertools import tee

import numpy as np
import matplotlib.colors as mcolours
import matplotlib.pyplot as plt


# Is this class overkill? Absolutely yes!
class ColourSink:
    def __init__(self):
        self.all_colours = [
            color
            for color in mcolours.CSS4_COLORS.keys()
            # Exclude dark colours
            if np.array(mcolours.to_rgb(color)).max(initial=0) > 0.5
            # Exclude boring grey colors
            and not np.allclose(mcolours.to_rgb(color)[:3], mcolours.to_rgb("grey")[:3])
            # Exclude very light colors
            and np.array(mcolours.to_rgb(color)).min(initial=1) < 0.6
        ]
        self.selected_colours = []

    def getColour(self, num_colours=1):
        if len(self.all_colours) < num_colours:
            print("Insufficient colours available.")
            return None

        # Select the very first colour randomly
        if len(self.selected_colours) == 0:
            first_colour = random.choice(self.all_colours)
            self.selected_colours.append(first_colour)
            self.all_colours.remove(first_colour)

        while len(self.selected_colours) < num_colours:
            max_distance = -1
            max_distance_colour = None

            for colour in self.all_colours:
                distance = self._calculate_colour_distance(colour)
                if distance > max_distance:
                    max_distance = distance
                    max_distance_colour = colour

            self.selected_colours.append(
                self.all_colours.pop(self.all_colours.index(max_distance_colour))
            )

        if len(self.all_colours) == 0:
            print("No more colours available.")

        return self.selected_colours

    def _calculate_colour_distance(self, colour):
        r1, g1, b1 = mcolours.to_rgb(colour)
        distances = []
        for selected_colour in self.selected_colours:
            r2, g2, b2 = mcolours.to_rgb(selected_colour)
            distance = math.sqrt((r2 - r1) ** 2 + (g2 - g1) ** 2 + (b2 - b1) ** 2)
            distances.append(distance)

        return min(distances) if distances else math.inf

    def displayColours(self):
        n_colours = len(self.selected_colours)
        colour_indices = np.arange(n_colours)
        bar_width = 0.8

        fig, ax = plt.subplots()
        rects = ax.bar(
            colour_indices,
            [1] * n_colours,
            width=bar_width,
            color=self.selected_colours,
        )
        ax.set_xticks(colour_indices)
        ax.set_xticklabels("")
        ax.set_yticks([])

        # Set the x-axis limits to include all the bars
        ax.set_xlim([-0.5, n_colours - 0.5])

        # Remove the top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add labels to the bars with colour names
        for i, rect in enumerate(rects):
            height = rect.get_height()
            height += 0.05 * (1 - (i % 2))
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height,
                self.selected_colours[i],
                ha="center",
                va="bottom",
            )

        plt.show()


if __name__ == "__main__":
    colour_sink = ColourSink()

    dissimilar_colours = colour_sink.getColour(num_colours=9)
    print(dissimilar_colours)

    colour_sink.displayColours()
