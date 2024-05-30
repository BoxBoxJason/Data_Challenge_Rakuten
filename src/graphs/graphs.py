import matplotlib.pyplot as plt


def drawScoresBarChart(score_type, scores_dicts, save_path=None, show=False):
    """
    @brief Draws a bar chart for the given scores.

    This function draws a bar chart for the given scores.

    @param scores_dicts The scores to draw the bar chart for.
    """
    model_names = []
    scores = []
    for model_name,score in sorted(scores_dicts.items(), key=lambda x: x[1]):
        model_names.append(model_name)
        scores.append(score)
    x = range(len(scores))

    # Graph styling
    fig = plt.figure()

    # Removing axis lines
    for side in ['right', 'top','left']:
        plt.gca().spines[side].set_visible(False)

    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title(f'{score_type} score by model')
    plt.legend(model_names)
    bars = plt.bar(x, scores)

    for bar in bars:
        bar.set_color(floatToHexColor(bar.get_height()))
        bar_color = bar.get_facecolor()
        plt.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                round(bar.get_height()*100, 1),
                horizontalalignment='center',
                color=bar_color,
                weight='bold'
                )
    plt.xticks(x, model_names)
    plt.xticks(rotation=90)
    plt.tight_layout()


    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


def floatToHexColor(value):
    """
    @brief Converts a float value to a hex color.

    This function converts a float value to a hex color, where the value is between 0 and 1.
    A value of under 0.5 corresponds to red, and a value of 1 corresponds to green.

    @param value The float value to convert.
    @return The hex color.
    """
    if not (0 <= value <= 1):
        raise ValueError("Value must be between 0 and 1")

    # Calculate the red and green components
    red = 255 if value < 0.5 else int(255 * (1 - value) ** 0.7 * 2)
    green = int(255 * value) if value >= 0.5 else 0
    blue = 0  # No blue component

    # Format as a hex color code
    hex_color = f"#{red:02x}{green:02x}{blue:02x}"
    return hex_color


def drawSimilarityHeatmap(matrix, model_names, save_path=None, show=True):
    """
    @brief Draws a heatmap for the given similarity matrix.

    This function draws a heatmap for the given similarity matrix. Darker colors indicate higher similarity.
    Display model names on the x and y axes. Shows the legend for color scale.

    @param similarity_matrix The similarity matrix to draw the heatmap for.

    """
    fig = plt.figure()

    plt.imshow(matrix, cmap='RdYlGn', interpolation='nearest')
    plt.xticks(range(len(model_names)), model_names, rotation=90)
    plt.yticks(range(len(model_names)), model_names)
    plt.colorbar()
    plt.title('Model Prediction similarity %')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
