import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_comparison(model_accuracies):
    """
    Plots a bar chart comparing the accuracies of different models.

    Args:
        model_accuracies (dict): A dictionary where keys are model names (str)
                                 and values are their accuracies (float).
                                 Example: {'BERT': 0.85, 'LR': 0.78, 'GPT-2': 0.82}
    """
    models = list(model_accuracies.keys())
    accuracies = list(model_accuracies.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison on Test Set")
    plt.ylim(0, 1)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center') # Add text labels

    plt.show()
    # TODO: Add option to save the plot to a file.

def plot_sentiment_score_comparison(model_sentiment_scores, true_sentiments=None):
    """
    Plots a comparison of sentiment scores from different models.
    This could be histograms, box plots, or density plots.

    Args:
        model_sentiment_scores (dict): A dictionary where keys are model names (str)
                                     and values are lists/arrays of sentiment scores
                                     predicted by the model for the test set.
                                     Example: {'BERT': [0.1, 0.8, -0.5, ...],
                                               'LR': [0.2, 0.7, -0.3, ...],
                                               'GPT-2': [0.15, 0.85, -0.4, ...]}
        true_sentiments (list, optional): A list/array of true sentiment scores or categories
                                          for the test set, if available, for comparison.
    """
    num_models = len(model_sentiment_scores)
    model_names = list(model_sentiment_scores.keys())

    # Example: Using histograms
    # TODO: Allow choice of plot type (histogram, box plot, density)
    fig, axes = plt.subplots(num_models, 1, figsize=(10, 4 * num_models), sharex=True, sharey=False)
    if num_models == 1: # Make axes an array even if it's a single subplot
        axes = [axes]

    bins = np.linspace(-1, 1, 50) # Assuming sentiment scores are between -1 and 1

    for i, model_name in enumerate(model_names):
        scores = model_sentiment_scores[model_name]
        axes[i].hist(scores, bins=bins, alpha=0.7, label=f'{model_name} Predicted')
        if true_sentiments is not None:
            # This assumes true_sentiments are comparable numbers, adjust if they are categories
            axes[i].hist(true_sentiments, bins=bins, alpha=0.5, label='True Sentiments', histtype='step', linewidth=1.5)
        axes[i].set_title(f'Sentiment Score Distribution: {model_name}')
        axes[i].set_ylabel("Frequency")
        axes[i].legend()

    axes[-1].set_xlabel("Sentiment Score")
    plt.tight_layout()
    plt.show()
    # TODO: Add option to save the plot to a file.

def load_model_outputs():
    """
    Placeholder function to load or compute model outputs.
    This function should return the accuracies and sentiment scores for each model.

    Returns:
        tuple: (model_accuracies, model_sentiment_scores_test)
               model_accuracies (dict): As defined in plot_accuracy_comparison.
               model_sentiment_scores_test (dict): As defined in plot_sentiment_score_comparison.
                                                   Scores are for the test set.
    """
    # These are dummy values. Replace with actual data loading/computation.
    print("INFO: Loading dummy model outputs. Replace with actual data.")
    dummy_accuracies = {
        'BERT': 0.88,
        'LR': 0.76,
        'GPT-2': 0.82
    }
    dummy_sentiment_scores_test = {
        'BERT': np.random.normal(0.6, 0.3, 1000).clip(-1, 1),
        'LR': np.random.normal(0.3, 0.4, 1000).clip(-1, 1),
        'GPT-2': np.random.normal(0.5, 0.35, 1000).clip(-1, 1)
    }
    # Example true sentiments (e.g. if scores were continuous labels)
    # If true labels are categorical (e.g., positive/negative/neutral),
    # the sentiment score plot might need adjustment or a different kind of plot.
    # For now, let's assume we might have continuous true scores for comparison, or we can ignore this.
    # true_test_labels_scores = np.random.normal(0.4, 0.5, 1000).clip(-1, 1) # Dummy true scores
    
    return dummy_accuracies, dummy_sentiment_scores_test #, true_test_labels_scores (if available)

def main():
    """
    Main function to generate and display visualizations.
    """
    # 1. Load model outputs (accuracies and sentiment scores on the test set)
    #    This is where you'd integrate with your actual model evaluation pipeline.
    #    You'll need to fetch:
    #    - Accuracy of BERT on the test set
    #    - Accuracy of Logistic Regression (LR) on the test set
    #    - Accuracy of GPT-2 on the test set
    #    - Sentiment scores from BERT for each item in the test set
    #    - Sentiment scores from LR for each item in the test set
    #    - Sentiment scores from GPT-2 for each item in the test set
    #    - (Optional) True sentiment scores/labels for the test set items for comparison in plots

    model_accuracies, model_sentiment_scores = load_model_outputs()
    # true_labels_for_scores = None # Or load actual true scores if available and relevant

    # 2. Plot accuracy comparison
    if model_accuracies:
        plot_accuracy_comparison(model_accuracies)

    # 3. Plot sentiment score comparison
    if model_sentiment_scores:
        # plot_sentiment_score_comparison(model_sentiment_scores, true_sentiments=true_labels_for_scores)
        plot_sentiment_score_comparison(model_sentiment_scores)


if __name__ == '__main__':
    main() 