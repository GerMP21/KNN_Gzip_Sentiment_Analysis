# Sentiment Analysis using NCD and K-Nearest Neighbors

This project involves sentiment analysis on a dataset of Dell-related tweets. Sentiment analysis is a process of determining the sentiment or emotion expressed in a text. In this code, we use the Normalized Compression Distance (NCD) as a feature to perform sentiment classification using the K-Nearest Neighbors (KNN) algorithm.
## Dependencies

    Python 3.x
    numpy
    pandas
    scikit-learn

## Getting Started

    Clone the repository or download the code files.
    Install the required dependencies using pip install numpy pandas scikit-learn.

## Code Overview

###    Data Loading and Preprocessing

    The code starts by loading a CSV file containing labeled Dell-related tweets. It then selects a subset of samples for processing. The 'Text' column is used as the input feature (X), and the 'sentiment' column is used as the target label (y). The labels are mapped to integers for compatibility with the classification algorithm.

###    Text Preprocessing

    Text preprocessing is performed to clean the tweet text. Tags, hashtags, and URLs are removed from the text using regular expressions. This ensures that the input data is clean and ready for further processing.

###    Train-Test Split

    The data is split into training and testing sets using the train_test_split function from scikit-learn. This division is essential to train the model on one subset and evaluate it on another, avoiding overfitting.

###    Normalized Compression Distance (NCD) Calculation

    NCD is calculated as a measure of similarity between pairs of texts. The calculate_ncd function computes the NCD between two input texts. The calculate_train_ncd and calculate_test_ncd functions use multiprocessing to efficiently compute the NCD matrix for training and testing samples.

###    K-Nearest Neighbors (KNN) Classification

    The K-Nearest Neighbors algorithm is used for sentiment classification. The training NCD array is used to fit the KNN model. Then, the NCD array of the test samples is used to predict sentiment labels using the trained KNN model.

###    Accuracy Calculation

    The accuracy of the sentiment classification is calculated using the accuracy_score function from scikit-learn. The predicted labels are compared to the actual test labels, and the accuracy is printed.

## Usage

    Ensure you have the required dependencies installed.
    Modify the n_samples variable to adjust the number of samples to process.
    Replace the file path in pd.read_csv() with the actual path to your CSV file.

## Notes

    The accuracy of the model may vary based on the dataset, preprocessing, and parameters chosen.
    You can experiment with different values of n_neighbors in the KNN classifier for potentially better results.
