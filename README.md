# DNA-Sequence-Classification-Using-Deep-Learning
In this project, we are going to classify DNA sequences using deep learning model. The model is designed to predict labels from input DNA sequences with LSTM( Long Short-Term Memory) networks. It is especially helpful for bioinformatics, e.g. in Genomics and mutation classifications etc.

## Table of Contents

* Features

 * Libraries Used

 * Dataset

 * Installation

 * Usage

 * Training

 * Evaluation

 * Results

 * Contributing

 ## Features
  * DNA Sequence Encoding: Converts DNA sequences into numerical format for model training.
  * Architecture: Utilizes an embedding layer and LSTM layers for sequential data processing.
  * Model Training and Evaluation: Implements early stopping and model checkpointing for optimal training.
  * Visualization: Provides visual insights into training and validation accuracy/loss.

## Libraries Used

* TensorFlow/Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

## Dataset
The dataset used in this project consists of DNA sequences and their corresponding labels. The CSV file format includes two columns: sequence and label.
```bash
     dna_sequences.csv
```
## Installation
To set up the project, clone the repository and install the required packages:
```bash
  git clone https://github.com/Bioinformatician-dev/DNA-Sequence-Classification-Using-Deep-Learning.git
  cd DNA-Sequence-Classification-Using-Deep-Learning
  pip  install -r requirements.txt
```
## Creating the CSV File
create the CSV file manually using a text editor or programmatically using Python follow

```bash
    file.py
```
## Usage
To run the model, make sure you have the dataset (dna_sequences.csv) in the project directory. You can then execute the script:

```bash
  python  classifying.py
```
## Training
The model is trained with the following configuration:

* Epochs: 50
* Batch Size: 32
* Early Stopping: Monitors validation loss to prevent overfitting.
## Evaluation
The model is evaluated on a test set with metrics including accuracy, precision, recall, and F1-score. The results are displayed in a classification report.

# Results
After training, the model's accuracy and loss curves are plotted for both training and validation datasets. A confusion matrix is also generated to visualize the classification performance.

## Contributing
Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

