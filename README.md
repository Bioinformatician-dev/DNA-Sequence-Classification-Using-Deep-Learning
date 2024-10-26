# DNA-Sequence-Classification-Using-Deep-Learning
Classify DNA sequences as "cancerous" or "non-cancerous" using a deep learning model.
## Installation
```bash
  pip install deepbio
```
## Creating the CSV File
create the CSV file manually using a text editor or programmatically using Python follow

```bash
    file.py
```
## Step
* Data Creation: 
We create a synthetic dataset of DNA sequences along with labels indicating whether they are cancerous or non-cancerous.

* One-Hot Encoding:
  The one_hot_encode function converts DNA sequences into a one-hot encoded format suitable for input to the neural network.

* Data Splitting: 
We split the dataset into training and test sets.

* Model Building: 
We build a simple LSTM model using Keras for classifying the sequences.

* Model Training: 
The model is trained on the training dataset for 50 epochs.

* Model Evaluation: 
We evaluate the model's performance on the test dataset.

Predictions: 
* Finally, we use the trained model to make predictions on new sequences.
