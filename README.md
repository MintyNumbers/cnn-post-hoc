# CNN Post-Hoc Analysis

Post-Hoc Analysis of a CNN.

## Description

This project is designed to analyse a CNN's performance using LIME and SHAP. The CNN is used to classify bark images as their respective tree's species. The analysis of the best performing model are found under ```results/results.md```.

## Getting Started

### Dependencies

The required dependencies are listed in ```requirements.txt```. First you can create a Virtual Environment in the .venv folder of the project, source it and update pip to newest version:

```bash
# Unix, macOS
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip --version
# OR: Windows
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
py -m pip --version
```

You can download them by executing the following command using your shell of choice:

```bash
# Unix, macOS
python3 -m pip install -r requirements.txt
# OR: Windows
py -m pip install -r requirements.txt
```

### Installing

* In a directory of your choosing you can either:
    * clone the repository using ```git clone```, or
    * unzip the project archive
* Then you need to download the dependencies like described in the previous chapter
* And finally, download the dataset from: https://www.kaggle.com/datasets/saurabhshahane/barkvn50

### Executing program

To run the whole program you can use the Jupyter Notebook ```train_cnn.ipynb```. It contains code to:
* perform the Train/Test split on the downloaded dataset
* train a CNN with specific hyperparameters, either with the ```Adam``` or ```SGD``` optimizer
* train a CNN using augmented data
* train a single CNN with all of the training data or use the K-Fold Cross Validation Training method
* evaluate the trained CNN model with:
    * a confusion matrix,
    * the LIME method,
    * SHAP values,
    * the activations of the Fully Connected output layer (with and without Softmax) and the two convolutional layers of the model, and
    * the maximized activations of the two ReLU layers
