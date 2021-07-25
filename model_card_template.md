# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

Training a logistic classification model to predict the income category of a person based on other personal informations.

## Model Details

A basic logistic regression classifier imported from scikit-learn library and fit onto the census data

## Intended Use

This model can be used to predict and get the idea of a person's income range which can used to identify if a person qaulifies for certain facilities or services like loan approval etc.

## Training Data

The training data is the census data available at the UCI library. It is the adult.income data from the data folder.
<br />
Link: [UCI Census Data](https://archive.ics.uci.edu/ml/datasets/census+income)

## Evaluation Data

Evaluation data is splitted from the training set itself. The split was done in 80:20 ratio.

## Metrics

Metrices to evaluate the model performance were:

- Precision
- Recall
- F-beta score

## Ethical Considerations

The UCI should be given proper credit and should be mentioned if model is publicly.

## Caveats and Recommendations

The data had whitespaces leading to multiple false unique categories. So make sure to go through the data cleaning process to remove all those whitespaces.
