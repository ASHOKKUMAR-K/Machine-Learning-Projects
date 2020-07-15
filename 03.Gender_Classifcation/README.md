<h1 align="center">Gender Classification</h1>

## Table of contents
- [Demo](#demo)
- [Introduction](#introduction)
- [Motivation](#motivation)
- [Project Explanation](#project-explanation)
  - [Data Collection](#data-collection)
  - [Feature Engineering](#feature-engineering)
  - [Data Pre-processing](#data-pre-processing)
  - [Feature Observations](#feature-observations)
  - [Building Machine Learning Model](#building-machine-learning-model)
  - [Model Performance](#model-performance)
- [Conclusion](#conclusion)

## Demo

**App Link** : [Gender Classification](https://ashok-machine-learning-project.herokuapp.com/#gender)

![Gender Classifier](../Demos/gender.gif)

## Introduction
- This dataset consists of three columns which is Height, Weight and Gender.
- In this project we are going to classify person's gender based on his/her height and weight.
- Our goal is to train a model with the given data to classify gender.
- Let's look more into the data

<div align="center"><img src="../static/images/photos/gender.png" height="300" weight="300"></div>

## Motivation
- This is my second end to end data science project.
- I have done this project to stay motivated on Data Science.
- In this project, I have learnt about classification kind of problems.
- I done this project to learn, how the classification works.

## Project Explanation
### Data Collection
- The dataset is collected from [kaggle](https://www.kaggle.com/mustafaali96/weight-height).
- The dataset consists of a CSV file name `weight-height.csv`.
Let's load the collected data.

```python
# Loading Dataset
wh = pd.read_csv('../data/weight-height.csv')
```
- The dataset consists of 3 features i.e., Height, Weight and Gender.
- The dataset consists of 10,000 samples of different persons.

### Feature Engineering

**1. Height**

- The heights are mentioned in inches.
- There are no null values in the Height column.
- Let's see the distribution of height data.

![Height Distribution](images/height_distribution.png)

- Here we get a well distributed data and a bell shaped curve shows that it is apt for analysis.

**2. Weight**

- The weights are mentioned in pounds.
- There are no null values in the Weight column.
- Let's see the distribution of weight data.

![Weight Distribution](images/weight_distribution.png)

- Here we get a well distributed data and a bell shaped curve shows that it is apt for analysis.

**3. Gender**

- Gender is a **nominal** kind of **qualitative data**, because there is no numerical relation between different genders.
- Let's see the relation of Gender with height and weight.

![Height Gender](images/gender_height.png) ![Weight Distribution](images/gender_weight.png)

### Data pre processing
The data which we collect from kaggle is already well pre-processed. So no need of pre-processing.

### Feature Observations
Let's see the correlation of features with a heatmap.

![Feature Correlation](images/feature-correlaion.png)
![Feature Correlation](images/feature-relations.png)

### Building Machine Learning Model

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = wh.iloc[:, [1, 2]].values
y = wh.iloc[:, 0].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability = True))
clf.fit(X_train, y_train)
```

```python
# Let's see the training and testing accuracy
print("Training Score : ", clf.score(X_train, y_train) * 100)
print("Training Score : ", clf.score(X_test, y_test) * 100)

OUTPUT :
Training Score :  91.47 %
Testing Score  :  91.90 %
```

### Model Performance

```python
# Let's see the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, clf.predict(X_test)))

REPORT :
              precision    recall  f1-score   support

      Female       0.92      0.91      0.92      1463
        Male       0.92      0.92      0.92      1537

    accuracy                           0.92      3000
   macro avg       0.92      0.92      0.92      3000
weighted avg       0.92      0.92      0.92      3000
```

## Conclusion
- Finally we finished our gender classification project.
- We have built a Support Vector Machine Classifier which performs well with given features and having the Training accuracy of 91.47% and Testing accuracy of 91.90%.

##### If you love my work, kindly give a Star to my repository.
##### Facing any issues, feel free to post your issues on my GitHub.
##### Thank you Everyone for seeing my work.
