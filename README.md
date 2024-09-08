# DSLR
#### The project is an introduction to Logistic Regression For Multiclass Classification Using One-vs-Rest(One-vs-All) Strategy.

Hogwart's Magic Hat lost its powers, we are here to save the day as data scientists by creating a Logistic Regression Model that classifies Hogwarts Students to the Hogwart House they belong to.

## Sumarry
- ### Data Analysis
    First of all, take a look at the available data. look in what format it is presented, if there are various types of data, the different ranges, and so on. It is important to make an idea of your raw material before starting. The more you work on data - the more you develop an intuition about how you will be able to use it. 

    For that we are reimplemting the ```pandas.DataFrame.describe``` function ([describe.py](https://github.com/EniddeallA/DSLR/blob/main/describe.py)), to display information for all numerical features, and explore the data set.

- ### Data Visualization
    Data visualization is a powerful tool for a data scientist. It allows you to make insights and develop an intuition of what your data looks like. Visualizing your data also allows you to detect defects or anomalies.

    For that we are using different visualization methods, each answering a particular question.

    - #### Which Hogwarts course has a homogeneous score distribution between all four houses?
        - [Histogram](https://github.com/EniddeallA/DSLR/blob/main/histogram.ipynb)

    - #### What are the two features that are similar?
        - [Scatter Plot](https://github.com/EniddeallA/DSLR/blob/main/scatter_plot.ipynb)

    - #### What features to use for our logistic regression?
        - [Pair Plot / Scatter Plot Matrix](https://github.com/EniddeallA/DSLR/blob/main/pair_plot.ipynb)

- ### Classification
    Coding the Magic Hat starts now, we are performing a Multi-Classifier using One-Vs-All Logistic Regression Method.
    
    - #### Logistic Regression
    Logistic regression’s output lies between 0 and 1 as the algorithm is designed to predict a binary outcome for an event based on the previous observations of a data set. It uses independent variables to predict the occurrence or failure of specific events.

  ![Logistic Regression](https://github.com/EniddeallA/DSLR/blob/main/Logistic_Regression.png)

    - #### One-Vs-All 
    For each class, build a logistic regression to find the probability the observation belongs to that class.
    For each data point, predict the class with the highest probability.

  ![One-Vs-ALL](https://github.com/EniddeallA/DSLR/blob/main/One-Vs-Rest.png)

    - #### Mathematics
    Logistic regression works almost like the linear regression. Here is a cost (loss) function:
    ```math
      J(θ) = −1/m \sum_{i=1}^{n} y^i log(hθ(x^i )) + (1 − y^i )log(1 − hθ(x^i ))
    ```
    Where hθ(x) is defined in the following way :
    ```math
      h_{θ}(x) = g(θ^T x)
    ```
    With :
    ```math
      g(z) = 1 /{1 + e^{-z}}
    ```
    The loss function gives us the following partial derivative :
    ```math
      ∂/∂θ_{j} J(θ) = 1/m \sum_{i=1}^{m}(h_θ(x^i ) − y^i )x^i_j
    ```

## Usage
  Running the [logreg_train.py](https://github.com/EniddeallA/DSLR/blob/main/logreg_train.py) will train the logistic regression using Gradient Descent and outputs ```predictData.json``` file containing the weights that will be used for the prediction.
  ```bash
    python logreg_tain.py /path/to/trainingDataset
  ```
  Adding a second argument ```--bonus``` will train the logistic regression using ```Stochastic Gradient Descent``` and ```Batch GD```
  ```bash
    python logreg_tain.py /path/to/trainingDataset --bonus
  ```
  Finally run [logreg_predict.py](https://github.com/EniddeallA/DSLR/blob/main/logreg_predict.py) and enjoy a 99% accuracy model.
  ```bash
    python logreg_predict.py /path/to/TestDataset predictData.json
  ```
