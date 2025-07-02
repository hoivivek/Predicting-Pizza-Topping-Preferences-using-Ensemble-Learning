# Predicting Pizza Topping Preferences Using Ensemble Learning

## Project Overview

This project explores the application of machine learning, specifically ensemble learning techniques, to predict pizza topping preferences. Instead of directly asking for topping preferences, the study leverages indirect personal interest questions (e.g., chess piece preferences, pen types, favorite sports, belief in ghosts, and favorite games) as input features. The goal is to enhance prediction accuracy and robustness in multi-label classification tasks.

The project successfully demonstrates the effectiveness of ensemble learning methods in delivering high predictive accuracy, stability, and robustness by strategically combining the strengths of different classifiers.

## Methodologies

### Data Collection
A dataset of 19 student responses was collected, capturing personal preferences through indirect questions. The target labels were multi-label pizza topping preferences.

### Data Preprocessing
* Categorical input features were converted into numerical values.
* Multi-label target values (pizza toppings) were transformed using `MultiLabelBinarizer` to convert the toppings column into a binary format, allowing each topping to be treated as a separate category.
* The dataset was split into training and testing sets using Stratified K-Fold Cross-Validation (8-fold) to ensure balanced class distribution across folds.

### Model Implementation

Four key ensemble learning techniques were implemented and evaluated:

1.  **Bagging (Bootstrap Aggregating)**:
    * **Concept**: Reduces variance by training multiple models on random subsets (bootstrapped samples) of the dataset and averaging their predictions.
    * **Implementation**: `BaggingClassifier` with `DecisionTreeClassifier` as the base learner. Hyperparameter tuning was performed using a grid search for tree depths (5, 10, 15, 20) and estimators (5, 10, 15, 20). The final configuration used 5 Decision Trees with a maximum depth of 5.

2.  **Boosting**:
    * **Concept**: Sequentially trains weak learners, with each model focusing on correcting the mistakes of the previous one by adjusting the weight of misclassified instances.
    * **Implementation**: `AdaBoostClassifier` and `GradientBoostingClassifier`.
        * **AdaBoost**: Tuned for estimators (10, 20, 30) and learning rates (0.01, 0.1, 0.5), using a `DecisionTreeClassifier` with a max depth of 2.
        * **Gradient Boosting**: Optimized for estimators (50, 100, 150), learning rates (0.01, 0.1, 0.2), and tree depths (2, 3, 4). A `GradientBoostingClassifier` with 20 estimators, a learning rate of 0.1, and a max depth of 2 was used.
    * Predictions from AdaBoost and Gradient Boosting were combined by averaging their outputs.

3.  **Stacking**:
    * **Concept**: A layered approach where multiple base models make predictions, and a meta-model learns how to best combine these predictions to produce the final decision.
    * **Implementation**: `StackingClassifier` with `DecisionTree` and `K-Nearest Neighbors (KNN)` as base models. `Logistic Regression` was used as the meta-model.
    * **Hyperparameter Tuning**:
        * Decision Tree: Depths (10, 15, 20) and min sample splits (2, 5, 10). Final configuration: max depth of 5.
        * KNN: Number of neighbors (2, 3, 5). Final configuration: 5 neighbors with distance-based weighting.
        * Logistic Regression: Regularization strengths (C values: 0.01, 0.1, 1, 10).

4.  **Voting Classifier**:
    * **Concept**: Combines multiple models and makes predictions based on majority agreement (hard voting).
    * **Implementation**: `VotingClassifier` with `RandomForestClassifier` and `GradientBoostingClassifier` as base models, using hard voting.
    * **Hyperparameter Tuning**:
        * Random Forest: Estimators (50, 100, 150), max depths (5, 7, 10), min samples split (2, 5, 10). Final configuration: 150 estimators, max depth of 5, random state 42.
        * Gradient Boosting: Estimators (50, 100, 150), learning rates (0.01, 0.1, 0.2), max depths (3, 5, 7). Final configuration: 150 estimators, learning rate of 0.01, max depth of 7, random state 42.

### Evaluation Metrics
Accuracy was used to evaluate model performance, reflecting how frequently a model's predictions are correct.

## Results

The following accuracies were achieved by each ensemble method:

| Ensemble Method    | Accuracy |
| :----------------- | :------- |
| Stacking           | 69.5%    |
| Boosting           | 65.4%    |
| Bagging            | 64.2%    |
| Voting             | 63.7%    |

**Analysis of Results:**

* **Stacking** achieved the highest accuracy, attributed to its ability to combine diverse algorithms through a meta-learner, effectively reducing both bias and variance. However, it did not meet the expected performance, potentially due to limited diversity among base models, overfitting in the meta-model, or inadequate feature representation.
* **Boosting** outperformed Bagging and Voting, demonstrating its strength in reducing bias by focusing on misclassified instances. Possible reasons for not achieving higher accuracy include overfitting from excessive iterations, noise in the dataset, or insufficient hyperparameter tuning.
* **Bagging** provided moderate accuracy, effectively reducing variance for high-variance models. Its lower accuracy might be due to limited diversity among base models, underfitting from weak learners, or insufficient data for effective bootstrapping.
* **Voting** achieved the lowest accuracy, highlighting its simplicity and limitations in leveraging the relative strengths of different models. This could be due to equal weighting of models or insufficient diversity among them.

**Key Observations:**
* **Mushrooms Dominance**: "Mushrooms" consistently appeared in all model outputs, suggesting it is the most influential and consistent feature for the given inputs.
* **Diversity in Boosting and Stacking**: These models showed more variety in outputs, indicating their iterative error correction and leveraging of diverse base models.
* **Stability in Bagging and Voting**: These models demonstrated more stability and less diversity, reaffirming "Mushrooms" as a dominant and consistent feature.

## Conclusion

This project successfully demonstrates the power of ensemble learning techniques in improving predictive accuracy and robustness in classification tasks. By leveraging multiple base models and combining their predictions, the ensemble methods effectively reduced bias and variance, leading to more reliable and consistent outputs.

The consistent preference for "Mushrooms" across models highlights the models' ability to identify strong patterns, reaffirming the value of ensemble learning in making accurate and reliable predictions. While challenges related to complexity, interpretability, and computational resources exist, the project validates the effectiveness of ensemble methods in delivering high predictive accuracy, stability, and robustness, making them a powerful solution for complex classification problems.

The modular design allows for easy integration of new models and features, enhancing scalability and adaptability for future projects, including potential live prediction capabilities.

## Technologies and Libraries Used

* **Python**: The primary programming language for implementation.
* **Scikit-learn**: For implementing ensemble learning algorithms (`BaggingClassifier`, `AdaBoostClassifier`, `GradientBoostingClassifier`, `StackingClassifier`, `VotingClassifier`, `DecisionTreeClassifier`, `KNeighborsClassifier`, `LogisticRegression`, `RandomForestClassifier`) and data preprocessing (`MultiLabelBinarizer`, `StratifiedKFold`).
* **NumPy**: For numerical operations.
* **Pandas**: For data manipulation and analysis.
* **Matplotlib / Seaborn**: (Assumed for visualizations in the report, though not explicitly mentioned for the code) For data visualization.

