# Supervised Learning Finals Study Guide

Course: Introduction to Machine Learning: Supervised Learning (DTSA 5509), CU Boulder MS in Data Science on Coursera.
One section per exam topic, following the five course modules. Each section ends with practice questions. Answers stay hidden until you click.

---

## 1. Supervised learning and simple linear regression

**Key ideas**
- Supervised learning uses labelled data: predictors X and a response Y. We estimate f in Y = f(X) + ε.
- Regression predicts a number. Classification predicts a category.
- Unsupervised learning has no response. It looks for structure such as clusters.
- Error splits into reducible error (a better model can shrink it) and irreducible error (the noise ε, which no model removes).
- Train on one part of the data and judge the model on data it has not seen. Training error almost always looks better than test error.
- Simple linear regression: Y ≈ β0 + β1X. Least squares picks β0 and β1 that minimise the residual sum of squares (RSS).
- Slope estimate: β1 = Σ(xi − x̄)(yi − ȳ) / Σ(xi − x̄)². Intercept: β0 = ȳ − β1x̄.
- A t statistic of β1 divided by its standard error tests whether the slope is zero. A small p value means the predictor matters.
- R² is the share of variance in Y the model explains. It runs from 0 to 1.

**Practice questions**
1. A model predicts next month's rent in dollars. Regression or classification?
   > Answer: Regression, because the output is a number.
2. Why can training MSE fall while test MSE rises?
   > Answer: The model starts fitting noise in the training set (overfitting), so it generalises worse.
3. Which part of the error can no model remove?
   > Answer: The irreducible error, the variance of ε.

---

## 2. Multiple linear regression for prediction and inference

**Key ideas**
- Y ≈ β0 + β1X1 + ... + βpXp. Each βj is a partial effect: the change in Y for a one unit change in Xj, holding the other predictors fixed.
- A categorical predictor with k levels enters as k − 1 dummy variables. The left out level is the baseline.
- An interaction term X1·X2 lets the effect of X1 depend on X2. Keep the main effects when you include the interaction (hierarchy principle).
- R² never goes down when you add a predictor, even a useless one. Adjusted R² and validation error correct for that.
- RMSE is the square root of mean squared error, in the same units as Y.
- More predictors can mean lower bias but higher variance. Check on held out data.

**Practice questions**
1. A dummy for "Toronto" has coefficient 120 in a rent model with baseline "Ottawa". What does 120 mean?
   > Answer: Holding the other predictors fixed, predicted rent is 120 higher for Toronto than for Ottawa.
2. You add a column of random numbers and R² rises slightly. Is the model better?
   > Answer: No. R² cannot fall when a predictor is added. Look at adjusted R² or test error.
3. What does a significant interaction between income and age tell you?
   > Answer: The effect of income on Y changes with age, so neither effect can be read alone.

---

## 3. Classification: logistic regression, discriminant analysis and KNN

**Key ideas**
- Logistic regression models the log odds: log(p / (1 − p)) = β0 + β1X. So p = 1 / (1 + e^−(β0 + β1X)), always between 0 and 1.
- e^β1 is the odds ratio for a one unit increase in X.
- Coefficients are fit by maximum likelihood, not least squares.
- A threshold (often 0.5) turns probabilities into classes. Moving it trades false positives for false negatives.
- LDA assumes each class is normal with a shared covariance matrix, which gives a linear boundary. QDA lets each class have its own covariance, which gives a curved boundary.
- KNN classifies a point by majority vote of its K nearest neighbours. Small K means low bias and high variance. Large K means smoother boundaries. Scale features first, because distance depends on units.
- Confusion matrix terms: precision = TP / (TP + FP). Recall = TP / (TP + FN). Accuracy can mislead when one class is rare.

**Practice questions**
1. In a loan default model, β1 for debt to income is 0.7. What is the odds ratio?
   > Answer: e^0.7 ≈ 2.01. Each one unit rise roughly doubles the odds of default.
2. Defaults are 2% of the data. A model that always predicts "no default" has 98% accuracy. What is wrong?
   > Answer: Its recall for defaults is 0. Use recall, precision or AUC for rare classes.
3. When would you pick QDA over LDA?
   > Answer: When classes clearly have different spreads and you have enough data to estimate a covariance matrix for each class.

---

## 4. Model evaluation, resampling and regularisation

**Key ideas**
- A single validation split is cheap but noisy, and it wastes data.
- k fold cross validation: split into k folds, train on k − 1, test on the one left out, average the k scores. k = 5 or 10 is common.
- Leave one out CV is k fold with k = n. Low bias, high variance, expensive for large n.
- The bootstrap resamples n rows with replacement to estimate the variability of a statistic. About 63.2% of the original rows appear in each bootstrap sample.
- Ridge regression adds λΣβj² to RSS. It shrinks coefficients toward zero but keeps all of them.
- Lasso adds λΣ|βj|. It can set some coefficients exactly to zero, so it also selects variables.
- Standardise predictors before ridge or lasso, because the penalty depends on scale.
- Choose λ by cross validation. λ = 0 is plain least squares. A very large λ pushes coefficients toward zero.

**Practice questions**
1. You need a model that keeps only a handful of predictors. Ridge or lasso?
   > Answer: Lasso, because the L1 penalty can drive coefficients to exactly zero.
2. Why standardise before fitting ridge?
   > Answer: The penalty treats every coefficient the same, so a predictor measured in small units would be penalised unfairly.
3. About what share of rows is left out of one bootstrap sample?
   > Answer: About 36.8%. These out of bag rows can be used to test.

---

## 5. Tree based methods and ensembles

**Key ideas**
- A decision tree splits the feature space into boxes with recursive binary splits. Each split is chosen greedily.
- Regression trees pick splits that cut RSS the most and predict the mean of each leaf.
- Classification trees use Gini impurity or entropy to choose splits and predict the majority class.
- Deep trees overfit. Cost complexity pruning grows a large tree, then cuts it back using a penalty α chosen by cross validation.
- Bagging fits many trees on bootstrap samples and averages them. That lowers variance. Out of bag error gives a free test estimate.
- Random forests also sample a subset of m predictors at each split (often about √p for classification), which decorrelates the trees.
- Boosting grows small trees one after another, each fit to what the current model still gets wrong, scaled by a learning rate. Too many trees can overfit, so tune the number of trees by cross validation.

**Practice questions**
1. Why does a random forest usually beat bagging when one predictor is very strong?
   > Answer: In bagging most trees split on that predictor first, so they are correlated. Random subsets of predictors make the trees more different, and averaging them cuts more variance.
2. What two settings matter most in boosting?
   > Answer: The learning rate and the number of trees (plus tree depth). A smaller learning rate needs more trees.
3. What is out of bag error?
   > Answer: The error on rows that a tree did not see in its bootstrap sample, averaged across trees.

---

## Cheat sheet

| Method | Output | Watch out for |
| --- | --- | --- |
| Linear regression | Number | Outliers, collinearity, nonlinearity |
| Logistic regression | Probability of a class | Rare classes, threshold choice |
| LDA and QDA | Class | Normal class assumption |
| KNN | Class or number | Feature scaling, choice of K |
| Ridge | Number, all predictors kept | Must standardise |
| Lasso | Number, some predictors dropped | Must standardise |
| Tree | Class or number | Overfits unless pruned |
| Random forest | Class or number | Less interpretable |
| Boosting | Class or number | Tune learning rate and tree count |

Formulas to know: precision = TP / (TP + FP); recall = TP / (TP + FN); odds ratio = e^β; ridge penalty λΣβj²; lasso penalty λΣ|βj|.

---

## Diagram: the bias and variance trade off

As model flexibility rises (smaller K in KNN, deeper trees, smaller λ), training error keeps falling, while test error falls and then rises again. The best model sits at the bottom of the test error curve. Cross validation is how you find that point without touching the test set.

Sources checked: course page https://www.colorado.edu/program/data-science/coursera/curriculum/dtsa5509 and https://www.coursera.org/learn/introduction-to-machine-learning-supervised-learning
