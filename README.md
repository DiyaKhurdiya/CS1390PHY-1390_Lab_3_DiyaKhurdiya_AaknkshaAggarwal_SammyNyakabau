# MNIST Digit Recognition by various ML Classifiers and comparing their accuracy:
1. Logistic Regression - 85% accuracy
2. Naive Bayes - 81% accuracy
3. SVM - 83% accuracy
4. K-Nearest Neighbours - 94% accuracy

Project Report:
OBSERVATIONS AND FINAL REPORT
Members: Aakanksha Agrawal, Diya Khurdiya, Sammy Nyakabau

Handwritten digit recognition is the ability of a computer to recognize the human handwritten digits from different sources like images, papers, touch screens, etc, and classify them into 10 predefined classes (0-9)

The plotting of random MNSIT dataset:

<img width="252" alt="Screenshot 2022-12-16 114316" src="https://user-images.githubusercontent.com/79498434/208034380-40b7e1d7-0f52-40b9-9f5c-6ca41e6b0a3d.png">

Frequency plot of input labels:
<img width="310" alt="Screenshot 2022-12-16 114350" src="https://user-images.githubusercontent.com/79498434/208034464-f534bbd5-7ffb-4959-b86d-371fd79df1c0.png">


# Logistic Regression
Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables. It supports categorizing data into discrete classes by studying the relationship from a given set of labelled data.  

Accuracy: 85%
Limitations: 
The major limitation of Logistic Regression is the assumption of linearity between the dependent variable and the independent variables

Advantages: 
1. Easier to implement, train and interpret.
2. Easily expandable to multiple classes and provides a normalized probability distribution.
3. Measures of how accurate a predictor is and it’s direction of association(positive/negative)
4. Decent accuracy for simple datasets and performs well when data is linearly separable i.e. when data graphed in two dimensions can be separated through a straight line.
5. It can overfit in high dimensional datasets then we can use regularization technique to avoid overfitting (low training error and high testing error)
Confusion Matrix is a performance measurement for machine learning classification 

Misclassification might occur if the model is too simple or the data is very noisy (data not so stable with high variability). The multiple classes might not be linearly separable. The problem of overfitting (model fits exactly against its training data but algorithm unfortunately cannot perform accurately against unseen data, defeating its purpose) might be another factor. 

Accuracy and performance can be used by better scaling and pre-processing techniques. Using a different set of optimal values for the hyperparameters such that it tunes perfectly. Increasing the training data sample can also contribute to increasing the accuracy.
                 <img width="404" alt="Screenshot 2022-12-16 114423" src="https://user-images.githubusercontent.com/79498434/208034525-d0043ce9-5773-4ce1-91e7-30a9cd2d6adf.png">

<img width="376" alt="Screenshot 2022-12-16 114500" src="https://user-images.githubusercontent.com/79498434/208034603-5536cf90-acd9-46a4-985d-a3b32dbee1fb.png">


# SVM
Given labeled training data the algorithm outputs the best hyperplane which classified new examples. In two-dimensional space, this hyperplane is a line splitting a plane into two parts where each class lies on either side. The intention of the support vector machine algorithm is to find a hyperplane in an N-dimensional space that separately classifies the data points.

Accuracy: 81%
Limitations: 
1. Long training time for large datasets
2. Since the final model is not so easy to see, we can not do small calibrations to the model hence its tough to incorporate our business logic

Advantages: 
1. Memory efficient
2. SVM works relatively well when there is a clear margin of separation between classes
3. more effective in high dimensional spaces
4. Works well with even unstructured and semi structured data like text and images

There are many reasons to prefer each type of classifier over others in different circumstances (e.g. time/memory required for training/evaluation, amount of tweaking/exploration required to get a decent working model, etc.). There are certainly domain-specific tricks than can make classifiers more suitable for digit recognition. Some of these tricks work by increasing invariance to particular transformations that one would expect in handwritten digits (e.g. translation, rotation, scaling, deformation).

 
<img width="385" alt="Screenshot 2022-12-16 114529" src="https://user-images.githubusercontent.com/79498434/208034660-6b41dab0-314f-4db6-b9ec-1f825e0c6a36.png">


Misclassification might have occurred due to less regularization of data. By adding a penalty to the cost function, overfitting can be discouraged. Also to improve the accuracy, the technique of cross-validation can be used that involves splitting your data into multiple sets, and then training your model on one set of data and testing it on another set of data. This helps to ensure that your model does not overfit to the data that it was trained on.

# Naive Bayes
Naive Bayes classifiers are a collection of classification algorithms based on Bayes’ Theorem. It relies on the common principle that every pair of features being classified is independent of each other. 
The formula for Bayes' theorem is given as:
Where,

P(A|B) is Posterior probability: Probability of hypothesis A on the observed event B.

P(B|A) is Likelihood probability: Probability of the evidence given that the probability of a hypothesis is true.

Accuracy: 83%
Limitations: The algorithm assumes that all features are independent or unrelated, so it cannot learn the relationship between features.

Advantages:
1. Naïve Bayes Classifier is one of the simple and most effective Classification algorithms which helps in building the fast machine learning models that can make quick predictions
2. It is suitable for both binary as well as multi-level classification.
3. It is highly scalable with the number of predictors and data points.
    

Accuracy can be improved by reducing the complexity of the model by removing noisy features; For unregularized models, you can use feature selection or feature extraction techniques to decrease the number of features.

<img width="412" alt="Screenshot 2022-12-16 114601" src="https://user-images.githubusercontent.com/79498434/208034729-ce09f081-cda6-4bac-b1b5-5a6aef15039e.png">

<img width="275" alt="Screenshot 2022-12-16 114625" src="https://user-images.githubusercontent.com/79498434/208034765-39c18f39-daeb-45cd-8e62-bd05cfd81460.png">


# KNN
The k-nearest neighbors algorithm, also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point. While it can be used for either regression or classification problems, it is typically used as a classification algorithm, working off the assumption that similar points can be found near one another.

Accuracy: 94%
Limitations:
1. Does not scale well: It takes up more memory and data storage compared to other classifiers. Costly and time consuming
2. Curse of dimensionality: It doesn’t perform well with high-dimensional data inputs. Additional features increase the amount of classification errors, especially when the sample size is smaller.
3. Prone to overfitting: While feature selection and dimensionality reduction techniques are leveraged to prevent this from occurring, the value of k can also impact the model’s behavior. Lower values of k can overfit the data, whereas higher values of k tend to “smooth out” the prediction values since it is averaging the values over a greater area, or neighborhood. However, if the value of k is too high, then it can underfit the data.

Advantages:
1. Easily implemented
2. Easily adaptable: Automatically adjusts for new data to be stored when new set of observations are added
3. Few hyperparameters: KNN only requires a k value and a distance metric, which is low when compared to other machine learning algorithms.

Accuracy can be further improved by using algorithmic tuning that is by parameter tuning. The objective of parameter tuning is to find the optimum value for each parameter to improve the accuracy of the model.


<img width="422" alt="Screenshot 2022-12-16 114644" src="https://user-images.githubusercontent.com/79498434/208034819-bf55277b-e524-4320-94b4-7fb3c32ef8a1.png">

<img width="203" alt="Screenshot 2022-12-16 114754" src="https://user-images.githubusercontent.com/79498434/208035101-d1fcd57a-0940-48bd-ae4a-57d9e5d808fd.png">


 
As we vary the value of K-Nearest neighbors, the training and testing dataset accuracy score is calculated and thus printed as the accuracy graph comparing training and testing accuracy score. We see here that for the training dataset with increase in KNN, accuracy drops while it is the other way for Testing set indicating a significant improvement.

The confusion matrices for all 4 algorithms provides us the value if true positives, true negatives, false positives and false negatives. By comparing two confusion matrices, we can determine the true positive value let’s say for the number 1 is 905 (KNN), 857 (Naive Bayes), 580 (SVM) and 786 (Logistic) that means KNN gives the most number of true 1’s for label 1 and hence more accurate. Similarly we can calculate other parameters for different labels thereby getting a sense of the accuracy score, precision and recall.

# Bias Variance tradeoff:
Bias is the difference between the average prediction of our model and the correct value which we are trying to predict.
Variance is the variability of model prediction for a given data point or a value which tells us spread of our data.
The algorithms used above all are supervised learning algorithms since the output label data has already been provided to us and we need to classify based on that. Hence most of these algorithms face the limitation of overfitting that happens when our model captures the noise along with the underlying pattern in data. It happens when we train our model a lot over noisy datasets. These models have low bias and high variance. Learning curves give us an opportunity to diagnose bias and variance in supervised learning models.
If the training error is high, it means that the training data is not fitted well enough by the estimated model. If the model fails to fit the training data well, it means it has high bias with respect to that set of data.
A narrow gap indicates low variance. Generally, the more narrow the gap, the lower the variance. The opposite is also true: the wider the gap, the greater the variance.

We should always choose hyperparameters so that both bias and variance are as low as possible and more training dataset.
Total Error = Bias + Variance + Irreducible Error


# Result:
Out of the four algorithms (Naive Bayes, Logistic Regression, SVM and KNN) KNN outperformed with an accuracy of about 94%.
Naive Bayes and KNN took comparatively less time.


# Learnings of ML:
1. Some algorithms work better for certain datasets and not for others. There is no one size fits all
2. Overfitting and underfitting is a huge issue that needs to be be tacked well
3. When in doubt - normalize. Most issues can be solved by feature rescaling
4. Optimal and accurate parameter tuning can increase accuracy by bounds. Careful study and understanding of the model can lead us to predict the right    hyperparameters.



# Resources used:
https://www.ibm.com/in-en/topics/knn
https://arxiv.org/pdf/2106.12614.pdf
https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229
https://statinfer.com/204-6-8-svm-advantages-disadvantages-applications/#:~:text=SVM%20Disadvantages&text=Long%20training%20time%20for%20large,to%20incorporate%20our%20business%20logic.
https://vitalflux.com/learning-curves-explained-python-sklearn-example/
https://zahidhasan.github.io/2020/10/13/bias-variance-trade-off-and-learning-curve.html

