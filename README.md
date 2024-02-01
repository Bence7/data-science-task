# **Binary sentiment classification for movie reviews**

Welcome to `data-science-task` Machine Learning project. The goal of this project to create a model, that can predict if a review is positive or negative on movie reviews on high accuracy.

# **DS part**
## **Feature Engineering**

I encoded the sentiment feature, which originally had only positive or negative values, by assigning the label "positive" to the numerical value 1 and the label "negative" to the numerical value 0.

On the reviews feature I had to apply a preprocessing pipeline. 
The provided `preprocess_text` function conducts several essential text preprocessing steps to prepare the data for further analysis and modeling. Here are the key conclusions regarding the data preprocessing steps:

1. The text contains HTML tags, like `<br>`, so I needed to apply a removal function `clean_html_tags`.
2. Next, I removed the punctuations and the numbers.
3. Later, I lower the letters before tokenizing.
4. Tokenization, break the text into individual words.
5. Removing common english stop words.
6. The lemmatization process is applied using the WordNet lemmatizer, reducing words to their base or root form.
7. Removing short tokens, that length is smaller than 3.
8. After these steps, the tokens are joined back into a text.

## **Vectorizing**
I had to implement vectorization before employing any modeling algorithms. When comparing two techniques, it was observed that the TfidfVectorizer outperformed the HashVectorizer in terms of model accuracy.

## **Modeling**
I chose to apply three algorithms to this classification task: Logistic Regression, Decision Tree Classifier, and Multinomial Naive Bayes. When considering model accuracy, the ranking from lowest to highest is as follows: Decision Tree Classifier has the lowest accuracy, followed by Multinomial Naive Bayes, and Logistic Regression has the highest accuracy.



## **Evaluation**
- Logistic Regression

        Classification Report:
                        precision    recall  f1-score   support

                    0       0.91      0.89      0.90      5000
                    1       0.89      0.91      0.90      5000

             accuracy                           0.90     10000
            macro avg       0.90      0.90      0.90     10000
         weighted avg       0.90      0.90      0.90     10000


- Multinominal Naive Bayes


            Classification Report:
                        precision    recall  f1-score   support

                    0       0.86      0.88      0.87      5000
                    1       0.88      0.85      0.87      5000

             accuracy                           0.87     10000
            macro avg       0.87      0.87      0.87     10000
         weighted avg       0.87      0.87      0.87     10000


- Logistic Regression has the best report, has the best precision, recall, f1-score and accuracy, so I will tune its hyperparameters.


## **Hyperparameter Tuning**
The `Logistic Regression` best hyperparameters are:
- 'C': 4.9
- 'penalty': 'l2' 
- 'solver': 'liblinear'
    
        Classification Report:
                        precision    recall  f1-score   support

                    0       0.91      0.90      0.90      5000
                    1       0.90      0.91      0.90      5000

             accuracy                           0.90     10000
            macro avg       0.90      0.90      0.90     10000
         weighted avg       0.90      0.90      0.90     10000


# **MLE part**