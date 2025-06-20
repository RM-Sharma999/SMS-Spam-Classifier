# SMS Spam Classification Using Machine Learning

This project focuses on classifying text messages as either **spam** or **ham (non-spam)** using classical machine learning models and natural language processing techniques. The goal is to develop a practical and accurate spam filter suitable for real-world deployment.

---

## Objective

To create a reliable and interpretable text classification model that detects spam messages with a focus on high precision, leveraging traditional machine learning algorithms and NLP preprocessing techniques.

---

## Dataset Overview

The dataset contains over 5,000 labeled text messages, categorized into **ham** and **spam**. However, the classes are notably imbalanced—approximately **87%** of the messages are ham, and only **13%** are spam.

This imbalance can lead to biased models that favor the majority class. Hence, we emphasize evaluation metrics like **precision** and utilize confusion matrices to properly gauge spam detection capability.

![](https://live.staticflickr.com/65535/54582600794_0c792c6692_z.jpg)

---

## Exploratory Data Analysis (EDA)

EDA was performed to extract meaningful patterns and guide feature engineering using a combination of descriptive statistics and informative visualizations.

### Distribution of Characters, Words, and Sentences by Class

Bar plots were created to compare the number of characters, words, and sentences in spam and ham messages. Spam messages tend to be longer, more structured, and typically contain persuasive language.

![](https://live.staticflickr.com/65535/54581552007_b771f7a050_b.jpg)

### Word Clouds for Spam and Ham Messages

Word clouds visually represented the most frequent terms in each class.

![](https://live.staticflickr.com/65535/54582637728_0487e166b2.jpg)
![](https://live.staticflickr.com/65535/54581552022_0b2079b127.jpg)

- **Spam messages** prominently feature promotional and urgent terms like `free`, `win`, `claim`, `urgent`, `now`, and `click`.
- **Ham messages** are conversational, with frequent use of informal, contextual terms like `ok`, `go`, `see`, `later`, and `home`.

### Top 30 Most Common Words in Spam and Ham

Bar plots of the top 30 most used words in each category were generated to better understand message structure and content. These insights played a key role in vocabulary selection and token filtering.

![](https://live.staticflickr.com/65535/54581552032_6783406985_b.jpg)
![](https://live.staticflickr.com/65535/54582739065_a04c268c56_b.jpg)

---

## Text Preprocessing

Steps performed:

- Lowercasing  
- Removing punctuation and stopwords  
- Tokenization  
- Lemmatization (with **NLTK**)  
- TF-IDF vectorization  

---

## Model Training & Evaluation

### Naive Bayes Models

Three Naive Bayes variants were evaluated:

- Multinomial NB  
- Bernoulli NB  
- Gaussian NB  

Multinomial NB emerged as the most effective due to its compatibility with count-based text features.

![](https://live.staticflickr.com/65535/54582637748_f7af51d911_b.jpg)

### Baseline Models

Multiple traditional models—including linear classifiers, tree-based algorithms, and boosting methods—were trained using TF-IDF features. Their performance was evaluated and visualized using bar plots, offering a clear benchmark.

![](https://live.staticflickr.com/65535/54582739120_9ef821dcf0_b.jpg)

---

## Ensemble Methods

To achieve more robust results, ensemble techniques were adopted by combining diverse models and leveraging their collective strength.

### Soft Voting Classifier

This ensemble combined predictions from:

- Multinomial Naive Bayes  
- Extra Trees Classifier  
- Support Vector Classifier  
- XGBoost  

The final decision was made by averaging the predicted probabilities. This method improved generalization and precision, achieving ~98% accuracy.

### Stacking Classifier

Using the same base models, a **Random Forest** was trained as a meta-learner on their outputs. This strategy enabled the model to learn from complex patterns across predictions and provided an extra performance boost.

To compare the effectiveness of the ensemble strategies, a side-by-side confusion matrix subplot was generated. The voting classifier demonstrated flawless spam detection with zero false positives, while the stacking classifier also performed strongly but showed a few misclassifications. This visualization highlights the precision advantage of the voting approach.

![](https://live.staticflickr.com/65535/54581552102_fb5301ab54_b.jpg)

---

## Technologies Used

- **Programming Language:** Python  
- **Natural Language Processing:** `nltk`  
- **Visualization:** `matplotlib`, `seaborn`, `wordcloud`  
- **Machine Learning:** `scikit-learn`, `xgboost`  
- **Web Interface:** `Streamlit`  
- **Deployment Platform:** `Render`  

---

## Deployment

The final model was deployed using **Streamlit** to create an intuitive web-based interface, and hosted on **Render** for public accessibility.

[SMS Spam Classifier Live App](https://sms-spam-classifier-igvf.onrender.com)

---

## Key Takeaways

- Multinomial Naive Bayes (MNB) was found to be a strong and reliable base model for text classification.
- Visual EDA clearly distinguished spam from ham based on structure and vocabulary.
- Ensemble models—particularly the soft voting classifier—delivered the best results, achieving **100% precision** and around **98% accuracy**.
- Using Streamlit and Render simplified deployment and made the model usable in real-world applications.

---
