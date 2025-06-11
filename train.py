import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import accuracy_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse

from text_cleaner import transform_text, TextCleaner

df = pd.read_csv("spam.csv", encoding = 'latin-1')
# print(df.head())
# print(df.shape)

# print(df.info()# )

# remove the last three columns
df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)

# rename the columns
df.rename(columns = {'v1':'target', 'v2':'text'}, inplace = True)

# label encode the target values
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df["target"] = encoder.fit_transform(df["target"])

# drop null values
df.dropna(inplace = True)

# drop duplicates
df.drop_duplicates(keep = "first", inplace = True)

# print(df.shape)

# # Separate the data into feature and target variables
X = df[["text"]]  # pandas DataFrame
y = df["target"]  # pandas Series

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)

# Function to convert numeric data to sparse matrix (avoids sparse-dense mix errors)
def to_sparse(X):
  return sparse.csr_matrix(X)

to_sparse_transformer = FunctionTransformer(to_sparse, accept_sparse = True)

#---------------------------------------------------

# Preprocessing pipelines

# Text pipeline
text_pipeline = Pipeline([
    ('text_cleaner', TextCleaner()),
    ('text_vectorizer', TfidfVectorizer(max_features = 3000))
    # ('to_dense', FunctionTransformer(lambda x: x.toarray(), accept_sparse = True)),       # Uncomment if wanna use it
    # ('scaler', MinMaxScaler())
])

# Numeric pipeline (for features like 'num_characters' etc)
num_pipeline = Pipeline([
    ('scaler', MinMaxScaler()),           # Scaling optional
    ('to_sparse', to_sparse_transformer)    # Convert to sparse for TF-IDF compatibility
])

# Full preprocessing pipeline
full_preprocessing = ColumnTransformer([
    ('text', text_pipeline, 'text')
    # ('num', num_pipeline, ['num_characters'])    # Uncomment if wanna use it
])

# Voting Classifier
base_models_vc = [
('mnb', MultinomialNB()),
('etc', ExtraTreesClassifier(n_estimators = 50, random_state = 2)),
('svc', SVC(kernel = 'sigmoid', gamma = 1.0, probability = True)),
('xgb', XGBClassifier(n_estimators = 50, random_state = 2))
]

# Define the Voting Classifier
voting_clf = VotingClassifier(estimators = base_models_vc, voting = 'soft')

# Wrap the Voting Classifier inside a Pipeline
voting_pipeline = Pipeline([
    ('preprocessing', full_preprocessing),
    ('voting', voting_clf)
])

# Train the Voting Classifier
voting_pipeline.fit(X_train, y_train)

# Predict on X_test
y_pred = voting_pipeline.predict(X_test)

# print("Accuracy Score for Voting Classifier:", accuracy_score(y_test, y_pred))
# print("Precision Score for Voting Classifier:", precision_score(y_test, y_pred, zero_division = 0))

# Save the Voting Pipeline
pickle.dump(voting_pipeline, open("voting_pipeline.pkl", "wb"))