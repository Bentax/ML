from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

data = pd.read_csv('Mesto.csv')
data.columns = ['ID', 'DATE', 'AGE', 'COUNTRY', 'PROJ', 'FROM', 'FROM1', 'ABOUT', 'PROF', 'PROF1', 'VALUE', 'VALUE1', 'L_FOR', 'L_FOR1', 'TYPE', 'ROLE']
data = data.dropna(subset=['ABOUT','TYPE'])
#reset index of DataFrame
data = data.reset_index(drop=True)
# Split the column at the '—' delimiter and expand into separate columns
split_type = data['TYPE'].str.split('—', expand=True)
# Extract the first part and save it to a new column
data['CAN'] = split_type[0].str.strip()
#data = data.dropna(subset='CAN')
# Extract the first part and save it to a new column
data['CAN'] = data['CAN'].str.replace('«','')
data['CAN'] = data['CAN'].str.replace('»','')
data['CAN'] = data['CAN'].str.replace('\n',' ')
data['CAN'] = data['CAN'].str.replace(':)','')
print(data['CAN'].unique())
# Sample text data and labels
texts = data['ABOUT']
labels = data['CAN']  # 0 for class A, 1 for class B, ...

# Vectorize text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5, random_state=42)

# Train a classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
# Accuracy: 0.2488736322677537
