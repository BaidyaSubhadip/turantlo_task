import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv('shorten_data.csv')


data['Symptom_List'] = data['Symptom_List'].apply(lambda x: x.strip("[]").replace("'", "").split(','))

data['Symptom_List'] = data['Symptom_List'].apply(lambda x: ' '.join([symptom.strip() for symptom in x]))

le = LabelEncoder()
data['Disease_Name'] = le.fit_transform(data['Disease_Name'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Symptom_List'])
y = data['Disease_Name']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

with open('model.pkl', 'wb') as model_file:
    pickle.dump((model, vectorizer, le), model_file)

#Optionally, print the classification report
# y_pred = model.predict(X_test)
# unique_classes_in_test = sorted(set(y_test))
# unique_classes_in_pred = sorted(set(y_pred))
# all_unique_classes = sorted(set(unique_classes_in_test).union(set(unique_classes_in_pred)))
# target_names = [le.classes_[i] for i in all_unique_classes]

# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred, labels=all_unique_classes, target_names=target_names))
