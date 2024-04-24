import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import joblib


df=pd.read_csv('final_new.csv')

df.drop(df.columns[0], axis=1, inplace=True)
df.drop('SEVERITY_CLASS', axis=1, inplace=True)

df.dropna(inplace=True)

df['ACCIDENT_PREDICTION'] = np.where(df['NUMBER OF PERSONS INJURED'] == 0, 0, 1)

le = LabelEncoder()
df['Broader_weather_condition_encoded'] = le.fit_transform(df['Broader_weather_condition'])
df['BOROUGH_encoded'] = le.fit_transform(df['BOROUGH'])
df['DAY_OF_WEEK_encoded'] = le.fit_transform(df['DAY OF WEEK'])
df['CONTRIBUTING VEHICLE 1'] = le.fit_transform(df['CONTRIBUTING FACTOR VEHICLE 1'])
df['CONTRIBUTING VEHICLE 2'] = le.fit_transform(df['CONTRIBUTING FACTOR VEHICLE 2'])


df = df.rename(columns={
    'Broader_weather_condition_encoded': 'WEATHER',
    'BOROUGH_encoded': 'CITY',
    'DAY_OF_WEEK_encoded': 'DAY'
})


features=['Temperature','Wind_speed', 'Dew_point', 'ACCIDENT_PREDICTION','WEATHER','CITY','DAY']
df_model=df[features]
# accident_column = df_model.pop('ACCIDENT_PREDICTION')
# df_model['ACCIDENT_PREDICTION'] = accident_column
df_model = df_model[[col for col in df_model.columns if col != 'ACCIDENT_PREDICTION'] + ['ACCIDENT_PREDICTION']]
features_to_standardize = df_model.iloc[:, :-1].values
features_to_standardize
target_variable = df_model.iloc[:, -1].values
target_variable
mean = np.mean(features_to_standardize, axis=0)
std_dev = np.std(features_to_standardize, axis=0)
features_standardized = (features_to_standardize - mean) / std_dev
df_model.iloc[:, :-1] = features_standardized
df_model.iloc[:, -1] = target_variable
X = df_model.iloc[:, :-1].values
y = df_model.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

xgb_classifier = XGBClassifier(random_state=42)
xgb_classifier.fit(X_train_resampled, y_train_resampled)
joblib.dump(xgb_classifier, 'xgb_classifier.pkl')
y_pred = xgb_classifier.predict(X_test)
accuracy_xg = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy_xg:.2f}')
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')



