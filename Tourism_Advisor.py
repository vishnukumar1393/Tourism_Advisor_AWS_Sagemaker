import pandas as pd
import numpy as np
import boto3
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler

bucket_name = "cloudaiml1393"
file_name = "TourismAdvisorDataset.csv"

s3 = boto3.client('s3')
obj = s3.get_object(Bucket=bucket_name, Key=file_name)
df = pd.read_csv(obj['Body'])

df['Destination'] = df['Destination'].str.strip().str.title()

imputer = SimpleImputer(strategy='most_frequent')
df.iloc[:, :] = imputer.fit_transform(df)

trip_encoder = LabelEncoder()
transport_encoder = LabelEncoder()
destination_encoder = LabelEncoder()

df['Type of Trip'] = trip_encoder.fit_transform(df['Type of Trip'])
df['Mode of Transport'] = transport_encoder.fit_transform(df['Mode of Transport'])
df['Destination'] = destination_encoder.fit_transform(df['Destination'])

X = df[['Budget', 'Type of Trip', 'Mode of Transport']]
y = df['Destination']

ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer_X = SimpleImputer(strategy='mean')
X_train = imputer_X.fit_transform(X_train)
X_test = imputer_X.transform(X_test)

rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, scoring='accuracy')
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

ada_params = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
ada_grid = GridSearchCV(AdaBoostClassifier(), ada_params, cv=5, scoring='accuracy')
ada_grid.fit(X_train, y_train)
best_ada = ada_grid.best_estimator_

ensemble = VotingClassifier(
    estimators=[('rf', best_rf), ('ada', best_ada)], voting='hard'
)

ensemble.fit(X_train, y_train)

y_pred = ensemble.predict(X_test)
print(f'Model Accuracy: {accuracy_score(y_test, y_pred):.4f}')

joblib.dump(ensemble, 'tourism_model.pkl')
s3.upload_file('tourism_model.pkl', bucket_name, 'tourism_model.pkl')

budget = int(input('Enter your budget: '))
if budget < 2000:
    print("Stay home, stay safe")
else:
    trip_type = input('Enter Type of Trip (Adventure, Relaxation, Historical, Nature, Beach, Hill Station, Cultural): ').strip().title()
    mode_transport = input('Enter Mode of Transport (Road, Flight): ').strip().title()

    if trip_type not in trip_encoder.classes_:
        print(f"Error: '{trip_type}' is not a recognized trip type. Please enter a valid option.")
    elif mode_transport not in transport_encoder.classes_:
        print(f"Error: '{mode_transport}' is not a recognized mode of transport. Please enter a valid option.")
    else:
        trip_type_encoded = trip_encoder.transform([trip_type])[0]
        mode_transport_encoded = transport_encoder.transform([mode_transport])[0]
        user_input = np.array([[budget, trip_type_encoded, mode_transport_encoded]])
        prediction = ensemble.predict(user_input)
        recommended_destination = destination_encoder.inverse_transform(prediction)[0]
        print(f'Recommended Destination: {recommended_destination}')
