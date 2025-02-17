import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report  # Import classification_report


# Load the data
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Preprocessing
df.drop('id', axis=1, inplace=True)
df.gender.replace({'Other': 'Female'}, inplace=True)
df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
df.bmi.fillna(df.bmi.mean(), inplace=True)

# Convert boolean/object columns to int (CRUCIAL!)
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = df[col].astype(int)
        except:
            df[col] = df[col].astype(bool)
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)

Y = df['stroke']
X = df.drop('stroke', axis=1)

# Shuffle and split
X_shuffled, y_shuffled = shuffle(X, Y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X_shuffled, y_shuffled, test_size=0.3, random_state=42
)

# Initial Model Training (Before SMOTE)
def create_baseline():
    model = Sequential()
    model.add(Dense(20, input_dim=20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    return model

model = create_baseline()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test)
print('Test Accuracy (Before SMOTE):', accuracy)
y_pred = (model.predict(X_test) > 0.5).astype(int)
print('Classification Report (Before SMOTE):')
print(classification_report(y_test, y_pred))


# SMOTE and subsequent training
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, Y)

# Convert boolean/object columns to int (CRUCIAL for SMOTE data too!)
for col in X_resampled.columns:
    if X_resampled[col].dtype == 'object':
        try:
            X_resampled[col] = X_resampled[col].astype(int)
        except:
            X_resampled[col] = X_resampled[col].astype(bool)
    if X_resampled[col].dtype == 'bool':
        X_resampled[col] = X_resampled[col].astype(int)


X_shuffled, y_shuffled = shuffle(X_resampled, y_resampled, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X_shuffled, y_shuffled, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

model = create_baseline()  # Recreate the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val))

loss, accuracy = model.evaluate(X_test, y_test)
print('Test Accuracy (After SMOTE, Basic Model):', accuracy)
y_pred = (model.predict(X_test) > 0.5).astype(int)
print('Classification Report (After SMOTE, Basic Model):')
print(classification_report(y_test, y_pred))

# Improved Model
def create_improved_model():
    model = Sequential()
    model.add(Dense(20, input_dim=20, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_improved_model()
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test, y_test)
print('Test Accuracy (After SMOTE, Improved Model):', accuracy)


from sklearn.metrics import classification_report  # Import classification_report

y_pred = (model.predict(X_test) > 0.5).astype(int)
print('Classification Report (After SMOTE, Improved Model):')
print(classification_report(y_test, y_pred))


scaling_stats = {}
numerical_features = ['age', 'avg_glucose_level', 'bmi']
for feature in numerical_features:
    scaling_stats[feature] = {'mean': X_train[feature].mean(), 'std': X_train[feature].std()}

import pickle
with open('scaling_stats.pkl', 'wb') as f: # Save to pickle file
    pickle.dump(scaling_stats, f)


model.save('model.h5')
print("Model saved as model.h5")