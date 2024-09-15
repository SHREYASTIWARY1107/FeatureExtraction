# sample_code.py

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from src.utils import download_images, extract_image_features, preprocess_text  # Absolute import
from src.constants import allowed_units  # Absolute import

# Define paths
DATASET_FOLDER = 'dataset/'  # Path to dataset folder
TRAIN_FILENAME = 'train.csv'
TEST_FILENAME = 'test.csv'
IMAGE_SAVE_DIR = 'downloaded_images'  # Path to the downloaded images

if __name__ == "__main__":
    # Download images from train and test CSV files
    download_images(
        os.path.join(DATASET_FOLDER, TRAIN_FILENAME),
        os.path.join(DATASET_FOLDER, TEST_FILENAME),
        IMAGE_SAVE_DIR
    )

    # Load data
    train_data = pd.read_csv(os.path.join(DATASET_FOLDER, TRAIN_FILENAME))
    test_data = pd.read_csv(os.path.join(DATASET_FOLDER, TEST_FILENAME))

    # Strip whitespace from column names
    train_data.columns = train_data.columns.str.strip()
    test_data.columns = test_data.columns.str.strip()

    # Print the columns to check their names
    print("Train Data Columns:", train_data.columns.tolist())
    print("Test Data Columns:", test_data.columns.tolist())

    # Preprocess text data
    train_data['preprocessed_text'] = train_data.apply(lambda row: preprocess_text(row['entity_name'] + ' ' + row['entity_value']), axis=1)

    # Extract image features
    train_data['image_features'] = train_data.apply(lambda row: extract_image_features(os.path.join(IMAGE_SAVE_DIR, f"{row.name}.jpg")), axis=1)

    # Filter out rows where image features are zero arrays (indicating missing images)
    valid_indices = [i for i, features in enumerate(train_data['image_features']) if not np.array_equal(features, np.zeros((224 * 224 * 3,)))]
    filtered_train_data = train_data.iloc[valid_indices]

    # Concatenate text and image features
    X_train = np.vstack(filtered_train_data['image_features'].tolist())
    y_train = filtered_train_data['group_id']  # Assuming group_id is the target variable

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Preprocess test data
    test_data['preprocessed_text'] = test_data.apply(lambda row: preprocess_text(row['entity_name']), axis=1)

    # Extract image features for test data
    test_data['image_features'] = test_data.apply(lambda row: extract_image_features(os.path.join(IMAGE_SAVE_DIR, f"{row.name}.jpg")), axis=1)

    # Filter out rows where image features are zero arrays (indicating missing images)
    valid_indices = [i for i, features in enumerate(test_data['image_features']) if not np.array_equal(features, np.zeros((224 * 224 * 3,)))]
    filtered_test_data = test_data.iloc[valid_indices]

    # Concatenate text and image features for testing
    X_test = np.vstack(filtered_test_data['image_features'].tolist())

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate F1 Score
    f1 = f1_score(filtered_test_data['group_id'], y_pred, average='weighted')  # Assuming group_id is also in test_data
    print(f'F1 Score: {f1:.4f}')

    # Save predictions to a CSV file
    OUTPUT_FILENAME = 'test_out.csv'
    filtered_test_data['prediction'] = y_pred
    filtered_test_data[['index', 'prediction']].to_csv(OUTPUT_FILENAME, index=False)