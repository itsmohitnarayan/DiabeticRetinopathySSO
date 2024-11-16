import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import models, transforms
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Directory containing your images
image_directory = r'C:\Users\rahul\OneDrive\Desktop\DiabeticRetinopathySSO\data'

# Get list of all image file paths
image_paths = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory) if filename.endswith(('.png', '.jpg', '.jpeg'))]

# Step 1: Preprocessing using Gabor filtering
def create_gabor_kernels(num_kernels=8, ksize=31, sigma=4.0, lambd=10.0, gamma=0.5):
    kernels = []
    theta_values = np.linspace(0, np.pi, num_kernels, endpoint=False)
    for theta in theta_values:
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        kernels.append(kernel)
    return kernels

def gabor_filter_bank(kernels, img):
    filtered_images = []
    for kernel in kernels:
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, kernel)
        filtered_images.append(filtered_img)
    return filtered_images

def preprocess_image(image):
    # Resize image to a smaller size for faster processing
    # image = cv2.resize(image, (4288, 2848))
    
    gabor_kernels = create_gabor_kernels(num_kernels=8, ksize=31, sigma=5, lambd=10, gamma=0.5)
    filtered_imgs = gabor_filter_bank(gabor_kernels, image)
    enhanced_img = np.sum(filtered_imgs, axis=0)
    enhanced_img = cv2.normalize(enhanced_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gray_image = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    return gray_image

def save_images(images, folder_path, prefix):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(folder_path, f"{prefix}_{i}.png"), img)

# Step 2: Feature extraction using MobileNetV3 Large
def extract_features(images):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.mobilenet_v3_large(weights='IMAGENET1K_V1').to(device)
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    features = []
    with torch.no_grad():
        for img in tqdm(images, desc="Extracting features"):
            # Convert grayscale image to 3-channel image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = preprocess(img).unsqueeze(0).to(device)
            feature = model(img).cpu().numpy()
            features.append(feature)
    
    return np.array(features)

def save_features(features, folder_path, filename):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    np.save(os.path.join(folder_path, filename), features)

# Step 3: Parameter tuning using modified Sparrow Search Optimization
def fitness_function(params, X_train, y_train, X_val, y_val):
    units, dropout = params
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(int(units), return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(int(units), return_sequences=True))  # Adding another LSTM layer
    model.add(Dropout(dropout))
    model.add(LSTM(int(units)))  # Adding another LSTM layer
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_val, y_val), verbose=0)
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    return -accuracy  # We want to maximize accuracy, so we minimize the negative accuracy

def modified_sparrow_search(X_train, y_train, X_val, y_val, num_agents=10, max_iter=20):
    # Initialize the population
    population = np.random.rand(num_agents, 2)
    population[:, 0] = population[:, 0] * 200 + 50  # units in range [50, 250]
    population[:, 1] = population[:, 1] * 0.5  # dropout in range [0, 0.5]
    
    best_agent = None
    best_fitness = float('inf')
    
    for iteration in tqdm(range(max_iter), desc="Sparrow Search Optimization"):
        fitness = np.array([fitness_function(agent, X_train, y_train, X_val, y_val) for agent in population])
        
        if fitness.min() < best_fitness:
            best_fitness = fitness.min()
            best_agent = population[fitness.argmin()]
        
        # Update the population
        for i in range(num_agents):
            if np.random.rand() < 0.8:
                population[i] = best_agent + np.random.randn(2) * 0.1
            else:
                population[i] = population[i] + np.random.randn(2) * 0.1
            
            # Clamp the dropout rate to the range [0, 1]
            population[i, 1] = np.clip(population[i, 1], 0, 1)
        
        print(f"Iteration {iteration+1}/{max_iter}, Best Fitness: {-best_fitness}")
    
    return best_agent

# Step 4: Classification using Stacked LSTM
def build_lstm_model(input_shape, params):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(int(params[0]), return_sequences=True))
    model.add(Dropout(params[1]))
    model.add(LSTM(int(params[0]), return_sequences=True))  # Adding another LSTM layer
    model.add(Dropout(params[1]))
    model.add(LSTM(int(params[0])))  # Adding another LSTM layer
    model.add(Dropout(params[1]))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 5: Performance measures
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    sensitivity = recall_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, sensitivity, specificity, auc

# Function to save output to a folder
def save_output(output, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    output_path = os.path.join(folder_path, 'output.txt')
    with open(output_path, 'w') as f:
        f.write(output)

# Function to process images in batches
def process_images_in_batches(image_paths, batch_size):
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = [cv2.imread(image_path) for image_path in batch_paths]
        yield images

# Example usage
if __name__ == "__main__":
    batch_size = 500  # Adjust batch size based on your memory capacity

    # Step 1: Preprocess images in batches
    preprocessed_images = []
    for images in tqdm(process_images_in_batches(image_paths, batch_size), desc="Preprocessing images"):
        preprocessed_batch = [preprocess_image(img) for img in tqdm(images, desc="Processing batch")]
        preprocessed_images.extend(preprocessed_batch)
    save_images(preprocessed_images, 'output_folder/preprocessed_images', 'preprocessed')

    # Step 2: Extract features
    features = extract_features(preprocessed_images)
    save_features(features, 'output_folder/features', 'features.npy')

    # Reshape features for LSTM input
    features = features.reshape(features.shape[0], features.shape[1], -1)

    # Split data
    labels = np.random.randint(0, 2, size=(len(preprocessed_images),))
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Step 3: Parameter tuning using modified Sparrow Search Optimization
    best_params = modified_sparrow_search(X_train, y_train, X_val, y_val)
    print(f"Best Parameters: Units: {best_params[0]}, Dropout: {best_params[1]}")

    # Step 4: Build and train model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), best_params)
    for epoch in tqdm(range(20), desc="Training model"):  # Increased epochs to 20
        print(f"Epoch {epoch+1}/20")
        model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2)

    # Step 5: Evaluate model
    accuracy, precision, sensitivity, specificity, auc = evaluate_model(model, X_test, y_test)
    output = f"Accuracy: {accuracy}, Precision: {precision}, Sensitivity: {sensitivity}, Specificity: {specificity}, AUC: {auc}"
    print(output)

    # Save output to a folder
    save_output(output, 'output_folder')