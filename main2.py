import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Function to create Gabor kernels
def create_gabor_kernels(num_kernels=8, ksize=31, sigma=4.0, lambd=10.0, gamma=0.5):
    kernels = []
    theta_values = np.linspace(0, np.pi, num_kernels, endpoint=False)
    for theta in theta_values:
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        kernels.append(kernel)
    return kernels

# Function to apply Gabor filter bank
def gabor_filter_bank(kernels, img):
    filtered_images = []
    for kernel in kernels:
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, kernel)
        filtered_images.append(filtered_img)
    return filtered_images

# Function to preprocess image
def preprocess_image(image):
    # Resize image to a smaller size for faster processing
    image = cv2.resize(image, (512, 512))
    
    gabor_kernels = create_gabor_kernels(num_kernels=8, ksize=31, sigma=5, lambd=10, gamma=0.5)
    filtered_imgs = gabor_filter_bank(gabor_kernels, image)
    enhanced_img = np.sum(filtered_imgs, axis=0)
    enhanced_img = cv2.normalize(enhanced_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gray_image = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    return gray_image

# Function to process images in batches
def process_images_in_batches(image_paths, batch_size):
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = []
        for image_path in batch_paths:
            img = cv2.imread(image_path)
            if img is not None:
                images.append(img)
        yield images

# Function to save images
def save_images(images, folder_path, prefix):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(folder_path, f"{prefix}_{i}.png"), img)

# Step 2: Feature extraction using MobileNetV3 Large with fine-tuning
def extract_features(images):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pre-trained MobileNetV3 Large model
    model = models.mobilenet_v3_large(weights='IMAGENET1K_V1').to(device)
    
    # Modify the model to include a 1x1 pointwise convolutional layer
    class MobileNetV3WithPointwiseConv(nn.Module):
        def __init__(self, original_model):
            super(MobileNetV3WithPointwiseConv, self).__init__()
            self.features = original_model.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.conv = nn.Conv2d(960, 128, kernel_size=1, stride=1, padding=0)
            self.bn = nn.BatchNorm2d(128)
            self.activation = nn.Hardswish()
        
        def forward(self, x):
            x = self.features(x)
            x = self.pool(x)
            x = self.conv(x)
            x = self.bn(x)
            x = self.activation(x)
            x = x.view(x.size(0), -1)  # Flatten the tensor
            return x
    
    model = MobileNetV3WithPointwiseConv(model).to(device)
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

# Function to save features
def save_features(features, folder_path, filename):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    np.save(os.path.join(folder_path, filename), features)

# SLSTM model class
class SLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(SLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)

# Function to build SLSTM model
def build_slstm_model(input_shape, params):
    input_size = input_shape[1]
    hidden_size = int(params[0])
    num_layers = 3  # Number of LSTM layers
    dropout = params[1]
    
    model = SLSTM(input_size, hidden_size, num_layers, dropout)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    return model, criterion, optimizer

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred = (y_pred > 0.5).float()
        accuracy = accuracy_score(y_test.cpu(), y_pred.cpu())
        precision = precision_score(y_test.cpu(), y_pred.cpu(), zero_division=0)
        sensitivity = recall_score(y_test.cpu(), y_pred.cpu())
        specificity = recall_score(y_test.cpu(), y_pred.cpu(), pos_label=0)
        auc = roc_auc_score(y_test.cpu(), y_pred.cpu())
    return accuracy, precision, sensitivity, specificity, auc

# Function to save output
def save_output(output, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(os.path.join(folder_path, "output.txt"), "w") as f:
        f.write(output)

# Function to calculate fitness
def fitness_function(params, X_train, y_train, X_val, y_val):
    units, dropout = params
    if units <= 0 or dropout < 0 or dropout > 1:
        return float('inf')  # Return a high error rate for invalid parameters

    hidden_size = int(units)
    if hidden_size <= 0:
        return float('inf')

    model, criterion, optimizer = build_slstm_model((X_train.shape[1], X_train.shape[2]), params)
    model = model.to(X_train.device)
    
    # Training loop for one epoch
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_outputs = (val_outputs > 0.5).float()
        error_rate = 1 - accuracy_score(y_val.cpu(), val_outputs.cpu())
    
    return error_rate


# Function to perform modified Sparrow Search Optimization
def modified_sparrow_search(X_train, y_train, X_val, y_val, num_agents=10, max_iter=20):
    dim = 2  # Number of hyperparameters to optimize (units, dropout)
    ST = 0.8  # Safety threshold
    a = 0.5  # Random integer
    L = np.ones(dim)  # Row vector with all elements as 1
    Q = np.random.uniform(0, 1, size=(num_agents, dim))  # Random integers following uniform distribution

    # Initialize the population of sparrows
    # Initialize the population of sparrows
    population = np.random.uniform(low=[50, 0.1], high=[200, 0.5], size=(num_agents, dim))  # Ensure valid ranges
    fitness = np.array([fitness_function(ind, X_train, y_train, X_val, y_val) for ind in population])
    best_idx = np.argmin(fitness)
    best_position = population[best_idx]
    best_fitness = fitness[best_idx]

    for t in range(max_iter):
        R2 = np.random.uniform(0, 1)
        for i in range(num_agents):
            if R2 < ST:
                population[i] = population[i] * np.exp(-i / (a * max_iter))
            else:
                population[i] = population[i] + Q[i] * L

            if i > num_agents / 2:
                population[i] = Q[i] * np.exp((population[best_idx] - population[i]) / (i ** 2))
            else:
                population[i] = best_position + np.abs(population[i] - best_position) * np.random.choice([-1, 1], size=dim) * L

            fitness[i] = fitness_function(population[i], X_train, y_train, X_val, y_val)

        best_idx = np.argmin(fitness)
        best_position = population[best_idx]
        best_fitness = fitness[best_idx]

    return best_position

# Example usage
if __name__ == "__main__":
    batch_size = 10  # Adjust batch size based on your memory capacity

    # Directory containing your images
    image_directory = r'C:\Users\ASUS\OneDrive\Desktop\DiabeticRetinopathySSO\data'

    # Get list of all image file paths
    image_paths = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory) if filename.endswith(('.png', '.jpg', '.jpeg'))]

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

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Step 3: Parameter tuning using modified Sparrow Search Optimization
    best_params = modified_sparrow_search(X_train, y_train, X_val, y_val)
    print(f"Best Parameters: Units: {best_params[0]}, Dropout: {best_params[1]}")

    # Step 4: Build and train model
    model, criterion, optimizer = build_slstm_model((X_train.shape[1], X_train.shape[2]), best_params)
    model = model.to(X_train.device)
    for epoch in tqdm(range(20), desc="Training model"):  # Increased epochs to 20
        print(f"Epoch {epoch+1}/20")
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Step 5: Evaluate model
    accuracy, precision, sensitivity, specificity, auc = evaluate_model(model, X_test, y_test)
    output = f"Accuracy: {accuracy}, Precision: {precision}, Sensitivity: {sensitivity}, Specificity: {specificity}, AUC: {auc}"
    print(output)

    # Save output to a folder
    save_output(output, 'output_folder')
