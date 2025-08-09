"""
# Functionality: This code implements knowledge distillation using KL divergence, a technique where a lightweight "student" model is trained to mimic the behavior of a larger, more complex "teacher" model. 
# The key idea is to transfer knowledge from the teacher to the student by minimizing the KL divergence between their output distributions (in addition to standard classification loss on true labels).
# The implementation includes:
# - Loading and preprocessing the CIFAR-10 dataset
# - Defining a complex teacher model and a lightweight student model
# - Training functions for baseline models (using standard cross-entropy loss)
# - A distillation training function that combines KL divergence loss (for knowledge transfer) and cross-entropy loss (for label alignment)
# - Evaluation code to compare the performance of the teacher, baseline student, and distillation-trained student
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Set up computation device
# Use GPU if available for faster training, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

######################################################################
# Data Loading and Preparation
######################################################################
# Define data preprocessing transformations
# Convert images to tensors and normalize with ImageNet statistics
transforms_cifar = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor (0-255 -> 0.0-1.0)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard ImageNet normalization
])

# Load CIFAR-10 dataset (32x32 color images, 10 classes)
# Training dataset - used for model training
train_dataset = datasets.CIFAR10(
    root='./data',         # Directory to store/locate the dataset
    train=True,            # Load training split
    download=True,         # Download dataset if not already in root directory
    transform=transforms_cifar  # Apply preprocessing transformations
)

# Test dataset - used for evaluating model performance
test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,           # Load test split
    download=True,
    transform=transforms_cifar
)

# Create data loaders to iterate over the datasets in batches
# Training data loader with shuffling for better training
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,        # Number of samples per batch
    shuffle=True,          # Shuffle training data to prevent order bias
    num_workers=2          # Number of subprocesses for data loading
)

# Test data loader without shuffling
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,         # No need to shuffle test data
    num_workers=2
)

######################################################################
# Model Definitions
######################################################################
# Teacher model - larger and more complex network with more parameters
class TeacherNet(nn.Module):
    def __init__(self, num_classes=10):
        super(TeacherNet, self).__init__()
        # Convolutional feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),  # 3 input channels (RGB), 128 output channels
            nn.ReLU(),                                    # Non-linear activation
            nn.Conv2d(128, 64, kernel_size=3, padding=1), # Reduce to 64 channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),        # Reduce spatial dimensions by half
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # Further reduce to 32 channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),        # Final spatial reduction
        )
        
        # Classifier head for final prediction
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),  # 2048 input features from flattened conv output
            nn.ReLU(),
            nn.Dropout(0.1),       # Regularization to prevent overfitting
            nn.Linear(512, num_classes)  # Output layer with 10 classes
        )

    def forward(self, x):
        """Forward pass through the network"""
        x = self.features(x)          # Extract features using convolutional layers
        x = torch.flatten(x, 1)       # Flatten feature maps into vector
        x = self.classifier(x)        # Generate class predictions
        return x

# Student model - smaller, lightweight network with fewer parameters
class StudentNet(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentNet, self).__init__()
        # Simpler convolutional feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # Fewer channels (16) than teacher
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), # Maintain 16 channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Simplified classifier head
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),  # Fewer input features and hidden units than teacher
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """Forward pass through the network"""
        x = self.features(x)          # Extract features
        x = torch.flatten(x, 1)       # Flatten
        x = self.classifier(x)        # Generate predictions
        return x

######################################################################
# Core Functions (KL Divergence Distillation)
######################################################################
def train_baseline(model, train_loader, epochs, learning_rate, device):
    """
    Train a model using standard cross-entropy loss (for both teacher and baseline student)
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Computation device (CPU/GPU)
    """
    criterion = nn.CrossEntropyLoss()  # Standard cross-entropy loss for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer
    model.train()  # Set model to training mode (enables dropout, batch norm updates)
    
    for epoch in range(epochs):
        running_loss = 0.0  # Track average loss across batches
        for inputs, labels in train_loader:
            # Move data to target device
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Reset gradients
            
            # Forward pass: get model predictions
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Calculate loss
            
            # Backward pass and optimize
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights
            
            running_loss += loss.item()  # Accumulate loss
        
        # Print average loss for the epoch
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

def train_distillation(teacher, student, train_loader, epochs, learning_rate, T, 
                      kl_weight, ce_weight, device):
    """
    Train student model using knowledge distillation with KL divergence
    
    Args:
        teacher: Pre-trained teacher model
        student: Student model to train
        train_loader: DataLoader for training data
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        T: Temperature parameter for softening probabilities
        kl_weight: Weight for KL divergence loss
        ce_weight: Weight for cross-entropy loss
        device: Computation device (CPU/GPU)
    """
    ce_criterion = nn.CrossEntropyLoss()  # Loss for hard labels
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    
    teacher.eval()  # Set teacher to evaluation mode (frozen)
    student.train() # Set student to training mode
    
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Get teacher outputs without computing gradients
            with torch.no_grad():
                teacher_logits = teacher(inputs)  # Teacher's raw predictions
            
            # Get student outputs
            student_logits = student(inputs)     # Student's raw predictions
            
            # Calculate KL divergence loss (core of knowledge distillation)
            # Teacher distribution: soft probabilities using temperature
            teacher_soft = nn.functional.softmax(teacher_logits / T, dim=-1)
            # Student distribution: log probabilities for KL calculation
            student_soft = nn.functional.log_softmax(student_logits / T, dim=-1)
            
            # KL divergence between teacher and student distributions
            kl_loss = torch.sum(teacher_soft * (teacher_soft.log() - student_soft)) / inputs.size(0)
            kl_loss *= T **2  # Compensate for temperature scaling
            
            # Calculate cross-entropy loss with hard labels
            ce_loss = ce_criterion(student_logits, labels)
            
            # Total loss: weighted combination of both losses
            total_loss = kl_weight * kl_loss + ce_weight * ce_loss
            
            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
        
        print(f"Distillation Epoch {epoch+1}/{epochs}, Total Loss: {running_loss/len(train_loader):.4f}")

def test(model, test_loader, device):
    """
    Evaluate model accuracy on test dataset
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        device: Computation device (CPU/GPU)
    
    Returns:
        Accuracy percentage
    """
    model.eval()  # Set model to evaluation mode
    correct = 0   # Count of correct predictions
    total = 0     # Total number of samples
    
    with torch.no_grad():  # Disable gradient computation for efficiency
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # Get class with highest score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # Count correct predictions
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

######################################################################
# Execution Pipeline
######################################################################
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # 1. Train the teacher model
    teacher = TeacherNet().to(device)
    print("===== Training Teacher Model =====")
    train_baseline(teacher, train_loader, epochs=10, learning_rate=0.001, device=device)
    teacher_acc = test(teacher, test_loader, device)
    
    # 2. Train baseline student model (without distillation) for comparison
    torch.manual_seed(42)  # Reset seed for fair comparison
    student_baseline = StudentNet().to(device)
    print("\n===== Training Baseline Student Model (without distillation) =====")
    train_baseline(student_baseline, train_loader, epochs=10, learning_rate=0.001, device=device)
    student_baseline_acc = test(student_baseline, test_loader, device)
    
    # 3. Train student model using KL divergence distillation
    torch.manual_seed(42)  # Reset seed for fair comparison
    student_distill = StudentNet().to(device)
    print("\n===== Training Student Model with KL Divergence Distillation =====")
    train_distillation(
        teacher=teacher,
        student=student_distill,
        train_loader=train_loader,
        epochs=10,
        learning_rate=0.001,
        T=2,               # Temperature parameter - higher = softer probabilities
        kl_weight=0.25,    # Weight for knowledge distillation loss
        ce_weight=0.75,    # Weight for hard label loss
        device=device
    )
    student_distill_acc = test(student_distill, test_loader, device)
    
    # 4. Print final comparison results
    print("\n===== Final Results =====")
    print(f"Teacher Model Accuracy: {teacher_acc:.2f}%")
    print(f"Baseline Student Accuracy: {student_baseline_acc:.2f}%")
    print(f"KL Distillation Student Accuracy: {student_distill_acc:.2f}%")
