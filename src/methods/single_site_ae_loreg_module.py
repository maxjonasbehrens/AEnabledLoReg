import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
import os

# Add the parent directory to the module search path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import custom modules
from utilities import weights
from utilities import loregs

# Constants for hyperparameters
DROPOUT_RATE = 0.2  # Dropout rate for regularization in neural network layers
REGULARIZATION_LAMBDA = 0.5  # L2 regularization parameter for weighted regression
SPARSITY_WEIGHT = 0.1  # Weight for L1 sparsity loss on latent representations
SPARSITY_LOSS_WEIGHT = 0.5  # Weight for sparsity loss in total loss computation
NUMERICAL_STABILITY_EPS = 1e-8  # Small epsilon for numerical stability in variance calculations

# Define the dataset class
class SingleSiteDataset(Dataset):
    def __init__(self, data, outcome_var, num_splits=3):
        """
        Dataset class for a single site

        Args:
            data (dataframe): Dataframe containing the data
            outcome_var (str): Name of the outcome variable
            num_splits (int): Number of splits to create mini-batches
        """

        # Move outcome variable to the end
        temp_outcome = data.pop(outcome_var)
        data[outcome_var] = temp_outcome

        # Convert features and outcome to tensors
        self.data_x = torch.as_tensor(data.iloc[:, :-1].values, dtype=torch.float32)
        self.data_y = torch.as_tensor(data.iloc[:, -1].values, dtype=torch.float32)

        self.num_splits = num_splits

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        split_size = len(self.data_y) // self.num_splits

        mini_batches = []

        for i in range(self.num_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i != self.num_splits - 1 else len(self.data_y)

            indices = torch.arange(start_idx, end_idx)

            # Ensure the selected index `idx` is included in the mini-batch
            if idx not in indices:
                indices = torch.cat((indices, torch.tensor([idx]))).unique()

            X_batch = self.data_x[indices]
            Y_batch = self.data_y[indices]

            idx_in_batch = (indices == idx).nonzero(as_tuple=True)[0].item()
            mini_batches.append((X_batch, Y_batch, idx_in_batch))

        return mini_batches
    
class LikelihoodLoss(nn.Module):
    """
    Custom loss function that returns the likelihood ratio as-is.
    
    This is used for likelihood-based loss components in the training process.
    """
    def __init__(self):
        super(LikelihoodLoss, self).__init__()

    def forward(self, likelihood_ratio):
        """
        Forward pass that returns the input likelihood ratio.
        
        Args:
            likelihood_ratio (torch.Tensor): The likelihood ratio value.
            
        Returns:
            torch.Tensor: The same likelihood ratio value.
        """
        return likelihood_ratio

# Autoencoder with Regression class
class AutoencoderWithRegression(nn.Module):
    """
    Neural network combining autoencoder architecture with weighted regression.
    
    This model encodes input data to a latent representation, performs weighted
    regression in the latent space, and can decode back to the original space.
    
    Args:
        input_size (int): Number of input features.
        hidden_size_1 (int): Size of the first hidden layer.
        hidden_size_2 (int): Size of the second hidden layer.
        latent_size (int): Size of the latent representation.
    """
    def __init__(self, input_size, hidden_size_1, hidden_size_2, latent_size):
        super(AutoencoderWithRegression, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.Tanh(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.Tanh(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size_2, latent_size),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size_2),
            nn.Tanh(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size_2, hidden_size_1),
            nn.Tanh(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size_1, input_size),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Applies Xavier uniform initialization to all Linear layers
        and initializes biases to zero.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x_data, y_data, idx, sigma, k_nearest, kernel, device):
        """
        Forward pass for single-site data.

        Args:
            x_data (Tensor): Features of the single site.
            y_data (Tensor): Target variable of the single site.
            idx (int): Index of the current sample.
            sigma (float): Bandwidth for kernel weights.
            k_nearest (float): Proportion for nearest neighbors.
            kernel (str): Type of kernel ('gaussian', etc.).
            device (torch.device): Device to use ('cpu' or 'cuda').

        Returns:
            tuple: Encoded representations, decoded reconstructions, regression coefficients, and weights.
        """
        # Encode all data
        encoded_all = self.encoder(x_data)

        # Compute weights
        weights_all = weights.ss_torch_weights(encoded_all[idx], encoded_all, sigma=sigma, k_nearest=k_nearest, kernel=kernel)

        # Standardize weights
        weights_all_std = weights_all / torch.sum(weights_all)

        # Perform regression with regularization
        beta = loregs.torch_weighted_regression_with_regularization(
            encoded_all, y_data.squeeze(0), weights_all_std, lambda_reg=REGULARIZATION_LAMBDA
        )

        # Decode all data
        decoded_all = self.decoder(encoded_all)

        return encoded_all, decoded_all, beta, weights_all
    
# Function to create the dataset and data loader
def create_data_loader(dataset, outcome_var, batch_size, num_batches, shuffle):
    """
    Creates a data loader for the dataset.

    Args:
        dataset (DataFrame): The dataset containing features and outcomes.
        outcome_var (str): Name of the outcome variable.
        batch_size (int): Batch size for data loading.
        num_batches (int): Number of mini-batches.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    dataset = SingleSiteDataset(dataset, outcome_var, num_splits=num_batches)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=False)
    return data_loader

# Function to initialize the autoencoder model
def initialize_autoencoder(input_size, hidden_size_1, hidden_size_2, latent_size):
    """
    Initializes the autoencoder model.

    Args:
        input_size (int): Number of input features.
        hidden_size_1 (int): Size of the first hidden layer.
        hidden_size_2 (int): Size of the second hidden layer.
        latent_size (int): Size of the latent space.

    Returns:
        nn.Module: Initialized autoencoder model.
    """
    model = AutoencoderWithRegression(input_size, hidden_size_1, hidden_size_2, latent_size)
    return model

# Function to initialize model, optimizer, and loss storage
def initialize_training(input_size, hidden_size_1, hidden_size_2, latent_size, device, learning_rate, weight_decay):
    """
    Initializes the model, optimizer, and loss storage.

    Args:
        input_size (int): Number of input features.
        hidden_size_1 (int): Size of the first hidden layer.
        hidden_size_2 (int): Size of the second hidden layer.
        latent_size (int): Size of the latent space.
        device (torch.device): Device to use ('cpu' or 'cuda').
        learning_rate (float): Learning rate for optimization.
        weight_decay (float): Weight decay for regularization.

    Returns:
        tuple: Model, optimizer, scheduler, and loss storage.
    """
    model = AutoencoderWithRegression(input_size, hidden_size_1, hidden_size_2, latent_size).to(device).float()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-8)
    
    loss_values = []
    return model, optimizer, scheduler, loss_values

# Function to train the model
def train_model(input_data, outcome_var, batch_size, switch_shuffle, num_batches, model, optimizer, scheduler, num_epochs, log_interval, alpha, theta, gamma, device, sigma, k_nearest, kernel):
    """
    Train the model using single-site data.

    Args:
        input_data (DataFrame): Input dataset containing features and outcomes.
        outcome_var (str): Name of the outcome variable column.
        batch_size (int): Size of batches for training.
        switch_shuffle (int): Epoch at which to switch from non-shuffled to shuffled data loading.
        num_batches (int): Number of mini-batches to create.
        model (nn.Module): The autoencoder model to train.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler: Learning rate scheduler.
        num_epochs (int): Total number of training epochs.
        log_interval (int): Interval for logging training progress.
        alpha (float): Weight for reconstruction loss component.
        theta (float): Weight for null model loss component.
        gamma (float): Weight for global/orthogonal loss component.
        device (torch.device): Device to use ('cpu' or 'cuda').
        sigma (float): Bandwidth parameter for kernel weights.
        k_nearest (float): Proportion for nearest neighbors selection.
        kernel (str): Type of kernel function ('gaussian', etc.).

    Returns:
        dict: Dictionary containing loss values and training metrics for each logged epoch.
    """
    criterion_reconstruction = nn.MSELoss(reduction='mean')
    criterion_likelihood = LikelihoodLoss()

    current_loss_values = {
        "alpha": alpha,
        "theta": theta,
        "gamma": gamma,
        "epoch": [],
        "loss": [],
        "reconstruction_loss": [],
        "null_loss": [],
        "global_loss": []
    }

    # Create data loader for single-site data
    data_loader = create_data_loader(input_data, outcome_var, batch_size, num_batches, shuffle=False)

    for epoch in tqdm(range(num_epochs), desc="Training"):

        if epoch == switch_shuffle:
            data_loader = create_data_loader(input_data, outcome_var, batch_size, num_batches, shuffle=True)

        running_loss_reconstruction = 0.0
        running_loss_null = 0.0
        running_loss_global = 0.0
        running_loss_sparsity = 0.0
        running_total_loss = 0.0
        num_batches_processed = 0

        for batch in data_loader:
            optimizer.zero_grad()

            total_loss_reconstruction_in_batch = 0
            total_loss_null_in_batch = 0
            total_loss_global_in_batch = 0
            total_loss_sparsity_in_batch = 0

            mini_batches = batch

            for mini_batch in mini_batches:
                X_target, Y_target, idx = mini_batch

                X_target = X_target.squeeze(0).to(device).float()
                Y_target = Y_target.to(device).float()
                idx = idx.to(device)

                # Forward pass
                encoded_all, decoded_all, beta_t, weights_t = model(X_target, Y_target, idx, sigma, k_nearest, kernel, device)

                # Compute decoded target
                decoded_target = decoded_all[:X_target.size(0)]

                # Add intercept to encoded representation
                encoded_target_int = torch.cat((torch.ones(encoded_all.size(0), 1, device=encoded_all.device), encoded_all), dim=1)

                # Standardize weights
                weights_t_std = weights_t / torch.sum(weights_t)

                # Compute residuals and variance
                def calculate_residuals_and_variance(encoded_int, beta, Y, weights_std):
                    y_pred = encoded_int @ beta
                    residuals = Y.squeeze(0) - y_pred
                    weighted_var = torch.sum(weights_std * (residuals ** 2)) + NUMERICAL_STABILITY_EPS
                    return residuals, weighted_var

                residuals_target, weighted_var_target = calculate_residuals_and_variance(encoded_target_int, beta_t, Y_target, weights_t_std)

                # Compute log-likelihood for target
                def compute_log_likelihood(residuals, weighted_var, weights_std):
                    return -0.5 * torch.sum(weights_std * (torch.log(weighted_var) + (residuals ** 2) / weighted_var))

                log_likelihood_target = compute_log_likelihood(residuals_target, weighted_var_target, weights_t_std)
                
                def calculate_null_model_log_likelihood(Y, weights_std):
                    # Compute weighted mean
                    mean_y = torch.sum(weights_std * Y) / torch.sum(weights_std)
                    null_pred = mean_y.expand_as(Y)
                    null_residuals = Y - null_pred
                    
                    # Compute weighted variance with numerical stability
                    var_null = torch.sum(weights_std * (null_residuals ** 2)) / torch.sum(weights_std) + NUMERICAL_STABILITY_EPS
                    return compute_log_likelihood(null_residuals, var_null, weights_std)

                log_likelihood_null = calculate_null_model_log_likelihood(Y_target, weights_t_std)

                # Compute likelihood ratio - if ratio is negative, then target model is better
                likelihood_ratio_test = 2 * (log_likelihood_null - log_likelihood_target)

                # Orthogonal loss of latent space
                encoded_centered = encoded_all - torch.mean(encoded_all, dim=0)
                cov_matrix = torch.mm(encoded_centered.T, encoded_centered) / encoded_all.size(0)

                identity_matrix = torch.eye(encoded_all.size(1)).to(device)
                ortho_loss = torch.norm(cov_matrix - identity_matrix)
                likelihood_ratio_global = ortho_loss

                # L1 Sparsity loss on latent space
                l1_sparsity_loss = torch.sum(torch.abs(encoded_all)) / encoded_all.size(0)
                sparsity_loss = SPARSITY_WEIGHT * l1_sparsity_loss

                # Compute losses
                loss_null = criterion_likelihood(likelihood_ratio_test)
                loss_reconstruction = criterion_reconstruction(decoded_target, X_target)
                loss_global = criterion_likelihood(likelihood_ratio_global)

                total_loss_reconstruction_in_batch += loss_reconstruction
                total_loss_null_in_batch += loss_null
                total_loss_global_in_batch += loss_global
                total_loss_sparsity_in_batch += sparsity_loss


            # Average losses over mini-batches within the current 'batch' from DataLoader
            avg_loss_reconstruction_batch = total_loss_reconstruction_in_batch / len(mini_batches)
            avg_loss_null_batch = total_loss_null_in_batch / len(mini_batches)
            avg_loss_global_batch = total_loss_global_in_batch / len(mini_batches)
            avg_loss_sparsity_batch = total_loss_sparsity_in_batch / len(mini_batches)

            # Combine losses for the current 'batch' from DataLoader
            current_batch_total_loss = (alpha * avg_loss_reconstruction_batch + 
                                       theta * avg_loss_null_batch + 
                                       gamma * avg_loss_global_batch + 
                                       SPARSITY_LOSS_WEIGHT * avg_loss_sparsity_batch)

            current_batch_total_loss.backward() # Backpropagate the loss for the current 'batch'
            optimizer.step()

            # Accumulate losses for epoch average
            running_loss_reconstruction += avg_loss_reconstruction_batch.item()
            running_loss_null += avg_loss_null_batch.item()
            running_loss_global += avg_loss_global_batch.item()
            running_loss_sparsity += avg_loss_sparsity_batch.item()
            running_total_loss += current_batch_total_loss.item()
            num_batches_processed += 1

        # Calculate average losses for the epoch
        avg_epoch_total_loss = running_total_loss / num_batches_processed
        avg_epoch_reconstruction_loss = running_loss_reconstruction / num_batches_processed
        avg_epoch_null_loss = running_loss_null / num_batches_processed
        avg_epoch_global_loss = running_loss_global / num_batches_processed

        # For ReduceLROnPlateau, pass the metric to monitor (e.g., avg_epoch_total_loss)
        scheduler.step(avg_epoch_total_loss)

        if (epoch + 1) % log_interval == 0:
            current_loss_values["epoch"].append(epoch + 1)
            current_loss_values["loss"].append(avg_epoch_total_loss)
            current_loss_values["reconstruction_loss"].append(avg_epoch_reconstruction_loss)
            current_loss_values["null_loss"].append(avg_epoch_null_loss)
            current_loss_values["global_loss"].append(avg_epoch_global_loss)

            # Log current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{num_epochs}], Avg Total Loss: {avg_epoch_total_loss:.4f}, LR: {current_lr:.6f}")

            # TensorBoard logging removed for reproducibility - logging functionality not needed for core analysis

    return current_loss_values
