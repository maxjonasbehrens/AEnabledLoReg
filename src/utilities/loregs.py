import numpy as np

def batch_weighted_regression(X, y, weights, intercept=True):
    """
    Perform weighted linear regression.

    Args:
        X (numpy.ndarray): The input features.
        y (numpy.ndarray): The target values.
        weights (numpy.ndarray): The weights for each data point.

    Returns:
        numpy.ndarray: The coefficients of the linear regression model.

    """
    # Add a column of ones to X for the intercept term
    if intercept:
        X = np.column_stack((np.ones(len(X)), X))

    # Calculate the weighted least squares solution
    W = np.diag(weights)
    
    # X_weighted = np.dot(W, X)
    # y_weighted = np.dot(W, y)
    
    # # More numerically stable method to solve for beta
    # beta = np.linalg.solve(X_weighted.T.dot(X), X_weighted.T.dot(y_weighted))

    # Compute the weighted version of X transpose
    XTW = np.matmul(X.T, W)

    # Compute the weighted version of X transpose times X
    XTWX = np.matmul(XTW, X)

    # Compute the inverse of XTWX
    XTWX_inv = np.linalg.inv(XTWX)

    # Compute the weighted version of X transpose times y
    XTWy = np.matmul(XTW, y)

    # Calculate the weights for the weighted linear regression
    beta = np.matmul(XTWX_inv, XTWy)
    
    return beta

def batch_weighted_regression_with_regularization(X, y, weights, lambda_reg, intercept=True):
    """
    Perform weighted linear regression with L2 regularization (Ridge regression).

    Args:
        X (numpy.ndarray): The input features.
        y (numpy.ndarray): The target values.
        weights (numpy.ndarray): The weights for each data point.
        lambda_reg (float): The regularization strength.

    Returns:
        numpy.ndarray: The coefficients of the linear regression model.

    """
    # Add a column of ones to X for the intercept term
    if intercept:
        X = np.column_stack((np.ones(len(X)), X))

    # Calculate the weighted least squares solution
    W = np.diag(weights)

    # Compute the weighted version of X transpose
    XTW = np.matmul(X.T, W)

    # Compute the weighted version of X transpose times X
    XTWX = np.matmul(XTW, X)

    # Add regularization term (lambda_reg * I, where I is identity matrix)
    reg_identity = np.eye(XTWX.shape[0]) * lambda_reg
    XTWX_reg = XTWX + reg_identity

    # Compute the inverse of the regularized XTWX
    XTWX_reg_inv = np.linalg.inv(XTWX_reg)

    # Compute the weighted version of X transpose times y
    XTWy = np.matmul(XTW, y)

    # Calculate the weights for the weighted linear regression with regularization
    beta = np.matmul(XTWX_reg_inv, XTWy)
    
    return beta

# Define functions to make predictions
def batch_predict(X, beta, intercept = True):
    """
    Predicts the target variable using linear regression.

    Parameters:
    X (numpy.ndarray): The input features.
    beta (numpy.ndarray): The regression coefficients.

    Returns:
    numpy.ndarray: The predicted target variable.
    """
    # Check if X is a single data point
    single = len(X.shape) == 1

    if intercept: 
        if single:
            X = np.insert(X, 0, 1)
        else:
            # Add a column of ones to X for the intercept term
            X = np.column_stack((np.ones(len(X)), X))
    
    return X @ beta


import torch
import torch.optim as optim

def torch_weighted_regression(X, y, weights):
    """
    Perform weighted linear regression.

    Args:
        X (torch.Tensor): The input features.
        y (torch.Tensor): The target values.
        weights (torch.Tensor): The weights for each data point.

    Returns:
        torch.Tensor: The coefficients of the linear regression model.

    """
    # # Add a column of ones to X for the intercept term
    ones_column = torch.ones(len(X), 1, device=X.device)
    X = torch.cat((ones_column, X), dim=1)

    # Add small value to weights to avoid singular matrix
    # weights += 1e-6

    # Calculate the weighted least squares solution
    W = torch.diag(weights) 

    # Compute the weighted version of X transpose
    XTW = torch.matmul(torch.transpose(X, 0, 1), W)

    # Compute the weighted version of X transpose times X
    XTWX = torch.matmul(XTW, X)

    # Compute the inverse of XTWX
    XTWX_inv = torch.linalg.pinv(XTWX.float()) 

    # Compute the weighted version of X transpose times y
    XTWy = torch.matmul(XTW, y)

    # Calculate the weights for the weighted linear regression
    beta = torch.matmul(XTWX_inv, XTWy)

    # Ensure inputs are float32
    # X = X.float()
    # y = y.float()
    # weights = weights.float()

    # # Add intercept term to X
    # ones_column = torch.ones(X.size(0), 1, device=X.device, dtype=torch.float32)
    # X_int = torch.cat((ones_column, X), dim=1) # X with intercept

    # # --- Use torch.linalg.lstsq ---
    # # Weighted least squares: minimize || W^(1/2) * (X*beta - y) ||^2
    # # Solve A * beta = b, where A = W_sqrt * X_int and b = W_sqrt * y

    # # Clamp weights to avoid sqrt of zero or negative numbers if weights can be zero
    # # Add a small epsilon for numerical stability if weights can be exactly zero
    # epsilon = 1e-8
    # weights_sqrt = torch.sqrt(torch.clamp(weights, min=epsilon))

    # # Apply weights
    # A = X_int * weights_sqrt.unsqueeze(1) # Equivalent to diag(weights_sqrt) @ X_int
    # b = y * weights_sqrt # Apply weights to y

    # # Ensure b is a column vector (or matrix if y has multiple columns)
    # if b.ndim == 1:
    #     b = b.unsqueeze(1)

    # # Solve the least squares problem
    # try:
    #     solution = torch.linalg.lstsq(A, b)
    #     beta = solution.solution.squeeze() # Get the coefficient vector
    # except torch.linalg.LinAlgError as e:
    #      print(f"Warning: torch.linalg.lstsq failed: {e}. Returning NaN coefficients.")
    #      # Return NaNs matching the expected output shape (num_features + intercept)
    #      beta = torch.full((X_int.shape[1],), float('nan'), device=X.device, dtype=torch.float32)
    # except RuntimeError as e:
    #     # Catch other potential runtime errors (like CUDA out of memory, though less likely here)
    #     print(f"Warning: Runtime error during torch.linalg.lstsq: {e}. Returning NaN coefficients.")
    #     beta = torch.full((X_int.shape[1],), float('nan'), device=X.device, dtype=torch.float32)


    return beta 

def torch_weighted_regression_with_regularization(X, y, weights, lambda_reg):
    """
    Perform weighted linear regression with L2 regularization (Ridge regression).

    Args:
        X (torch.Tensor): The input features.
        y (torch.Tensor): The target values.
        weights (torch.Tensor): The weights for each data point.
        lambda_reg (float): The regularization strength.

    Returns:
        torch.Tensor: The coefficients of the linear regression model.
    """
    # Add a column of ones to X for the intercept term
    ones_column = torch.ones(len(X), 1, device=X.device)
    X = torch.cat((ones_column, X), dim=1)

    # Add small value to weights to avoid singular matrix
    # weights += 1e-6

    # Calculate the weighted least squares solution
    W = torch.diag(weights)

    # Compute the weighted version of X transpose
    XTW = torch.matmul(torch.transpose(X, 0, 1), W)

    # Compute the weighted version of X transpose times X
    XTWX = torch.matmul(XTW, X)

    # Add regularization term (lambda_reg * I, where I is identity matrix)
    reg_identity = torch.eye(XTWX.size(0)) * lambda_reg
    XTWX_reg = XTWX + reg_identity

    # Compute the inverse of the regularized XTWX
    XTWX_reg_inv = torch.linalg.pinv(XTWX_reg.float())

    # Compute the weighted version of X transpose times y
    XTWy = torch.matmul(XTW, y)

    # Calculate the weights for the weighted linear regression with regularization
    beta = torch.matmul(XTWX_reg_inv, XTWy)

    return beta

# Inside loregs.py
import torch
import torch.optim as optim # Keep if you want to implement Adam logic manually later

# Keep your original torch_weighted_regression if desired

def torch_weighted_regression_gd(X, y, weights, iters=10, lr=1e-3):
    """
    Perform weighted linear regression using Gradient Descent,
    designed to be compatible with outer autograd loops.

    Args:
        X (torch.Tensor): The input features (Expected on target device, float32).
        y (torch.Tensor): The target values (Expected on target device, float32).
        weights (torch.Tensor): The weights for each data point (Expected on target device, float32).
        iters (int): Number of gradient descent iterations.
        lr (float): Learning rate for the inner optimization.

    Returns:
        torch.Tensor: The estimated coefficients (intercept first).
    """
    # Ensure inputs are float32 and on the correct device
    X = X.float()
    y = y.float()
    weights = weights.float()
    device = X.device # Get device from inputs

    # Add intercept term to X
    ones_column = torch.ones(X.size(0), 1, device=device, dtype=torch.float32)
    X_int = torch.cat((ones_column, X), dim=1) # Shape: (n_samples, n_features + 1)

    # Ensure y is a column vector
    if y.ndim == 1:
        y = y.unsqueeze(1) # Shape: (n_samples, 1)

    # Initialize beta coefficients
    # Requires grad = True because we need gradients w.r.t. it for the inner update
    beta = torch.zeros(X_int.shape[1], 1, requires_grad=True, device=device, dtype=torch.float32)

    # Inner optimization loop
    for _ in range(iters):
        # --- Manual Gradient Calculation and Update ---
        # Temporarily set beta to require grad for this iteration's gradient calc
        # (Might be redundant if requires_grad is set at init, but safe)
        beta_req_grad = beta.detach().requires_grad_(True)

        # Predict using current beta
        y_pred = X_int @ beta_req_grad # Shape: (n_samples, 1)
        # Calculate weighted squared error loss
        residuals = y - y_pred
        # Calculate loss as a scalar tensor
        loss = torch.sum(weights.unsqueeze(1) * (residuals ** 2))

        if torch.isnan(loss):
            # print(f"Warning: NaN loss in WLS GD iter {_}. Returning current beta.")
            return beta.detach().squeeze() # Return the non-NaN beta from previous step

        # Calculate gradients of loss w.r.t. beta *explicitly* for the inner update
        # Use create_graph=False to prevent this gradient calculation from
        # interfering with the outer backward pass graph state.
        grad_beta = torch.autograd.grad(loss, beta_req_grad, create_graph=False)[0]

        # Manual gradient descent step (Simple SGD)
        # Perform update outside of autograd tracking
        with torch.no_grad():
            beta -= lr * grad_beta # Update the original beta tensor
            # We don't need to zero grad_beta.grad as torch.autograd.grad doesn't populate it
        # --- End Manual Update ---

    # Return the final optimized beta, detached.
    # The operations mapping (X, y, weights) -> final beta remain in the outer graph.
    return beta.detach().squeeze() # Return shape (n_features + 1,)


def torch_predict(X, beta):
    """
    Predicts the target variable using linear regression.

    Parameters:
    X (torch.Tensor): The input features.
    beta (torch.Tensor): The regression coefficients.

    Returns:
    torch.Tensor: The predicted target variable.
    """
    # Check if X is a single data point
    # single = len(X.shape) == 1

    # if single:
        # ones_column = torch.tensor([1.0])
        # X_int = torch.cat((torch.tensor([1.0]), X))
    # else:
        # Add a column of ones to X for the intercept term
    ones_column = torch.ones(len(X), 1, device=X.device, dtype=torch.float32)
    X_int = torch.cat((ones_column, X), dim=1)
    
    # return torch.mm(X, beta).squeeze()
    return X_int @ beta

