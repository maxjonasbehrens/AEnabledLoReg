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

def torch_weighted_regression(X, y, weights, device):
    """
    Perform weighted linear regression.

    Args:
        X (torch.Tensor): The input features.
        y (torch.Tensor): The target values.
        weights (torch.Tensor): The weights for each data point.

    Returns:
        torch.Tensor: The coefficients of the linear regression model.

    """
    # Add a column of ones to X for the intercept term
    ones_column = torch.ones(len(X), 1).to(device)
    X = torch.cat((ones_column, X), dim=1)

    # Calculate the weighted least squares solution
    W = torch.diag(weights) 

    # Compute the weighted version of X transpose
    XTW = torch.matmul(torch.transpose(X, 0, 1), W)

    # Compute the weighted version of X transpose times X
    XTWX = torch.matmul(XTW, X)

    # Compute the inverse of XTWX
    XTWX_inv = torch.inverse(XTWX) 

    # Compute the weighted version of X transpose times y
    XTWy = torch.matmul(XTW, y)

    # Calculate the weights for the weighted linear regression
    beta = torch.matmul(XTWX_inv, XTWy)

    return beta 

def torch_weighted_regression_with_regularization(X, y, weights, device, lambda_reg):
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
    ones_column = torch.ones(len(X), 1).to(device)
    X = torch.cat((ones_column, X), dim=1)

    # Calculate the weighted least squares solution
    W = torch.diag(weights)

    # Compute the weighted version of X transpose
    XTW = torch.matmul(torch.transpose(X, 0, 1), W)

    # Compute the weighted version of X transpose times X
    XTWX = torch.matmul(XTW, X)

    # Add regularization term (lambda_reg * I, where I is identity matrix)
    reg_identity = torch.eye(XTWX.size(0)).to(device) * lambda_reg
    XTWX_reg = XTWX + reg_identity

    # Compute the inverse of the regularized XTWX
    XTWX_reg_inv = torch.inverse(XTWX_reg)

    # Compute the weighted version of X transpose times y
    XTWy = torch.matmul(XTW, y)

    # Calculate the weights for the weighted linear regression with regularization
    beta = torch.matmul(XTWX_reg_inv, XTWy)

    return beta


def torch_predict(X, beta, device):
    """
    Predicts the target variable using linear regression.

    Parameters:
    X (torch.Tensor): The input features.
    beta (torch.Tensor): The regression coefficients.

    Returns:
    torch.Tensor: The predicted target variable.
    """
    # Add intercept term
    ones_column = torch.ones(len(X), 1).to(device)
    X_int = torch.cat((ones_column, X), dim=1)
    
    # Return the predicted target variable
    return X_int @ beta

