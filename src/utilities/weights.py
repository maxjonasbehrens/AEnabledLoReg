import numpy as np
def ms_batch_weights(encoded, encoded_external, encoded_target, sigma=0.5, k_nearest=0.5, kernel='gaussian'):
    """
    Calculate the weights for the target and external data based on the encoded data.

    Parameters:
    encoded (numpy.ndarray): The encoded data batch.
    encoded_external (numpy.ndarray): The encoded external data.
    encoded_target (numpy.ndarray): The encoded target data.
    sigma (float, optional): The standard deviation for the Gaussian kernel. Default is 0.5.
    k_nearest (int, optional): The number of nearest neighbors to consider. Default is 10.
    kernel (str, optional): The kernel function to use. Supported values are 'gaussian' and 'tricube'. Default is 'gaussian'.

    Returns:
    weights_target (numpy.ndarray): The weights for the target data.
    weights_external (numpy.ndarray): The weights for the external data.

    Raises:
    ValueError: If the specified kernel is not supported.

    """

    # calculate the euclidean distance between the encoded data and the external and target data
    dist_external = np.linalg.norm(encoded - encoded_external, axis=1)
    dist_target = np.linalg.norm(encoded - encoded_target, axis=1)

    # Calculate the weights for the target data
    k_nearest_tar = int(k_nearest * encoded_target.shape[0])
    kth_nearest_t = np.sort(dist_target)[k_nearest_tar]

    if kth_nearest_t == 0:
        kth_nearest_t = 1
    
    weights_target = dist_target / kth_nearest_t

    # Calculate the weights for the external data
    k_nearest_ext = int(k_nearest * encoded_external.shape[0])
    kth_nearest_ext = np.sort(dist_external)[k_nearest_ext]

    if kth_nearest_ext == 0:
        kth_nearest_ext = 1
        
    weights_external = dist_external / kth_nearest_ext

    # Define the Gaussian kernel
    def gaussian_kernel(dist_target, dist_external, sigma):
        return np.exp(-dist_target**2 / (2 * sigma**2)), np.exp(-dist_external**2 / (2 * sigma**2))
    
    # Define the Tricube kernel
    def tricube_kernel(dist_target, dist_external):
        return np.where(dist_target <= 1, (1 - np.abs(dist_target)**3)**3, 0), np.where(dist_external <= 1, (1 - np.abs(dist_external)**3)**3, 0)

    # If clause to determine the kernel
    if kernel == 'gaussian':
        weights_target, weights_external = gaussian_kernel(weights_target, weights_external, sigma)
    # elif for tricube kernel
    elif kernel == 'tricube':
        weights_target, weights_external = tricube_kernel(weights_target, weights_external)
    else:
        raise ValueError("Kernel not supported. Please choose 'gaussian' or 'tricube'.")

    return weights_target, weights_external

def ss_batch_weights(encoded, encoded_all, sigma=1, k_nearest=0.3, kernel='gaussian'):
    """
    Calculate weights for the data points in `encoded_all` relative to a reference `encoded`, 
    using a specified kernel function.

    Parameters:
    - encoded (numpy.ndarray): A numpy array representing the reference data point(s).
    - encoded_all (numpy.ndarray): A numpy array representing the data points for which weights are computed.
    - sigma (float, optional): The standard deviation parameter for the Gaussian kernel. Default is 1.
    - k_nearest (int, optional): The fraction of the total data in `encoded_all` to consider for scaling weights. 
                                 Expressed as an integer multiplied by `encoded_all.shape[0]`. Default is 0.3.
    - kernel (str, optional): The kernel function to apply. Supported values are:
        - 'gaussian': Uses a Gaussian kernel.
        - 'tricube': Uses a tricube kernel. 
      Default is 'gaussian'.

    Returns:
    - weights_all (numpy.ndarray): A numpy array of computed weights for `encoded_all` data points.

    Raises:
    - ValueError: If an unsupported kernel is specified.

    Notes:
    - The distance metric used is the Euclidean distance.
    - Weights are normalized using the k-th nearest
    neighbor distance to scale them within the kernel function.
    """
    
    # Calculate the euclidean distance between the encoded data and the external and target data
    dist_all = np.linalg.norm(encoded - encoded_all, axis=1)

    # Calculate the weights for the target data
    k_nearest_all = int(k_nearest * encoded_all.shape[0])
    kth_nearest_t = np.sort(dist_all)[k_nearest_all]

    if kth_nearest_t == 0:
        kth_nearest_t = 1

    weights_all = dist_all / kth_nearest_t

    # Define the Gaussian kernel
    def gaussian_kernel(dist_all, sigma):
        return np.exp(-dist_all**2 / (2 * sigma**2))

    # Define the Tricube kernel
    def tricube_kernel(dist_all):
        return np.where(dist_all <= 1, (1 - np.abs(dist_all)**3)**3, 0)

    # If clause to determine the kernel
    if kernel == 'gaussian':
        weights_all = gaussian_kernel(weights_all, sigma)
    elif kernel == 'tricube':
        weights_all = tricube_kernel(weights_all)
    else:
        raise ValueError("Kernel not supported. Please choose 'gaussian' or 'tricube'.")

    return weights_all


import torch

def ms_torch_weights(encoded, 
                  encoded_external, 
                  encoded_target, 
                  sigma=0.5,  
                  k_nearest=0.5, 
                  kernel='gaussian'):
    """
    Calculate the weights for the target and external data based on the encoded data.

    Parameters:
    encoded (torch.Tensor): The encoded data batch.
    encoded_external (torch.Tensor): The encoded external data.
    encoded_target (torch.Tensor): The encoded target data.
    sigma (float, optional): The standard deviation for the Gaussian kernel. Default is 0.5.
    k_nearest (int, optional): The number of nearest neighbors to consider. Default is 10.
    kernel (str, optional): The kernel function to use. Supported values are 'gaussian' and 'tricube'. Default is 'gaussian'.

    Returns:
    weights_target (torch.Tensor): The weights for the target data.
    weights_external (torch.Tensor): The weights for the external data.

    Raises:
    ValueError: If the specified kernel is not supported.

    """
    # Calculate the euclidean distance between the encoded data and the external and target data
    dist_external = torch.norm(encoded - encoded_external, dim=1)
    dist_target = torch.norm(encoded - encoded_target, dim=1)

    # Calculate the weights for the target data

    # Length of the target data
    n_target = encoded_target.shape[0]

    # K nearest based on the length of the target data
    k_nearest_tar = int(k_nearest * n_target)

    kth_nearest_t = torch.sort(dist_target)[0][k_nearest_tar]

    if kth_nearest_t == 0:
        kth_nearest_t = 1

    kernel_target = dist_target / kth_nearest_t

    # Calculate the weights for the external data

    # Length of the external data
    n_external = encoded_external.shape[0]

    # K nearest based on the length of the external data
    k_nearest_ext = int(k_nearest * n_external)

    kth_nearest_ext = torch.sort(dist_external)[0][k_nearest_ext]

    if kth_nearest_ext == 0:
        kth_nearest_ext = 1

    kernel_external = dist_external / kth_nearest_ext

    # Define the Gaussian kernel
    def gaussian_kernel(dist, sigma):
        return torch.exp(-dist**2 / (2 * sigma**2))

    # Define the Tricube kernel
    def tricube_kernel(dist):
        return torch.where(dist <= 1, (1 - torch.abs(dist)**3)**3, torch.zeros_like(dist))

    # If clause to determine the kernel
    if kernel == 'gaussian':
        weights_tar = gaussian_kernel(kernel_target, sigma)
        weights_ext = gaussian_kernel(kernel_external, sigma)
    elif kernel == 'tricube':
        weights_tar = tricube_kernel(kernel_target)
        weights_ext = tricube_kernel(kernel_external)
    else:
        raise ValueError("Kernel not supported. Please choose 'gaussian' or 'tricube'.")

    return weights_tar, weights_ext

def ss_torch_weights(encoded, encoded_all, sigma=1, k_nearest=0.3, kernel='gaussian'):
    """
    Calculate weights for the data points in `encoded_all` relative to a reference `encoded`, 
    using a specified kernel function.

    Parameters:
    - encoded (torch.Tensor): A tensor representing the reference data point(s).
    - encoded_all (torch.Tensor): A tensor representing the data points for which weights are computed.
    - sigma (float, optional): The standard deviation parameter for the Gaussian kernel. Default is 1.
    - k_nearest (int, optional): The fraction of the total data in `encoded_all` to consider for scaling weights. 
                                 Expressed as an integer multiplied by `encoded_all.shape[0]`. Default is 0.3.
    - kernel (str, optional): The kernel function to apply. Supported values are:
        - 'gaussian': Uses a Gaussian kernel.
        - 'tricube': Uses a tricube kernel. 
      Default is 'gaussian'.

    Returns:
    - weights_all (torch.Tensor): A tensor of computed weights for `encoded_all` data points.

    Raises:
    - ValueError: If an unsupported kernel is specified.

    Notes:
    - The distance metric used is the Euclidean distance.
    - Weights are normalized using the k-th nearest neighbor distance to scale them within the kernel function.
    """

    # Calculate the euclidean distance between the encoded data and the external and target data
    dist_all = torch.norm(encoded - encoded_all, dim=1)

    # Calculate the weights for the target data
    k_nearest_all = int(k_nearest * encoded_all.shape[0])
    kth_nearest_t = torch.sort(dist_all)[0][k_nearest_all]

    if kth_nearest_t == 0:
        kth_nearest_t = 1

    weights_all = dist_all / kth_nearest_t

    # Define the Gaussian kernel
    def gaussian_kernel(dist, sigma):
        return torch.exp(-dist**2 / (2 * sigma**2))

    # Define the Tricube kernel
    def tricube_kernel(dist):
        return torch.where(dist <= 1, (1 - torch.abs(dist)**3)**3, torch.zeros_like(dist))

    # If clause to determine the kernel
    if kernel == 'gaussian':
        weights_all = gaussian_kernel(weights_all, sigma)
    elif kernel == 'tricube':
        weights_all = tricube_kernel(weights_all)
    else:
        raise ValueError("Kernel not supported. Please choose 'gaussian' or 'tricube'.")

    return weights_all

