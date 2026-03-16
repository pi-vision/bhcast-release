import torch
import torch.nn as nn
from math import prod
from abc import abstractmethod

class Transform(torch.nn.Module):
    """
    Applies transforms or inverse transforms to 
    model inputs or outputs, respectively
    """
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def inverse_transform(self):
        pass

    @abstractmethod
    def cuda(self):
        pass

    @abstractmethod
    def cpu(self):
        pass

    @abstractmethod
    def to(self, device):
        pass


def count_tensor_params(tensor, dims=None):
    """Returns the number of parameters (elements) in a single tensor, optionally, along certain dimensions only

    Parameters
    ----------
    tensor : torch.tensor
    dims : int list or None, default is None
        if not None, the dimensions to consider when counting the number of parameters (elements)
    
    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    if dims is None:
        dims = list(tensor.shape)
    else:
        dims = [tensor.shape[d] for d in dims]
    n_params = prod(dims)
    if tensor.is_complex():
        return 2*n_params
    return n_params

class ZeroOneNormalizer(Transform):
    """
    ZeroOneNormalizer normalizes data to lie in the range [0, 1]
    using a min-max scaling: (x - min) / (max - min).
    
    Parameters
    ----------
    min_val : torch.Tensor or None
        The minimum values. Should include a singleton batch dimension if needed.
    max_val : torch.Tensor or None
        The maximum values.
    eps : float, default=1e-7
        A small constant added to the denominator for numerical stability.
    dim : int or list of int, default=None
        Dimensions to reduce over when computing the min and max.
        For instance, for data of shape (batch_size, channels, height, width),
        to compute per-channel min/max, you might reduce over [0, 2, 3].
    mask : torch.Tensor or None, default=None
        A tensor of the same spatial shape as a sample (or broadcastable to it)
        with value 0 where data should be ignored and 1 everywhere else.
    normalize_channels : list, slice, or None, default=None
        Indices of channels to normalize. If None, normalize all channels.
    """
    def __init__(self, min_val=None, max_val=None, eps=1e-7, dim=None, mask=None, normalize_channels=None):
        super().__init__()
        self.register_buffer("min", min_val)
        self.register_buffer("max", max_val)
        self.register_buffer("mask", mask)
        self.eps = eps
        # If min is provided, record its ndim.
        if min_val is not None:
            self.ndim = min_val.ndim
        # Allow dim to be an int.
        if isinstance(dim, int):
            dim = [dim]
        self.dim = dim
        self.n_elements = 0
        self.normalize_channels = normalize_channels

    def update_min_max(self, data_batch):
        # Only operate on selected channels
        if self.normalize_channels is not None:
            data = data_batch[:, self.normalize_channels]
        else:
            data = data_batch
        """
        Compute the min and max of data_batch over the specified dims.
        If a mask is provided, masked elements are ignored in the min/max.
        """
        # When no dims are provided, reduce over all elements.
        if self.dim is None:
            dims = None
        else:
            dims = self.dim

        if self.mask is None:
            self.min = torch.amin(data, dim=dims, keepdim=True)
            self.max = torch.amax(data, dim=dims, keepdim=True)
            self.n_elements = data.numel() if dims is None else 1
        else:
            mask_bool = self.mask.to(torch.bool)
            inf = torch.tensor(float("inf"), dtype=data.dtype, device=data.device)
            ninf = torch.tensor(float("-inf"), dtype=data.dtype, device=data.device)
            data_for_min = torch.where(mask_bool, data, inf)
            data_for_max = torch.where(mask_bool, data, ninf)
            if dims is None:
                self.min = torch.amin(data_for_min, keepdim=True)
                self.max = torch.amax(data_for_max, keepdim=True)
            else:
                self.min = torch.amin(data_for_min, dim=dims, keepdim=True)
                self.max = torch.amax(data_for_max, dim=dims, keepdim=True)
            self.n_elements = torch.count_nonzero(self.mask)
    
    def incremental_update_min_max(self, data_batch):
        if self.normalize_channels is not None:
            data = data_batch[:, self.normalize_channels]
        else:
            data = data_batch
        """
        Incrementally update the min and max with a new data_batch.
        This is done by taking the elementwise minimum/maximum between the
        current stored value and the new batch's min/max.
        """
        if self.mask is None:
            if self.dim is None:
                new_min = torch.amin(data, keepdim=True)
                new_max = torch.amax(data, keepdim=True)
            else:
                new_min = torch.amin(data, dim=self.dim, keepdim=True)
                new_max = torch.amax(data, dim=self.dim, keepdim=True)
        else:
            mask_bool = self.mask.to(torch.bool)
            inf = torch.tensor(float("inf"), dtype=data.dtype, device=data.device)
            ninf = torch.tensor(float("-inf"), dtype=data.dtype, device=data.device)
            data_for_min = torch.where(mask_bool, data, inf)
            data_for_max = torch.where(mask_bool, data, ninf)
            if self.dim is None:
                new_min = torch.amin(data_for_min, keepdim=True)
                new_max = torch.amax(data_for_max, keepdim=True)
            else:
                new_min = torch.amin(data_for_min, dim=self.dim, keepdim=True)
                new_max = torch.amax(data_for_max, dim=self.dim, keepdim=True)

        if self.n_elements:
            self.min = torch.minimum(self.min, new_min)
            self.max = torch.maximum(self.max, new_max)
        else:
            self.min = new_min
            self.max = new_max
        self.n_elements += data.numel() if self.dim is None else 1

    def fit(self, data_batch):
        self.update_min_max(data_batch)

    def partial_fit(self, data_batch, batch_size=1):
        """
        Incrementally update the normalizer using data_batch.
        """
        if 0 in list(data_batch.shape):
            return
        n_samples = len(data_batch)
        count = 0
        while count < n_samples:
            samples = data_batch[count : count + batch_size]
            if self.n_elements:
                self.incremental_update_min_max(samples)
            else:
                self.update_min_max(samples)
            count += batch_size

    def transform(self, x):
        # Only normalize selected channels
        if self.normalize_channels is not None:
            x_norm = x.clone()
            x_norm[:, self.normalize_channels] = (x[:, self.normalize_channels] - self.min) / (self.max - self.min + self.eps)
            return x_norm
        else:
            return (x - self.min) / (self.max - self.min + self.eps)

    def inverse_transform(self, x):
        if self.normalize_channels is not None:
            x_inv = x.clone()
            x_inv[:, self.normalize_channels] = x[:, self.normalize_channels] * (self.max - self.min + self.eps) + self.min
            return x_inv
        else:
            return x * (self.max - self.min + self.eps) + self.min

    def forward(self, x):
        """
        Forward pass that applies the 0–1 normalization.
        """
        return self.transform(x)

    def cuda(self):
        self.min = self.min.cuda()
        self.max = self.max.cuda()
        if self.mask is not None:
            self.mask = self.mask.cuda()
        return self

    def cpu(self):
        self.min = self.min.cpu()
        self.max = self.max.cpu()
        if self.mask is not None:
            self.mask = self.mask.cpu()
        return self

    def to(self, device):
        self.min = self.min.to(device)
        self.max = self.max.to(device)
        if self.mask is not None:
            self.mask = self.mask.to(device)
        return self

    @classmethod
    def from_dataset(cls, dataset, dim=None, keys=None, mask=None):
        """
        Create normalizer instances for the given dataset. Each element of the dataset
        is expected to be a dictionary mapping keys to samples.
        """
        for i, data_dict in enumerate(dataset):
            if not i:
                if not keys:
                    keys = data_dict.keys()
        instances = {key: cls(dim=dim, mask=mask) for key in keys}
        for i, data_dict in enumerate(dataset):
            for key, sample in data_dict.items():
                if key in keys:
                    # Assume that sample is a torch.Tensor. Add batch dimension.
                    instances[key].partial_fit(sample.unsqueeze(0))
        return instances

class LogZeroOneNormalizer(ZeroOneNormalizer):
    """
    A subclass of ZeroOneNormalizer that first converts data to log-space
    and then applies min-max normalization (mapping data to [0,1]) 
    in that log-space.

    For strictly positive data x, we do:
        log_x = log(x + log_offset)

    The min and max of log_x are used for the 0–1 scaling:
        scaled = (log_x - min(log_x)) / (max(log_x) - min(log_x))

    Inverse transform:
        log_x = scaled * [max(log_x)-min(log_x)] + min(log_x)
        x = exp(log_x) - log_offset
    """

    def __init__(
        self, 
        min_val=None, 
        max_val=None, 
        eps=1e-7, 
        dim=None, 
        mask=None,
        log_offset=1e-7,
        normalize_channels=None
    ):
        """
        Parameters
        ----------
        min_val : torch.Tensor or None, default=None
            The minimum of the log-transformed data, if known beforehand.
        max_val : torch.Tensor or None, default=None
            The maximum of the log-transformed data, if known beforehand.
        eps : float, default=1e-7
            A small constant added to the denominator for numerical stability.
        dim : int or list of int, default=None
            Dimensions to reduce over when computing min and max in log-space.
        mask : torch.Tensor or None, default=None
            A binary mask with shape broadcastable to x. Where mask==0, data are ignored.
        log_offset : float, default=1e-7
            Added inside the log to avoid log(0). Must be > 0 for strictly positive data.
        normalize_channels : list, slice, or None, default=None
            Indices of channels to normalize. If None, normalize all channels.
        """
        super().__init__(
            min_val=min_val,
            max_val=max_val,
            eps=eps,
            dim=dim,
            mask=mask,
            normalize_channels=normalize_channels
        )
        self.log_offset = log_offset
    
    def fit(self, data_batch):
        # Only operate on selected channels
        if self.normalize_channels is not None:
            log_data_batch = torch.log(data_batch[:, self.normalize_channels] + self.log_offset)
        else:
            log_data_batch = torch.log(data_batch + self.log_offset)
        super().fit(log_data_batch)

    def partial_fit(self, data_batch, batch_size=1):
        # Skip empty batches
        if 0 in list(data_batch.shape):
            return
        n_samples = len(data_batch)
        count = 0
        while count < n_samples:
            samples = data_batch[count : count + batch_size]
            # Only operate on selected channels
            if self.normalize_channels is not None:
                log_samples = torch.log(samples[:, self.normalize_channels] + self.log_offset)
            else:
                log_samples = torch.log(samples + self.log_offset)
            super().partial_fit(log_samples, batch_size=len(log_samples))
            count += batch_size

    def transform(self, x):
        # Only normalize selected channels
        if self.normalize_channels is not None:
            x_norm = x.clone()
            x_log = torch.log(x[:, self.normalize_channels] + self.log_offset)
            x_norm[:, self.normalize_channels] = super().transform(x_log)
            return x_norm
        else:
            log_x = torch.log(x + self.log_offset)
            return super().transform(log_x)

    def inverse_transform(self, x):
        if self.normalize_channels is not None:
            x_inv = x.clone()
            # Inverse normalization in log space
            log_x = super().inverse_transform(x[:, self.normalize_channels])
            x_inv[:, self.normalize_channels] = torch.exp(log_x) - self.log_offset
            return x_inv
        else:
            log_x = super().inverse_transform(x)
            return torch.exp(log_x) - self.log_offset

    @classmethod
    def from_dataset(cls, dataset, dim=None, keys=None, mask=None, log_offset=1e-7):
        """
        Create a set of LogZeroOneNormalizer instances from a dataset.

        Parameters
        ----------
        dataset : Iterable of dict
            Each element is a dictionary {key: sample_tensor}.
        dim : int or list of int, default=None
            Dimensions to reduce over for min-max in log-space.
        keys : list of str or None, default=None
            If None, normalizers are built for all keys in the dataset's first entry.
        mask : torch.Tensor or None, default=None
            A mask to ignore some data points.
        log_offset : float, default=1e-7
            Offset for the log transform to avoid log(0).

        Returns
        -------
        dict of str -> LogZeroOneNormalizer
            A dictionary mapping each key to a fitted LogZeroOneNormalizer.
        """
        # If keys are not specified, infer them from the first sample in the dataset
        for i, data_dict in enumerate(dataset):
            if not i:
                if not keys:
                    keys = data_dict.keys()
                break

        instances = {
            key: cls(dim=dim, mask=mask, log_offset=log_offset)
            for key in keys
        }

        # Fit each normalizer on the dataset (in log-space)
        for data_dict in dataset:
            for key, sample in data_dict.items():
                if key in keys:
                    # Expand sample with batch dimension at dim=0
                    instances[key].partial_fit(sample.unsqueeze(0))

        return instances
    

