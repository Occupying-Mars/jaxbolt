import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Union, List, Callable
from functools import partialmethod, partial

class Tensor:
    def __init__(self, data, requires_grad=False):
        # Convert input data to JAX array
        self.data = jnp.array(data)
        self.requires_grad = requires_grad
        self.grad: Optional[jnp.ndarray] = None
        self._ctx = None

    @property
    def shape(self) -> Tuple:
        return self.data.shape

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    # Basic arithmetic operations
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)

    # Reverse arithmetic operations
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __rsub__(self, other): return Tensor(other) - self
    def __rtruediv__(self, other): return Tensor(other) / self

    # Neural network operations
    def relu(self):
        return Tensor(jax.nn.relu(self.data), requires_grad=self.requires_grad)

    def sigmoid(self):
        return Tensor(jax.nn.sigmoid(self.data), requires_grad=self.requires_grad)

    def tanh(self):
        return Tensor(jnp.tanh(self.data), requires_grad=self.requires_grad)

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def _jitted_matmul(a, b):
        return jnp.matmul(a, b)

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self._jitted_matmul(self.data, other.data), 
                     requires_grad=self.requires_grad or other.requires_grad)

    # Reduction operations
    @staticmethod
    @jax.jit
    def _jitted_reduce_sum(x, axis=None):
        return jnp.sum(x, axis=axis)

    def sum(self, axis=None):
        return Tensor(self._jitted_reduce_sum(self.data, axis), 
                     requires_grad=self.requires_grad)

    def mean(self, axis=None):
        return Tensor(jnp.mean(self.data, axis=axis), requires_grad=self.requires_grad)

    # Shape operations
    def reshape(self, *shape):
        return Tensor(jnp.reshape(self.data, shape), requires_grad=self.requires_grad)

    def transpose(self, axes=None):
        return Tensor(jnp.transpose(self.data, axes), requires_grad=self.requires_grad)

    # Gradient computation
    def backward(self, gradient=None, accumulate=False):
        if not self.requires_grad:
            return

        if gradient is None:
            gradient = jnp.ones_like(self.data)

        @jax.jit
        def grad_fn(x, grad):
            return jax.grad(lambda y: jnp.sum(y * grad))(x)

        new_grad = grad_fn(self.data, gradient)
        
        if accumulate and self.grad is not None:
            self.grad += new_grad
        else:
            self.grad = new_grad

    # Utility methods
    def numpy(self):
        return jnp.array(self.data)

    @staticmethod
    def zeros(*shape):
        return Tensor(jnp.zeros(shape))

    @staticmethod
    def ones(*shape):
        return Tensor(jnp.ones(shape))

    @staticmethod
    def randn(*shape):
        return Tensor(jax.random.normal(jax.random.PRNGKey(0), shape))

# Add common operations as methods
for op in ['log', 'exp', 'sqrt']:
    setattr(Tensor, op, partialmethod(lambda self, op: Tensor(getattr(jnp, op)(self.data), 
            requires_grad=self.requires_grad), op))

    @staticmethod
    @jax.vmap
    def _batch_matmul(a, b):
        return jnp.matmul(a, b)

    def batch_matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self._batch_matmul(self.data, other.data),
                     requires_grad=self.requires_grad or other.requires_grad)

    def checkpoint(self):
        """Memory efficient gradient computation using checkpointing"""
        @jax.checkpoint
        def forward(x):
            return self.data

        return Tensor(forward(self.data), requires_grad=self.requires_grad)

    def to_device(self, device=None):
        """Transfer tensor to specific device"""
        if device is None:
            return self
        return Tensor(jax.device_put(self.data, device), 
                     requires_grad=self.requires_grad)

    @staticmethod
    def get_default_device():
        return jax.devices()[0]

    @staticmethod
    def parallel_map(fn, tensors):
        """Parallel execution across multiple devices"""
        mapped_fn = jax.pmap(fn)
        data = [t.data for t in tensors]
        result = mapped_fn(jnp.stack(data))
        return Tensor(result)

    @staticmethod
    def from_numpy(array, chunks=None):
        """Efficient data loading with optional chunking"""
        if chunks is None:
            return Tensor(jnp.array(array))
        
        # Load data in chunks to manage memory
        chunks = jnp.array_split(array, chunks)
        return [Tensor(chunk) for chunk in chunks]

    def chunk_operation(self, operation, chunk_size=1000):
        """Perform operations on large tensors in chunks"""
        chunks = jnp.array_split(self.data, 
                               max(1, len(self.data) // chunk_size))
        results = [operation(chunk) for chunk in chunks]
        return Tensor(jnp.concatenate(results), 
                     requires_grad=self.requires_grad)
