import argparse
import time
from functools import partial
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import optax

from flax import linen as nn
from flax.training.train_state import TrainState
from jax.experimental.pjit import pjit



# Type annotations
Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
Initializer = Callable[[PRNGKey, Shape, DType], Array]

class Benchmark(nn.Module):
  activation: str

 # @nn.compact
  def __call__(self, input, weight, bias):
    y = jnp.dot(input, weight, precision=jax.lax.Precision.HIGHEST) + bias
    if self.activation == 'relu':
      return jax.nn.relu(y)
    elif self.activation == 'gelu':
      return jax.nn.gelu(y, approximate='True')
    elif self.activation == 'swish':
      return jax.nn.swish(y)
    else:
      return y

def run_benchmark(M,N,K, act, use_mixed):
  dtype = jnp.bfloat16 if use_mixed else jnp.float32
  key = jax.random.PRNGKey(12)
  x_shape = (M, K)
  w_shape = (K, N)
  b_shape = (N,)
  x_data = jax.random.uniform(key, shape=x_shape, dtype=dtype)
  w_data = jax.random.uniform(key, shape=w_shape, dtype=dtype)
  b_data = jax.random.uniform(key, shape=b_shape, dtype=dtype)
  
  timing_iters = 20
  warmup_iters = 20
  
  benchmark = Benchmark(act)
 
  @jax.jit
  def run(x, w, b):
    return benchmark(x, w, b)



  # Warmup runs
  for _ in range(timing_iters):
    _ = run(x_data, w_data, b_data)#.block_until_ready()

  st = time.time()
  for _ in range(timing_iters):
    _ = run(x_data, w_data, b_data)#.block_until_ready()
  elapsed_time = (time.time() - st) / timing_iters * 1000
  print("LOG >>> Execution Time (ms): %5.6f" % (elapsed_time,))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark matrix multiplication with activation functions.")
    parser.add_argument("--m", type=int, default=1024, help="Number of rows in the input matrix.")
    parser.add_argument("--n", type=int, default=1024, help="Number of columns in the output matrix.")
    parser.add_argument("--k", type=int, default=1024, help="Number of columns in the input matrix and rows in the weight matrix.")
    parser.add_argument("--a", type=str, default="relu", choices=["relu", "gelu", "swish"], help="Activation function type ('relu', 'gelu', 'swish').")
    parser.add_argument('--mixed', action='store_true', help='Enable mixed precision and bf16 compute type')
    args = parser.parse_args()
    run_benchmark(args.m, args.n, args.k, args.a, args.mixed)
