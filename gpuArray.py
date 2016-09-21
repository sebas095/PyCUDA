import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy

mat_gpu = gpuarray.to_gpu(numpy.random.randn(4, 4).astype(numpy.float32))
mat_doubled = (2 * mat_gpu).get()

print mat_doubled
print mat_gpu
