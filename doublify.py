import numpy
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# HOST
mat_cpu = numpy.random.randn(4, 4)
mat_cpu = mat_cpu.astype(numpy.float32)

#HOST TO DEVICE
mat_gpu = cuda.mem_alloc(mat_cpu.nbytes)
cuda.memcpy_htod(mat_gpu, mat_cpu)

#Executing a Kernel
mod = SourceModule("""
    __global__ void doublify(float *mat) {
        int idx = threadIdx.x + threadIdx.y * 4;
        mat[idx] *= 2;
    }
""")

func = mod.get_function("doublify")
func(mat_gpu, block=(4, 4, 1))

#DEVICE TO HOST
mat_doubled = numpy.empty_like(mat_cpu)
cuda.memcpy_dtoh(mat_doubled, mat_gpu)

print mat_doubled
print mat_cpu
