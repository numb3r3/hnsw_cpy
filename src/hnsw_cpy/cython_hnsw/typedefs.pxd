# cython: language_level=3

import  numpy as np
cimport numpy as np

np.import_array()

ctypedef np.uint8_t   uint8
ctypedef np.uint8_t   bool_t
ctypedef np.uint64_t  uint64

ctypedef np.int32_t   int32
ctypedef np.int64_t   int64

ctypedef np.float64_t float64
ctypedef np.float32_t float32
ctypedef np.float32_t real

ctypedef uint8[::1]      BoolVector
ctypedef float32[:, ::1] FloatMatrix
ctypedef float32[::1]    FloatVector
ctypedef int32[::1]      IntVector