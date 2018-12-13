from ctypes import *

dll = CDLL("DllTest.dll")
print(dll.add(10, 102))
