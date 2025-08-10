
import pyburn as py
import unittest 

class NetModuleTest(unittest.TestCase):
    def __init__():
        Net()
        NetNdarray()
        
class Net(py.wgpu.Module):
    def __init__(self):
        super().__init__()
        self.linear = py.wgpu.PyLinear(10,20) 
        self.relu = py.wgpu.PyRelu()

class NetNdarray(py.ndarray.Module):
    def __init__(self):
        super().__init__()
        self.linear = py.ndarray.PyLinear(10,20) 
        self.relu = py.ndarray.PyRelu()
