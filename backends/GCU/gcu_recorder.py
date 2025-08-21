import time, threading
import pyefml
from backends.recorder import recorder

class gcu_recorder(recorder):
    def __init__(self):
        super().__init__()
        self.handle = None
        self.name = None
        self.memory = None

    @property
    def dev_name(self):
        return self.name
    
    @property
    def total_memory(self):
        return self.memory

    def record_init(self):
        # pyefml.efmlInit()
        pyefml.efmlInitV2()

    def set_device(self, index):
        self.index = index
        self.handle = pyefml.efmlDeviceGetHandleByIndex(idx = index)
        self.name = pyefml.efmlDeviceGetName(self.handle)#.decode('utf-8')
        self.memory = pyefml.efmlDeviceGetMemoryInfo(self.handle).total

    def get_utilization_rates(self):
        Utilization = pyefml.efmlDeviceGetUtilizationRates(self.handle)
        util = Utilization.gcu
        return util
    
    def get_power_usage(self):
        return pyefml.efmlDeviceGetPowerUsage(self.handle) / 1000  # unit: w

    def record_free(self):
        pyefml.efmlShutdown()

