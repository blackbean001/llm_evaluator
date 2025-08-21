import os
import time, importlib
import multiprocessing as mp
from abc import ABC, abstractmethod

import csv

def _record_proc(cconn, backend_type:str):
    recorder = getattr(importlib.import_module(
        'backends.' + backend_type + '.' + backend_type.lower() + '_recorder'), backend_type.lower() + '_recorder')()
    recorder.record_init()
    records = {'dev_name': ..., 'total_memory': ..., 'util': [], 'power': []}
    msg = None
    while msg != 'exit':
        if cconn.poll(0):
            msg = cconn.recv()
        if msg == 'set':
            msg = None
            index = cconn.recv()
            recorder.set_device(index)
            records['dev_name'] = recorder.dev_name
            records['total_memory'] = recorder.total_memory
        elif msg == 'start':
            util = recorder.get_utilization_rates()
            power = recorder.get_power_usage()
            records['util'].append(util)
            records['power'].append(power)
            time.sleep(0.05)
        elif msg == 'stop':
            msg = None
            cconn.send(records)
            #records['dev_name'] = ''
            #records['total_memory'] = 0
            #records['util'] = []
            #records['power'] = []
    recorder.record_free()

class recorder(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def dev_name(self):
        pass

    @property
    @abstractmethod
    def total_memory(self):
        pass

    @abstractmethod
    def record_init(self):
        pass

    @abstractmethod
    def set_device(self, index):
        pass

    @abstractmethod
    def get_utilization_rates(self):
        pass

    @abstractmethod
    def get_power_usage(self):
        pass

    @abstractmethod
    def record_free(self):
        pass

class recording(object):
    def __init__(self, backend_type, model_name) -> None:
        self.running = False
        self.subprocess = None
        self.pconn = None
        self.cconn = None

        self.records = None
        #print(f"backend_type:{backend_type}")
        self.init(backend_type)
        self.model_name = model_name

    def init(self, backend_type):
        self.pconn, self.cconn = mp.Pipe()
        self.subprocess = mp.Process(
            target = _record_proc, 
            args = (self.cconn, backend_type))
        self.subprocess.start()

    def set_device(self, index):
        self.index = index
        self.pconn.send('set')
        self.pconn.send(index)

    def trigger(self):
        if self.running:
            self.running = False
            self.pconn.send('stop')
            self.records = self.pconn.recv()
            
        else:
            self.running = True
            self.pconn.send('start')

    def free(self):
        self.pconn.send('exit')
        self.subprocess.join()
        self.pconn.close()
        self.cconn.close()

    def get_records_summary(self):
        if self.records == None:
            self.records = {'dev_name': 'unknown', 'total_memory': 0, 'util': [0,], 'power': [0,]}

        summary = {
            "name": self.records['dev_name'],
            "total_memory(MB)": self.records['total_memory'] / 1024 / 1024,
            "utilization_rates(%)": {
                "max": max(self.records['util']), 
                "ave": sum(self.records['util']) / len(self.records['util'])},
            "power_usage(W)": {
                "max": max(self.records['power']), 
                "ave": sum(self.records['power']) / len(self.records['power'])}
        }

        self.records = None
        return summary

    def get_records(self):
        return self.records
    
    def records_dump(self, mode, records, model):
        
        # 确保records不是None
        if records is None:
            print("No records to dump.")
            return
        
        # 获取util和power列表
        util = records['util']
        power = records['power']
        

        dirname = f"./results/csv/{self.model_name}/"
        filename = dirname + f"device{self.index}_records_{mode}_{model}.csv"
        os.makedirs(dirname, exist_ok=True)
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['utilization_rate', 'power_usage']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            # 假设util和power列表长度相同
            for i in range(len(util)):
                writer.writerow({
                    'utilization_rate': util[i],
                    'power_usage': power[i]
                })

        print(f"Records dumped to {filename}.")

