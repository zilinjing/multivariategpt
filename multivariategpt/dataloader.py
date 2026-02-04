import torch
import numpy as np
import os
import random

class DataLoader:
    def __init__(self,config,ti_path,tv_path,split='train'):
        self.B = config.batch_size
        self.T = config.block_size
        self.position = 0
        self.split = split
        self.ti_path = ti_path
        self.tv_path = tv_path
        self.device = config.device
        self.epoch = 0
        # peek at the tokens but can't store due to memmap memory leak issue
        ti = np.memmap(self.ti_path, dtype=np.uint16, mode='r')
        self.total_tokens = len(ti)

    @classmethod
    def from_params(cls, b, t, device, ti_path, tv_path, split='train'):
        class Config:
            batch_size = b
            block_size = t
            device = device

        return cls(Config(), ti_path, tv_path, split)

    def get_all(self):
        return np.memmap(self.ti_path, dtype=np.uint16, mode='r'),np.memmap(self.tv_path, dtype=np.uint16, mode='r')
    

    def next_batch(self):
        
        # have to recreate this memmap every time to avoid memory leak. there is a numpy github issue
        ti = np.memmap(self.ti_path, dtype=np.uint16, mode='r')
        tv = np.memmap(self.tv_path, dtype=np.float32, mode='r')
        bufi = torch.tensor(ti[self.position:self.position+self.B*self.T+1],dtype=torch.long)
        bufv = torch.tensor(tv[self.position:self.position+self.B*self.T+1],dtype=torch.float32)
        xi = (bufi[:-1].view(self.B,self.T))
        xv = (bufv[:-1].view(self.B,self.T))
        yi = (bufi[1:].view(self.B,self.T))
        yv = (bufv[1:].view(self.B,self.T))
        self.position += self.B*self.T

        # if exceed len then loop back around, epoch completed
        if self.position + (self.B*self.T+1) > len(ti):
            # to make sure batches don't have the same cut-points on the data
            # set for each epoch, introduce a random offset from 0 to min(B*T,
            # len(tokens) - B*T - 1)
            self.position = random.randint(0,min(self.B*self.T,self.total_tokens-self.B*self.T-1))
            # self.position = 0
            self.epoch += 1

        # TODO: revisit this pin memory
        if  'cuda' in self.device:
            xi,yi = xi.pin_memory().to(self.device, non_blocking=True),yi.pin_memory().to(self.device, non_blocking=True)
            xv,yv = xv.pin_memory().to(self.device, non_blocking=True),yv.pin_memory().to(self.device, non_blocking=True)
        else:
            xi, yi = xi.to(self.device),yi.to(self.device)
            xv, yv = xv.to(self.device),yv.to(self.device)
        return xi,xv,yi,yv
    
    def batch_no_step(self):
        
        # have to recreate this memmap every time to avoid memory leak. there is a numpy github issue
        ti = np.memmap(self.ti_path, dtype=np.uint16, mode='r')
        tv = np.memmap(self.tv_path, dtype=np.float32, mode='r')
        bufi = torch.tensor(ti[self.position:self.position+self.B*self.T+1],dtype=torch.long)
        bufv = torch.tensor(tv[self.position:self.position+self.B*self.T+1],dtype=torch.float32)
        xi = (bufi[:-1].view(self.B,self.T))
        xv = (bufv[:-1].view(self.B,self.T))
        yi = (bufi[1:].view(self.B,self.T))
        yv = (bufv[1:].view(self.B,self.T))
        
        # TODO: revisit this pin memory
        if  'cuda' in self.device:
            xi,yi = xi.pin_memory().to(self.device, non_blocking=True),yi.pin_memory().to(self.device, non_blocking=True)
            xv,yv = xv.pin_memory().to(self.device, non_blocking=True),yv.pin_memory().to(self.device, non_blocking=True)
        else:
            xi, yi = xi.to(self.device),yi.to(self.device)
            xv, yv = xv.to(self.device),yv.to(self.device)
        return xi,xv,yi,yv
    
    

class DataLoaderDDP:
    def __init__(self,config,ti_path,tv_path,process_rank,num_processes,split='train'):
        self.B = config.batch_size
        self.T = config.block_size
        self.split = split
        self.ti_path = ti_path
        self.tv_path = tv_path
        self.device = config.device
        self.epoch = 0
        self.process_rank = process_rank
        self.num_processes = num_processes
        # peek at the tokens but can't store due to memmap memory leak issue
        ti = np.memmap(self.ti_path, dtype=np.uint16, mode='r')
        self.total_tokens = len(ti)
        self.position = self.B*self.T*self.process_rank # stride the processes


    def next_batch(self):
        
        # have to recreate this memmap every time to avoid memory leak. there is a numpy github issue
        ti = np.memmap(self.ti_path, dtype=np.uint16, mode='r')
        tv = np.memmap(self.tv_path, dtype=np.float32, mode='r')
        bufi = torch.tensor(ti[self.position:self.position+self.B*self.T+1],dtype=torch.long)
        bufv = torch.tensor(tv[self.position:self.position+self.B*self.T+1],dtype=torch.float32)
        xi = (bufi[:-1].view(self.B,self.T))
        xv = (bufv[:-1].view(self.B,self.T))
        yi = (bufi[1:].view(self.B,self.T))
        yv = (bufv[1:].view(self.B,self.T))
        self.position += self.B*self.T*self.num_processes

        # if exceed len then loop back around, epoch completed
        if self.position + (self.B*self.T*self.num_processes+1) > len(ti):
            # to make sure batches don't have the same cut-points on the data
            # set for each epoch shift by epoch
            # this is not as nice as the non ddp version, but i can't think of an easy way to synchronize strides
            self.position = min(self.epoch + self.B*self.T*self.process_rank,self.total_tokens-self.B*self.T*self.num_processes-1)
            # self.position = 0
            self.epoch += 1

        # TODO: revisit this pin memory
        if  'cuda' in self.device:
            xi,yi = xi.pin_memory().to(self.device, non_blocking=True),yi.pin_memory().to(self.device, non_blocking=True)
            xv,yv = xv.pin_memory().to(self.device, non_blocking=True),yv.pin_memory().to(self.device, non_blocking=True)
        else:
            xi, yi = xi.to(self.device),yi.to(self.device)
            xv, yv = xv.to(self.device),yv.to(self.device)
        return xi,xv,yi,yv
    
