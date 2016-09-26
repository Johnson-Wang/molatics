# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 08:46:44 2013

@author: xwangan
"""
import numpy as np
import os
from string import maketrans

from realmd.information import error,warning


def bool2string(bool_num):
    if not type(bool_num) is bool:
        return None
    if bool_num ==True:
        return ".true."
    else:
        return ".false."

def string2bool(bool_str):
    if bool_str.find("true")!=-1:
        b=True
    elif bool_str.find("false")!=-1:
        b=False
    else:
        b=None
    return b

class Settings():
    def __init__(self, options,option_list, args,config_file=None):
        self._options=options
        self._option_list=option_list
        self.input_filename=None
        if len(args) > 0:
            self.input_filename=args[0]
        self._conf={}
        self.file_format="l"
        self.atom_type=None
        self.step_range=slice(None)
        self.is_average_input=False
        self.is_normalize=True
        self.sample_length=500
        self.correlation_length=400
        self.smooth_tune=1.0
        self.time_step=1.0
        self.is_average_output=False
        self.is_convert_velocity = False
        self.direction_tensor = None
        if options is not None:
            self.read_options()
        if config_file is not None:
            self.read_conf_file(config_file)
        self.set_settings()
        self.parameter_coupling()

    def read_options(self):
        for opt in self._option_list:
            if opt.dest=="file_format":
                self._conf["file_format"]=self._options.file_format.strip(" \'\"")
            
            if opt.dest=="atom_type":
                self._conf["atom_type"]=self._options.atom_type.strip(" \'\"")
                
            if opt.dest=="out_filename":
                self._conf["out_filename"]=self._options.out_filename.strip(" \'\"")
            
            if opt.dest=="step_range":
                ran=self._options.step_range.strip(" \'\"")
                self._conf["step_range"]=ran
                
            if opt.dest=="is_average_input":
                isavi=self._options.is_average_input
                self._conf["is_average_input"]=bool2string(isavi)
                
            if opt.dest=="is_normalize":
                isnorm=self._options.is_normalize
                self._conf["is_normalize"]=bool2string(isnorm)
                
            if opt.dest=="is_summation":
                issum=self._options.is_summation
                self._conf["is_summation"]=bool2string(issum)
            
            if opt.dest=="is_smoothing":
                ismooth=self._options.is_smoothing
                self._conf["is_smoothing"]=bool2string(ismooth)
            
            if opt.dest=="smooth_tune":
                smt=self._options.smooth_tune
                self._conf["smooth_tune"]=str(smt)
                
            if opt.dest=="correlation_length":
                cl=self._options.correlation_length
                self._conf["correlation_length"]=str(cl)
                
            if opt.dest=="time_step":
                ts=self._options.time_step
                self._conf["time_step"]=str(ts)
                
            if opt.dest=="sample_length":
                sl=self._options.sample_length
                self._conf["sample_length"]=str(sl)
                
            if opt.dest=="is_average_output":
                isao=self._options.is_average_output
                self._conf["is_average_output"]=bool2string(isao)
            
            if opt.dest=="is_write_ac":
                iswa=self._options.is_write_ac
                self._conf["is_write_ac"]=bool2string(iswa)
            
            if opt.dest=="is_convert_velocity":
                iscv=self._options.is_convert_velocity
                self._conf["is_convert_velocity"]=bool2string(iscv)

            if opt.dest == "direction_tensor":
                self._conf["direction_tensor"]=self._options.direction_tensor

            if opt.dest=="is_plot":
                isp=self._options.is_plot
                self._conf["is_plot"]=bool2string(isp)
                
            if opt.dest=="is_save":
                iss=self._options.is_save
                self._conf["is_save"]=bool2string(iss)
            
    
    def read_conf_file(self, filename):
        if not os.path.exists(filename):
            warning("file %s does not exist, but it will continue anyway!" %filename)
        else:
            c = open(filename, 'r')
            confs = self._conf
            is_continue = False
            for line in c:
                new_line=line.split("#")[0]
                if new_line.strip() == '':
                    is_continue = False
                    continue   
                
                if new_line.find('=') != -1:
                    left, right = [x.strip().lower() for x in new_line.split('=')]
                    if left.find("filename")!=-1 or left=="atom_type":
                        right=new_line.split("=")[1].strip()
                    confs[left] = right
                    
                if is_continue:
                    confs[left] += new_line.strip()
                    confs[left] = confs[left].replace('\\', ' ')
                    is_continue = False 
                    
                if new_line.find('\\') != -1:
                    is_continue = True
                    
    def set_settings(self):
        if self._conf.has_key("input_filename"):
            self.input_filename=self._conf["input_filename"]
        
        if self._conf.has_key("file_format"):
            self.file_format=self._conf["file_format"][0]
        
        if self._conf.has_key("atom_type"):
            #self.atom_type=self._conf["atom_type"]
            try:
                typ=self._conf["atom_type"]
                typ=typ.replace(" ","")
                if typ!="":
                    if typ.find("*")!=-1:
                        assert typ.count("*")==typ.count(",")+1
                        specie=typ.split(",")  
                        atom_type=[[s.split("*")[0]]*int(s.split("*")[1]) for s in specie]
                        atom_type=sum(atom_type,[])# flattening
                    else:
                        if typ.find(",")!=-1:
                            atom_type=typ.split(",")
                        else:
                            atom_type=[typ]
                    assert "" not in atom_type
                    self.atom_type=np.array(atom_type)
            except:
                error("Wrong format for atom_type!")
        
        if self._conf.has_key("out_filename"):
            self.out_filename=self._conf["out_filename"]
            
        if self._conf.has_key("step_range"):
            ran=self._conf["step_range"]
            if ran=="":
                self.step_range=slice(None, None, None)
            elif ran.count(":")>0:
                exec "s=np.s_[%s]"%ran
                self.step_range=s
            else:
                error("Wrong format for step_range")
            
        if self._conf.has_key("is_average_input"):
            self.is_average_input=string2bool(self._conf["is_average_input"])
            
        if self._conf.has_key("is_normalize"):
            self.is_normalize=string2bool(self._conf["is_normalize"])
        
        if self._conf.has_key("is_summation"):
            self.is_summation=string2bool(self._conf["is_summation"])
            
        if self._conf.has_key("is_smoothing"):
            self.is_smoothing=string2bool(self._conf["is_smoothing"])

        if self._conf.has_key("is_average_output"):
            self.is_average_output=string2bool(self._conf["is_average_output"])
        
        if self._conf.has_key("is_write_ac"):
            self.is_write_ac=string2bool(self._conf["is_write_ac"])
            
        if self._conf.has_key("is_convert_velocity"):
            self.is_convert_velocity=string2bool(self._conf["is_convert_velocity"])
        
        if self._conf.has_key("is_plot"):
            self.is_plot=string2bool(self._conf["is_plot"])
        
        if self._conf.has_key("is_save"):
            self.is_save=string2bool(self._conf["is_save"])
        
        if self._conf.has_key("smooth_tune"):
            self.smooth_tune=float(self._conf["smooth_tune"])
            if self.smooth_tune==0.0:
                error("tune factor for Gaussian smoothing cannot be 0!")
            
        if self._conf.has_key("correlation_length"):
            self.correlation_length=int(self._conf["correlation_length"])
            
        if self._conf.has_key("time_step"):
            self.time_step=float(self._conf["time_step"])
            
        if self._conf.has_key("sample_length"):
            self.sample_length=int(self._conf["sample_length"])

        if self._conf.has_key("direction_tensor"):
            d=self._conf['direction_tensor']
            tran=maketrans("xyz", "012")
            if d is not None:
                try:
                    self.direction_tensor=tuple(map(int, d.translate(tran)))
                except ValueError:
                    error("The direction can only be set as characters 'x', 'y' and 'z'")
                if len(self.direction_tensor) == 1:
                    self.direction_tensor = (self.direction_tensor[0],self.direction_tensor[0])
                elif len(self.direction_tensor) != 2:
                    error("The direction is a second-order tensor, please recheck the settings!")
            else:
                self.direction_tensor = d
        
    def parameter_coupling(self):
        if self.input_filename is not None and not self._conf.has_key("file_format"):
            if self.input_filename.split(".")[-1]=="xyz":
                if self.file_format != "x":
                    warning("xyz file format detected, format converted forcibly!")
                    self.file_format="x"  
            elif self.input_filename.split(".")[-1]=="hdf5":
                if self.file_format != "h":
                    warning("hdf5 file format detected, format converted forcibly!")
                    self.file_format="h"
            elif self.input_filename == "XDATCAR":
                if self.file_format != "v":
                    warning("vasp file format detected, format converted forcibly!")
                    self.file_format="v"
        if self.file_format is not None:
            if self.file_format=="v":
                if self.input_filename is None:
                    print "XDATCAR as input file name used by default for the file format of vasp"
                    self.input_filename="XDATCAR"
            if self.file_format=="h":
                if self.input_filename is None:
                    print "md_velocities.hdf5 as input file name used by default for the file format of hdf5"
                    self.input_filename="md_velocities.hdf5"
                elif self.input_filename.split(".")[-1] != "hdf5":
                    warning("'filename.hdf5' should be used for hdf5 file type, 'md_velocities.hdf5' is used!")
                    self.input_filename="md_velocities.hdf5"
            
            
        
        
            

        
        