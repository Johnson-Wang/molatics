# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 08:46:44 2013

@author: xwangan
"""
from realmd.information import warning
from realmd.mddos.settings import Settings

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

class KappaSettings(Settings):
    def __init__(self, options,option_list, args,config_file=None):
        Settings.__init__(self,options, option_list, args, config_file)
        if len(args) >= 3:
            self.md_input_filename = args[2]
            self.fe_input_filename=args[1]
        elif len(args) >= 2:
            warning("The general information file is set default as md.out")
            self.md_input_filename = "md.out"
            self.fe_input_filename = args[1]
        else:
            warning("The force_and_energy file is set default as heat_flux.out")
            warning("The general information file is set default as md.out")
            self.fe_input_filename= None
            self.md_input_filename = None
        self.temperature = 300
        self.is_convert_input = False
        self.volume = None
        self.is_difference = False
        self.hf_filename = None

        if options is not None:
            self._read_options()
        self._set_settings()

    def _read_options(self):
        for opt in self._option_list:
            if opt.dest=="hf_filename":
                fhf=self._options.hf_filename
                self._conf["hf_filename"]=fhf

            if opt.dest=="is_convert_input":
                convert = bool2string(self._options.is_convert_input)
                self._conf['is_convert_input'] = convert

            if opt.dest=="temperature":
                self._conf['temperature']=self._options.temperature

            if opt.dest=="volume":
                if self._options.volume is not None:
                    self._conf['volume'] = self._options.volume

            if opt.dest == "is_difference":
                self._conf['is_difference'] = self._options.is_difference

    def _set_settings(self):
        if self._conf.has_key("hf_filename"):
            self.hf_filename=self._conf["hf_filename"]

        if self._conf.has_key("is_convert_input"):
            self.is_convert_input = string2bool(self._conf["is_convert_input"])

        if self._conf.has_key("temperature"):
            self.temperature=self._conf["temperature"]

        if self._conf.has_key("volume"):
            self.volume = self._conf['volume']

        if self._conf.has_key("is_difference"):
            self.is_difference = self._conf['is_difference']



            
            
        
        
            

        
        