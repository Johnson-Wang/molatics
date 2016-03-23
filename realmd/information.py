# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 22:06:56 2013

@author: xwangan
"""
from datetime import datetime
import sys
import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te=time.time()
        print '%r %r: %.2f sec' %\
              (method.__name__, args, te-ts)
        return result
    return timed
# define standard printing functions
def print_mddos(time=None):
    print """
MM    MM DDDDD   DDDDD    OOOOO   SSSSS  
MMM  MMM DD  DD  DD  DD  OO   OO SS      
MM MM MM DD   DD DD   DD OO   OO  SSSSS  
MM    MM DD   DD DD   DD OO   OO      SS 
MM    MM DDDDDD  DDDDDD   OOOO0   SSSSS
"""
    if time:
        print "Program begins at", str(time)

def print_end(time=None):
    if time:
        print "Program ends at:", str(time)    
    print """
EEEEEEE NN   NN DDDDD   
EE      NNN  NN DD  DD  
EEEEE   NN N NN DD   DD 
EE      NN  NNN DD   DD 
EEEEEEE NN   NN DDDDDD
"""
        
def program_exit(mode=0):
    if mode==0:
        print "Program completed successfully!"
    else:
        print "Program terminated abnormally!"
    print_end(time=datetime.now())
    sys.exit(mode)

def error(message):
    print "Error:"+message
    program_exit(mode=1)

def warning(message):
    print "Warning:"+message

def print_error():
    print """  ___ _ __ _ __ ___  _ __
 / _ \ '__| '__/ _ \| '__|
|  __/ |  | | | (_) | |
 \___|_|  |_|  \___/|_|
"""

def print_attention(attention_text):
    print "*******************************************************************"
    print attention_text
    print "*******************************************************************"
    print ""

def print_error_message(message):
    print
    print message
    print_error()