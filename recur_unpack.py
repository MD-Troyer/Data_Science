#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 11:02:38 2016

@author: troyer
"""
# Function

def write_arg(arg, path, msg=None, starting_level=0):
    """Accepts an arg and [path] txt [will write to existing or create new] and
       recursively unpacks arg and writes that data like a baller!
       Uses explicit file open/close to ensure data write integity"""

    level = starting_level
    if level == 0 and msg:
        txtfile = open(path, 'a')
        txtfile.write("\n"+"_"*80)
        txtfile.write("\n"+str(msg)+"\n")
        del(msg)
        txtfile.close()
    if type(arg) == dict:
        txtfile = open(path, 'a')
        txtfile.write("\n"+(level*"\t")+str(arg)+"\n")
        txtfile.write((level*"\t")+str(type(arg))+"\n")
        txtfile.write((level*"\t")+"Keys : Values"+"\n")
        txtfile.close()
        for key, value in arg.items():
            txtfile = open(path, 'a')
            txtfile.write("\n\t"+(level*"\t")+str(key)+": "+str(value)+"\n")
            txtfile.write("\t"+(level*"\t")+str(type(key))+": "+str(type(value))+"\n")
            txtfile.close()
            if hasattr(value, '__iter__'):
                txtfile = open(path, 'a')
                txtfile.write("\t"+(level*"\t")+"Values of "+str(key)+":"+"\n")
                txtfile.close()
                for val in value:
                    _write_arg(val, path, starting_level=level+2)
                txtfile = open(path, 'a')
                txtfile.write("\n")
                txtfile.close()
    else:
        txtfile = open(path, 'a')
        txtfile.write("\n"+(level*"\t")+str(arg)+"\n")
        txtfile.write((level*"\t")+str(type(arg))+"\n")
        txtfile.close()
        if hasattr(arg, '__iter__'): #does not include strings
            txtfile = open(path, 'a')
            txtfile.write((level*"\t")+"Iterables:"+"\n")
            txtfile.close()
            for a in arg:
                _write_arg(a, path, starting_level=level+1)
        else:
            txtfile = open(path, 'a')
            txtfile.write((level*"\t")+"Iterables: None\n")
            txtfile.close()
            return
        
# Method

def _write_arg(self, arg, path, msg=None, starting_level=0):
    """Accepts an arg and [path] txt [will write to existing or create new] and
       recursively unpacks arg and writes that data like a baller!
       Uses explicit file open/close to ensure data write integity"""

    level = starting_level
    if level == 0 and msg:
        txtfile = open(path, 'a')
        txtfile.write("\n"+"_"*80)
        txtfile.write("\n"+str(msg)+"\n")
        del(msg)
        txtfile.close()
    if type(arg) == dict:
        txtfile = open(path, 'a')
        txtfile.write("\n"+(level*"\t")+str(arg)+"\n")
        txtfile.write((level*"\t")+str(type(arg))+"\n")
        txtfile.write((level*"\t")+"Keys : Values"+"\n")
        txtfile.close()
        for key, value in arg.items():
            txtfile = open(path, 'a')
            txtfile.write("\n\t"+(level*"\t")+str(key)+": "+str(value)+"\n")
            txtfile.write("\t"+(level*"\t")+str(type(key))+": "+str(type(value))+"\n")
            txtfile.close()
            # if type(value) == dict
            if hasattr(value, '__iter__'):
                txtfile = open(path, 'a')
                txtfile.write("\t"+(level*"\t")+"Values of "+str(key)+":"+"\n")
                txtfile.close()
                for val in value:
                    self._write_arg(val, path, starting_level=level+2)
                txtfile = open(path, 'a')
                txtfile.write("\n")
                txtfile.close()
    else:
        txtfile = open(path, 'a')
        txtfile.write("\n"+(level*"\t")+str(arg)+"\n")
        txtfile.write((level*"\t")+str(type(arg))+"\n")
        txtfile.close()
        if hasattr(arg, '__iter__'): #does not include strings
            txtfile = open(path, 'a')
            txtfile.write((level*"\t")+"Iterables:"+"\n")
            txtfile.close()
            for a in arg:
                self._write_arg(a, path, starting_level=level+1)
        else:
            txtfile = open(path, 'a')
            txtfile.write((level*"\t")+"Iterables: None\n")
            txtfile.close()
            return
