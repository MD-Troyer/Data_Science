#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 11:02:38 2016

@author: troyer
"""

#def unpack_arg(arg, msg=None, starting_level=0):
#    """Unpack that data like a baller!"""
#    if msg:
#        print "\n"+str(msg)
#        del(msg)
#    level = starting_level
#    if type(arg) == dict:
#        print ("\n"+(level*("\t"))+str(arg))
#        print (level*("\t"))+str(type(arg))
#        for key, value in arg.items():
#            print ((level*("\t\t"))+str(key)+": "+str(value))
#    else:
#        print ("\n"+(level*("\t"))+str(arg))
#        print (level*("\t"))+str(type(arg))
#        if hasattr(arg, '__iter__'): #does not include strings
#            print (level*("\t")+"Iterables:")
#            for a in arg:
#                unpack_arg(a, starting_level=level+1)
#        else:
#            print (level*("\t")+"Iterables: None")
#    return

#def write_arg(arg, path, msg=None, starting_level=0):
#    """Accepts a [path] txt from open(path)
#       and unpacks that data like a baller!"""
#    level = starting_level
#    txtfile = open(path, 'a')
#    if level == 0:
#        txtfile.write("\n"+str(msg)+"\n")
#    if type(arg) == dict:
#        txtfile.write("\n"+(level*"\t")+str(arg)+"\n")
#        txtfile.write((level*"\t")+str(type(arg))+"\n")
#        for key, value in arg.items():
#            txtfile.write((level*"\t\t")+str(key)+": "+str(value)+"\n")
#            txtfile.close()
#    else:
#        txtfile.write("\n"+(level*"\t")+str(arg)+"\n")
#        txtfile.write((level*"\t")+str(type(arg))+"\n")
#        if hasattr(arg, '__iter__'): #does not include strings
#            txtfile.write((level*"\t")+"Iterables:"+"\n")
#            txtfile.close()
#            for a in arg:
#                write_arg(a, path, starting_level=level+1)
#        else:
#            txtfile.write((level*"\t")+"Iterables: None\n")
#            txtfile.close()
#            return

#def print_arg(arg, msg=None, starting_level=0):
#    """Accepts a [path]txt from open(path) and
#       unpacks that data like a baller!"""
#    level = starting_level
#    if msg:
#        print "\n"+str(msg)+"\n"
#        del(msg)
#    if type(arg) == dict:
#        print "\n"+(level*("\t"))+str(arg)+"\n"
#        print (level*("\t"))+str(type(arg))+"\n"
#        for key, value in arg.items():
#            print (level*("\t\t"))+str(key)+": "+str(value)+"\n"
#    else:
#        print "\n"+(level*("\t"))+str(arg)+"\n"
#        print (level*("\t"))+str(type(arg))+"\n"
#        if hasattr(arg, '__iter__'): #does not include strings
#            print (level*("\t"))+"Iterables:"+"\n"
#            for a in arg:
#                print_arg(a, starting_level=level+1)
#        else:
#            print (level*("\t"))+"Iterables: None\n"
#            return
#    return

def _write_arg(arg, path, msg=None, starting_level=0):
    """Accepts an arg and [path] txt [will write to existing or create new] and
       recursively unpacks arg and writes that data like a baller!
       Uses explicit file open/close to ensure data write integity"""

    level = starting_level
    if level == 0:
        txtfile = open(path, 'a')
        txtfile.write("\n"+"_"*80)
        txtfile.write("\n"+str(msg)+"\n")
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