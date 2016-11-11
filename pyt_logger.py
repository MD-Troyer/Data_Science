#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 11:02:38 2016

@author: troyer
"""
import os

class pyt_log(object):
    def __init__(self, report_path, log_path, log_active=True):
        self.report_path = report_path
        self.log_path = log_path
        self.log_active = log_active

    def _unpack_arg(self, arg, path):
        """accepts a txt from open(path)
        """
        with open(path, 'a') as txtfile:
            txtfile.write(str(type(arg))+"\n")
            txtfile.write(str(arg)+"\n\n")
            if hasattr(arg, '__iter__'): #does not include strings
                txtfile.write("Iterable = True\n")
                for a in arg:
                    txtfile.write(str(type(a))+"\n")
                    txtfile.write(str(a)+"\n\n")
            else:
                txtfile.write("Iterable = False\n")
                txtfile.write(str(type(arg))+"\n")
                txtfile.write(str(arg)+"\n\n")
        return

    def _writer(self, msg, path, *args):
        if os.path.exists(path):
            write_type = 'a'
        else:
            write_type = 'w'
        with open(path, write_type) as txtfile:
            txtfile.write("\n"+msg+"\n")
            txtfile.close()
            if args:
                for arg in args:
                    self._unpack_arg(arg, path)

    def console(self, msg):
        print msg

    def report(self, msg):
        self._writer(msg, path=self.report_path)

    def logfile(self, msg, *args):
        if self.log_active:
            path = self.log_path
            self._writer(msg, path, *args)

    def log_all(self, msg, *args):
        self.console(msg)
        self.report(msg)
        self.logfile(msg, *args)




