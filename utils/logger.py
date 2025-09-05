import os
import time
import json
import torch
import shutil
import datetime

def init(log_path='', split_id = 'None'):
    
    global log

    log = logger(log_path = log_path, split_id = split_id)

def print_log(s, is_print_file = True):

    log.print_log(s, is_print_file)

class logger(object):

    def __init__(self, log_path = '', print_screen = True, print_file = True, split_id = 0):

        self.split_id = split_id
        self.print_screen = print_screen
        self.print_file = print_file
        self.log_path = log_path
        self.st_time = datetime.datetime.now().strftime("%I%M%p_on_%B_%d_%Y")
        self.file_name = "save_models_" + str(split_id) + "_logs_at_" + self.st_time + ".log"
        self.file_path = os.path.join(self.log_path, self.file_name)

        if self.print_file:
            self.initFileSaver()

    def initFileSaver(self):
        os.makedirs(self.log_path, exist_ok = True)
        self.f = open(self.file_path,'w') 

    def print_log(self, s, is_print_file = True):
        
        if not isinstance(s, str):

            s = str(s)

        if self.print_screen:

            print(s)

        if self.print_file and is_print_file:

            self.f.write(s)
            self.f.write('\n')

    def close(self):

        self.print_file = False
        self.print_screen = False
        self.f.close()