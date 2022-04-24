# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 09:35:36 2017

@author: user
"""

import sys 
try: 
    f = open('myfilename.txt', 'r')
except FileNotFoundError:
    print("The file couldn't be found. " + "This program stops here.")
    sys.exit(1) # a way to exit the program 8 9 for line in f: 10 print(line, end='') 11 f.close()
