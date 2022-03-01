#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sys
import os

module_path = os.path.abspath('helper/')
if module_path not in sys.path:
    sys.path.append(module_path+"")  
    
module_path = os.path.abspath('models/mf/')
if module_path not in sys.path:
    sys.path.append(module_path+"")

module_path = os.path.abspath('models/mb/')
if module_path not in sys.path:
    sys.path.append(module_path+"")
    

module_path = os.path.abspath('models/habit/')
if module_path not in sys.path:
    sys.path.append(module_path+"")
    
    module_path = os.path.abspath('models/wsls/')
if module_path not in sys.path:
    sys.path.append(module_path+"")
    
module_path = os.path.abspath('models/kdh/')
if module_path not in sys.path:
    sys.path.append(module_path+"")


