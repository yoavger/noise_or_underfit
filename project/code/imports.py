#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sys
import os

module_path = os.path.abspath('helper/')
if module_path not in sys.path:
    sys.path.append(module_path+"")   

module_path = os.path.abspath('models/Hybrid/')
if module_path not in sys.path:
    sys.path.append(module_path+"")
    
module_path = os.path.abspath('models/MB/')
if module_path not in sys.path:
    sys.path.append(module_path+"")
    
module_path = os.path.abspath('models/MF/')
if module_path not in sys.path:
    sys.path.append(module_path+"")
    
module_path = os.path.abspath('models/nWS/')
if module_path not in sys.path:
    sys.path.append(module_path+"")
    
module_path = os.path.abspath('models/pWSLS/')
if module_path not in sys.path:
    sys.path.append(module_path+"")
    
module_path = os.path.abspath('models/kDH/')
if module_path not in sys.path:
    sys.path.append(module_path+"")


