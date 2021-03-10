# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 23:17:06 2021

@author: KeremÖzkılıç
"""
from PyQt5 import uic
from pathlib import Path

if __name__ == '__main__':
    cwd = Path.cwd()
    ui_src = Path.joinpath(cwd, 'inventory', 'user_interface.ui')
    ui_out = Path.joinpath(cwd, 'inventory', 'ui.py')
    with open(ui_out,'w',encoding ='utf-8') as out:
        uic.compileUi(ui_src, out)
