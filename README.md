# ML lib

[//]: # (Markdown CheatSheet:https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#links)

This **ML-lib** is based on [pytorch](https://pytorch.org/), [pytorch lightning](https://github.com/PyTorchLightning/pytorch-lightning) and [neptune](https://neptune.ai/)

Clone it to find a collection of:
 - Evaluation Functions
 - Utility Function for Model I/O
 - DL Modules
 - A Plotter Object
 - Audio related Tools and Funtion
   - Librosa
   - Scipy Signal
 - PointCloud related Tools and Functions
 
 ###Notes:
 - Use directory links to link from your main project folder to the ml_lib folder. Pycharm will automatically use 
 included packages.
    \
    \
    For Windows Users:
    ``` bash
     mklink /d "ml_lib" "..\ml_lib""
    ```
    For Unix User:
    ``` bash
    ln -s ../ml_lib ml_lib
    ```
