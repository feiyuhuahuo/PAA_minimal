#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# import numpy as np
# import pdb
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(8, 8))
#
# xx = np.array([0, 2, 3, 4, 5, 6, 7, 8, 8.5, 9.5])
# yy = np.array([10, 9, 8, 6, 5, 3, 2, 1, 0.7, 0.2])
#
# plt.title('Car', size=20, color='green')
# plt.xlim(0, 10)
# plt.xlabel('Recall', size=14)
# plt.ylim(0, 10)
# plt.ylabel('Precision', size=14)
# plt.tick_params(labelsize=14)  # set tick font size
#
# plt.hlines(3, xmin=0, xmax=6, color='blue', linestyles='dashed')
# plt.vlines(6, ymin=0, ymax=3, color='blue', linestyles='dashed')
#
# # hatch: ('/', '//', '-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
# shadow = plt.bar(x=0, height=1, width=20, hatch='//', color='white', edgecolor='grey')
# # loc: ('upper right', 'lower left', 'center', 'lower center', (0.4, 0.5) ...)
# plt.legend(handles=[shadow], labels=['Area where score < 0.05'], loc='upper right', fontsize=14)
#
# plt.scatter(6, 3, color='red')
# plt.text(6, 3, 0.23, ha='left', va='bottom', fontsize=14, color='red')
# plt.text(0 + 0.2, 3 - 0.2, 'AP=0.224', ha='left', va='top', fontsize=14, color='blue')
# plt.plot(xx, yy, color='black')
# plt.show()

a=1
b=0
c=0
d=1
if a and (b and c or d):
    print('ccc')