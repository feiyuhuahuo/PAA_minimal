#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# import numpy as np
# import pdb
# import matplotlib.pyplot as plt
#
# fig = plt.figure(figsize=(15, 10))
#
# for i in range(10):
#     plt.subplot(3, 4, i + 1)
#     plt.margins(x=0, y=0)
#     xx = np.array([0.1, 0.2, 0.3, 0.4, 0.55, 0.6, 0.7, 0.8, 0.85, 0.95])
#     yy = np.array([1, 0.9, 0.8, 0.6, 0.5, 0.3, 0.2, 0.1, 0.07, 0.02])
#
#     plt.title('thre=0.5', size=12, color='black')
#     plt.xlim(0, 1)
#     plt.xlabel('Recall', size=12)
#     plt.ylim(0, 1)
#
#     plt.ylabel('Precision', size=12)
#     plt.xticks([0., 0.2, 0.6], [0, 2, 6], rotation=10)
#     plt.tick_params(labelsize=12)  # set tick font size
#
#     plt.hlines(3, xmin=0, xmax=6, color='blue', linestyles='dashed')
#     plt.vlines(6, ymin=0, ymax=3, color='blue', linestyles='dashed')
#
#     # hatch: ('/', '//', '-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
#     shadow = plt.bar(x=0, height=1, width=20, hatch='//', color='white', edgecolor='grey')
#     # loc: ('upper right', 'lower left', 'center', 'lower center', (0.4, 0.5) ...)
#
#     plt.scatter(6, 3, color='red')
#     plt.text(6, 3, 0.23, ha='left', va='bottom', fontsize=12, color='red')
#     plt.text(0 + 0.2, 3 + 0.3, 'AP=0.224', ha='left', va='bottom', fontsize=12, color='black')
#     plt.text(0 + 0.2, 3 - 0.3, 'AP=0.224, FF=33%', ha='left', va='top', fontsize=12, color='blue')
#
#     # plt.xticks([5], [0.5], rotation=10)
#
#     plt.plot(xx, yy, color='black')
#
# fig.suptitle('Car', size=16, color='red')
# fig.legend(handles=[shadow], labels=['Area where score < 0.05'], loc='upper right', fontsize=12)
# plt.tight_layout()  # resolve the overlapping issue when using subplot()
# plt.savefig(f'results/mpp_result/cckkc.jpg')
# plt.show()

print(f'{round(0.23264,3)}')