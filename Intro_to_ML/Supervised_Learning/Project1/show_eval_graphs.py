#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches

graph_settings = open("model_eval_graph_data.txt","r")
settings = graph_settings.readlines()

labels = settings[0].strip().split(",")
labels.pop()
settings.pop(0)
y_pos = np.arange(len(labels))
x = np.arange(10)
ys = [i+x+(i*x)**2 for i in range(10)]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))

def make_graph(list,figure,title):
    plt.subplot(2,3,figure)
    plt.bar(y_pos,list,align="center",alpha=0.5,color=colors)
    # plt.xticks(y_pos,labels)
    plt.ylabel('Value')
    plt.title(title)

fig = 1
title_map = {
    1:'Accuracy - Train',
    2:'Accuracy - Test',
    3:'F-Score - Train',
    4:'F-Score - Test',
    5:'Time - Train',
    6:'Time - Test',
}
for s in settings:
    s = s.strip().split(",")
    s.pop()
    list = []
    for item in s:
        list.append(float(item))
    make_graph(list,fig,title_map[fig])
    fig += 1

# Create patches for the legend
patches = []
for i, learner in enumerate(labels):
    patches.append(mpatches.Patch(color = colors[i], label = learner))
plt.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \
           loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 'small')
plt.show()
