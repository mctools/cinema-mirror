from Cinema.Prompt import PromptFileReader
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import argparse
from matplotlib.patches import Rectangle

from Cinema.Interface import plotStyle

plotStyle()

def plot_zoom(mcpl, text):

	f = PromptFileReader(mcpl)
	data=f.getData('content')
	x=f.getData('xedge')[90:140]; y=f.getData('yedge')[60:100]; X, Y = np.meshgrid(x, y)

	fig=plt.figure()
	ax = fig.add_subplot(111)
	plt.xlabel('x, mm')
	plt.ylabel('y, mm')
	plt.text(0.75, 0.9, text, c='r', fontsize=14, weight='bold',transform=ax.transAxes)
	pcm = ax.pcolormesh(X, Y, data[90:140,:][:,60:100].T, vmin=0, vmax=22, cmap=plt.cm.jet,shading='auto')
	fig.colorbar(pcm, ax=ax)

def plot_overview(mcpl, text, text_zoom):

	f = PromptFileReader(mcpl)
	data=f.getData('content')
	x=f.getData('xedge'); y=f.getData('yedge'); X, Y = np.meshgrid(x, y)

	fig=plt.figure()
	ax = fig.add_subplot(111)
	plt.xlabel('x, mm')
	plt.ylabel('y, mm')
	plt.text(0.75, 0.9, text, c='black', fontsize=14, weight='bold',transform=ax.transAxes)
	plt.text(0.5, 0.35, text_zoom, c='red', fontsize=12, transform=ax.transAxes)
	pcm = ax.pcolormesh(X, Y, data.T, cmap=plt.cm.jet,shading='auto')
	fig.colorbar(pcm, ax=ax)
	zoom_area([x[90],y[60]], [x[140],y[100]])

def zoom_area(leftbottom, righttop):

	width = righttop[0] - leftbottom[0]
	height = righttop[1] - leftbottom[1]
	area = Rectangle(tuple(leftbottom), width, height,
		  fill=False,
		  color='r',
		  linewidth=1,
		  linestyle='--')
	plt.gca().add_patch(area)


plot_overview('bias0p1.mcpl.gz', 'bias=0.1', '(b)')
plot_overview('bias1.mcpl.gz', 'bias=1', '(d)')
plot_zoom('bias0p1.mcpl.gz', 'bias=0.1')
plot_zoom('bias1.mcpl.gz', 'bias=1')


plt.show()
