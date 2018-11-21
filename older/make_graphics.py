import numpy as np
from scipy import stats

import plotly.offline as py
import plotly.graph_objs as go

# Create surfaces Z1 and Z2
n = 100

r = 10
x = np.linspace(-1.8, 1.8, n)
y = np.linspace(-1.8, 1.8, n)

X, Y = np.meshgrid(x, y)

XY = np.empty((n * n, 2))
XY[:, 0] = X.flatten()
XY[:, 1] = Y.flatten()

# Z1
cov = np.eye(2) * 0.2
dist = stats.multivariate_normal(np.array([-0.6, -0.6]), cov)
Z1 = dist.pdf(XY).reshape((n, n))

cov = np.eye(2) * 0.2
dist = stats.multivariate_normal(np.array([0.6, 0.6]), cov)
Z1 += dist.pdf(XY).reshape((n, n))

cov = np.array([[0.1, 0.002],
               [0.005, 0.1]])
dist = stats.multivariate_normal(np.array([0.5, -0.5]), cov)
Z1 += dist.pdf(XY).reshape((n, n)) * 1

cov = np.array([[0.05, 0.002],
               [0.003, 0.05]])
dist = stats.multivariate_normal(np.array([-0.5, -1.0]), cov)
Z1 += dist.pdf(XY).reshape((n, n)) * 0.15

# Z2
cov = np.eye(2) * 0.2
dist = stats.multivariate_normal(np.array([0.6, -0.6]), cov)
Z2 = dist.pdf(XY).reshape((n, n))

cov = np.eye(2) * 0.2
dist = stats.multivariate_normal(np.array([-0.6, 0.6]), cov)
Z2 += dist.pdf(XY).reshape((n, n))

cov = np.array([[0.1, 0.005],
               [0.002, 0.1]])
dist = stats.multivariate_normal(np.array([-0.5, 0.5]), cov)
Z2 += dist.pdf(XY).reshape((n, n)) * 0.6

cov = np.array([[0.05, 0.004],
               [0.004, 0.05]])
dist = stats.multivariate_normal(np.array([0.5, 1.0]), cov)
Z2 += dist.pdf(XY).reshape((n, n)) * 0.3

Z2 += 2.5

# Set plotting settings
color_scale1 = 'Greens'
color_scale2 = 'Greys'

layout = go.Layout(
	scene = dict(
    	camera = dict(
        	up=dict(x=0, y=0, z=1),
        	center=dict(x=0, y=0, z=0),
        	eye=dict(x=0, y=-1.8, z=0.1)
    	),
    	xaxis = dict(
        	title='Sequence',
        	showgrid=False,
        	zeroline=True,
        	mirror=True,
        	showline=True,
        	ticks='',
        	showticklabels=False
    	),
    	yaxis = dict(
        	title='Sequence',
        	showgrid=False,
        	zeroline=True,
        	mirror=True,
        	showline=True,
        	ticks='',
        	showticklabels=False
    	),
    	zaxis = dict(
        	title='Fitness',
        	showgrid=False,
        	zeroline=True,
        	showline=True,
        	ticks='',
        	showticklabels=False
    	)
	)
)

data = [
    go.Surface(z=Z1, opacity=1, colorscale=color_scale1, showscale=False, reversescale=True)#,
    #go.Surface(z=Z2, opacity=1, colorscale=color_scale2, showscale=False, reversescale=True)
]

fig = go.Figure(data=data, layout=layout)

py.plot(fig, filename='temp_landscape.html')