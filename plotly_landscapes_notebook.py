# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.7.0
# ---

# +
import numpy as np
from scipy import stats

import plotly.offline as py
import plotly.graph_objs as go
# -

# ## Plotting settings

# +
# Colors
color_scale1 = 'Greens'
color_scale2 = 'Greys'

# Plot layout
layout = go.Layout(
	scene = dict(
    	camera = dict(
        	up=dict(x=0, y=0, z=1),
        	center=dict(x=0, y=0, z=0),
        	eye=dict(x=0, y=-1.8, z=0.1)
    	),
    	xaxis = dict(
        	title='',
        	showgrid=False,
        	zeroline=False,
        	mirror=False,
        	showline=False,
        	ticks='',
        	showticklabels=False
    	),
    	yaxis = dict(
        	title='',
        	showgrid=False,
        	zeroline=False,
        	mirror=False,
        	showline=False,
        	ticks='',
        	showticklabels=False
    	),
    	zaxis = dict(
        	title='',
        	showgrid=False,
        	zeroline=False,
        	showline=False,
        	ticks='',
        	showticklabels=False
    	)
	)
)
# -

# ## Two landscapes

# +
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

#Z2 += 2.5
# -

# ## Complex peak

# +
n = 100

r = 1
x = np.linspace(-1.4, 1.4, n)
y = np.linspace(-1.4, 1.4, n)
X, Y = np.meshgrid(x, y)


XY = np.empty((n * n, 2))
XY[:, 0] = X.flatten()
XY[:, 1] = Y.flatten()

cov = np.eye(2) * 0.2
dist = stats.multivariate_normal(np.zeros(2), cov)
Z_complex = dist.pdf(XY).reshape((n, n))

cov = np.array([[0.01, 0.002],
               [0.005, 0.01]])
dist = stats.multivariate_normal(np.array([-0.5, -0.2]), cov)
Z_complex += dist.pdf(XY).reshape((n, n)) * 0.01

cov = np.array([[0.01, 0.002],
               [0.003, 0.01]])
dist = stats.multivariate_normal(np.array([0.7, 0.2]), cov)
Z_complex += dist.pdf(XY).reshape((n, n)) * 0.015

cov = np.array([[0.01, 0.002],
               [0.003, 0.01]])
dist = stats.multivariate_normal(np.array([0.35, 0.1]), cov)
Z_complex += dist.pdf(XY).reshape((n, n)) * 0.005

cov = np.array([[0.01, 0.002],
               [0.003, 0.02]])
dist = stats.multivariate_normal(np.array([0.9, -0.8]), cov)
Z_complex += dist.pdf(XY).reshape((n, n)) * 0.02

cov = np.array([[0.02, 0.002],
               [0.003, 0.01]])
dist = stats.multivariate_normal(np.array([0.6, -0.3]), cov)
Z_complex += dist.pdf(XY).reshape((n, n)) * 0.03

cov = np.array([[0.01, 0.002],
               [0.003, 0.01]])
dist = stats.multivariate_normal(np.array([0., -0.8]), cov)
Z_complex += dist.pdf(XY).reshape((n, n)) * 0.035
# -

# ## Do the plots

# +
# data = [
#     go.Surface(z=Z1, opacity=1, colorscale=color_scale1, showscale=False, reversescale=True),
#     go.Surface(z=Z2, opacity=1, colorscale=color_scale2, showscale=False, reversescale=True)
# ]

# fig = go.Figure(data=data, layout=layout)

# py.plot(fig, filename='temp_landscape.html')

# +
data = [
    go.Surface(z=Z1, opacity=1, colorscale=color_scale1, showscale=False, reversescale=True)
]

fig = go.Figure(data=data, layout=layout)

py.plot(fig, filename='three_peak_landscape.html')

# +
data = [
    go.Surface(z=Z_complex, opacity=1, colorscale=color_scale1, showscale=False, reversescale=True)
]

fig = go.Figure(data=data, layout=layout)

py.plot(fig, filename='complex_peak_landscape.html')
# -


