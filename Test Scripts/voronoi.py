from matplotlib import projections, pyplot as plt, cm
from PIL import Image, ImageShow
import colorcet as cc
import numpy as np
from sklearn.datasets import make_blobs

def lonely(p,X,r):
    m = X.shape[1]
    x0,y0 = p
    x = y = np.arange(-r,r)
    x = x + x0
    y = y + y0

    u,v = np.meshgrid(x,y)

    u[u < 0] = 0
    u[u >= m] = m-1
    v[v < 0] = 0
    v[v >= m] = m-1

    return not np.any(X[u[:],v[:]] > 0)

def generate_samples(m=2500,r=200,k=30):
    # m = extent of sample domain
    # r = minimum distance between points
    # k = samples before rejection
    active_list = []

    # step 0 - initialize n-d background grid
    X = np.ones((m,m))*-1

    # step 1 - select initial sample
    x0,y0 = np.random.randint(0,m), np.random.randint(0,m)
    active_list.append((x0,y0))
    X[active_list[0]] = 1

    # step 2 - iterate over active list
    while active_list:
        i = np.random.randint(0,len(active_list))
        rad = np.random.rand(k)*r+r
        theta = np.random.rand(k)*2*np.pi

        # get a list of random candidates within [r,2r] from the active point
        candidates = np.round((rad*np.cos(theta)+active_list[i][0], rad*np.sin(theta)+active_list[i][1])).astype(np.int32).T

        # trim the list based on boundaries of the array
        candidates = [(x,y) for x,y in candidates if x >= 0 and y >= 0 and x < m and y < m]

        for p in candidates:
            if X[p] < 0 and lonely(p,X,r):
                X[p] = 1
                active_list.append(p)
                break
        else:
            del active_list[i]

    return X

def p_dist(x, y, p):
    # (x-y).T * (x-y)
    y = np.array(y)

    return np.array(list(map(
        lambda x: list(map(lambda y: np.linalg.norm(x-y, p), y))
        , x)))

def mahalanobis_dist(x, y, S):
    # (x-y).T * S * (x-y)
    y = np.array(y)
    S = np.array(S)

    return np.array(list(map(
        lambda x: list(map(lambda y: np.sqrt(np.dot((x-y).T, S).dot(x-y)), y))
        , x)))

plt.figure(figsize=(10, 10))
#plt.subplots_adjust(bottom=.2, top=.95)

xmin = -10
xmax = 10
ymin = -10
ymax = 10

res = 500

xx = np.linspace(xmin, xmax, res)
yy = np.linspace(ymin, ymax, res).T
xx, yy = np.meshgrid(xx, yy)
Xfull = np.c_[xx.ravel(), yy.ravel()]

#print(type(Xfull))
gen_points = np.array([
    [5,3], [-2,-5], [4,-1], [-8,4], [2,8], [3,-8]
])

gen_points = generate_samples(20, 3, 10)
gen_points = np.where(gen_points>0)
gen_points = np.array(gen_points).T + [-10, -10]
#print(gen_points)

gen_points, _ = make_blobs(
    n_samples=100, n_features=2, centers=[[0,0]], cluster_std=2.0
)

#for point in gen_points:
    # find closest point 
#vdist = mahalanobis_dist(Xfull, gen_points, [[0.1,.0],[.0,0.1]])
p = 2
vdist = p_dist(Xfull, gen_points, p)
zones = vdist.argmin(1)
shades = vdist.min(1)

#im_data = zones.reshape((res,res))
#im = Image.fromarray(np.uint8(im_data * 255), 'L')
#im.show()

colormap = cm.get_cmap('rainbow') 
imshow_handle = plt.imshow(zones.reshape((res, res)),
                                   extent=(xmin, xmax, ymin, ymax), origin='lower', cmap=colormap)
plt.plot(gen_points[:,0], gen_points[:,1], 'o', c='k')


plt.savefig('voronoi1.png', dpi=150)