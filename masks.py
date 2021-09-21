#masks.py

import numpy as np
from scipy.spatial.distance import cdist

class Mask():

    def mask1d(N,cutoff,periodic=False,sparsity = False):

        d = np.zeros((N,1))
        d[:,0] = np.arange(0,N)
        dst = cdist(d,d)

        if periodic:
            dst = min(dst,N-dst)

        else:
            dst = dst   

        if(sparsity):
            interactions = (dst<cutoff).astype(int)
            sparsity = sum(sum(interactions))/N**2
            return interactions,sparsity

        else:
            interactions = (dst<cutoff).astype(int)
            return interactions

    def mask2d(N_x,N_y,cutoff,periodic,sparsity = False):

        x1 = np.linspace(-(N_x)//2,(N_x)//2-1,N_x)
        x1 = np.expand_dims(x1,axis=1)

        x2 = np.linspace(-(N_y)//2,(N_y)//2-1,N_y)
        x2 = np.expand_dims(x2,axis=1)

        x_coordinates = np.expand_dims(np.repeat(x1,N_y,axis = 0).reshape(N_x,N_y).transpose().flatten(),axis=1)
        y_coordinates = np.expand_dims(np.repeat(x2,N_x,axis = 0).reshape(N_x,N_y).flatten(),axis=1)
        
        #calculate torus distance on 2d sheet
        distances_x = cdist(x_coordinates,x_coordinates)
        distances_y = cdist(y_coordinates,y_coordinates)
        
        if (periodic==True):
            distances_y = np.minimum(N_y-distances_y,distances_y)
            distances_x = np.minimum(N_x-distances_x,distances_x)
        
        distances = np.sqrt(np.square(distances_x) + np.square(distances_y))
        dist = distances.reshape(N_y,N_x,N_y,N_x)
        dist = dist.reshape(N_x*N_y,N_x*N_y)
        dist[dist<cutoff] = 1
        dist[dist>cutoff-1] = 0

        if (sparsity):
            return dist, sum(sum(dist))/(N_x*N_y)**2
            return dist

    #random sparse mask
    def sparsemask2d(N_x,N_y,sparsity):
        elements = np.random.uniform(0,1,(N_x,N_y))
        mask = (elements<sparsity).astype(int)
        return mask

    #Duplicate and diversify mask: Ricard V. Sole et al, A Model of Large-Scale Proteome Evolution.
    def dupdivmask(N,alpha,delta):
        #N = max size of network
        N = 256

        #probability of pruning old edges
        delta = delta

        #probability of adding new edges
        alpha = alpha

        #adjacency matrix (of directed graph)
        interactions = np.zeros((N,N))

        #start with 2 nodes connected to each other:
        interactions[0,1] = 1
        interactions[1,0] = 1

        for i in range(2,N):
            #at the beginning of timestep i, network size = i

            #choose one of the older nodes from the set {0,1,2,....,i-1} 
            duplicate = np.random.choice(i+1)

            #duplicate its interactions
            interactions[i,:] = interactions[duplicate,:]

            #prune a subset of these interactions with prob = delta
            interactions[i,:] = interactions[i,:]*(np.random.uniform(0,1,N)>delta).astype(int)

            #add new edges from this node to the rest with prob = alpha
            interactions[:,i] = (np.random.uniform(0,1,N)<alpha).astype(int)
        
        return interactions

    # TODO: don't use for now (what is p??)
    #small world mask
    def smallworldmask(N, cutoff, p):
        interactions = Mask.mask1d(N,cutoff,periodic = True)- np.eye(N)
        return interactions

#apply h2h mask
# N_x = 16
# N_y = 16
# periodic = False
# d = 3
# sparsity = 0.03228759765625
# plt.figure(figsize = (15,15))
# plt.imshow(mask2d(N_x,N_y,d,False),cmap ='jet');plt.colorbar();plt.title('2d Mask with '+ 'd=' + str(d))