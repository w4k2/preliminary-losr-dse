from sklearn.datasets import make_blobs
from generators.NIC import NIC_Stream
import numpy as np
import matplotlib.pyplot as plt

n_classes = 8
n_features = 2

X, y = make_blobs(n_samples=50000, centers=n_classes, n_features = 2, cluster_std=1.5, random_state=17688)

stream = NIC_Stream(X, y, min_classes=2)

accumulated_samples_before = []
gt_class_before = []

accumulated_samples_after = []
gt_class_after = []

fig, ax = plt.subplots(1,2,figsize=(10,5))
plt.suptitle('Before and after data shift')

__X = []
__y = []

for chunk in range(stream.max_chunk):
    _X, _y = stream.get_chunk()
    print(np.unique(_y, return_counts=True))
    
    #train with 10 chunks
    if chunk<10:
        __X.append(_X)
        __y.append(_y)
        known_y = np.unique(_y)
    
    # concept shift in the second part:
    if chunk>0.5*stream.max_chunk:
        phi = 0.5
        print('bef', _X[0])

        affine_matrix = [[np.cos(phi), -np.sin(phi), 0],[np.sin(phi), np.cos(phi), 0],[0,0,1]]
        _X_e = np.column_stack((_X, np.zeros(_X.shape[0])))
        _X = (_X_e@affine_matrix)[:,:2]
        # print(_X.shape)
        print('aft', _X[0])
        # exit()
        
        mask_known = np.zeros((_X.shape[0])).astype(bool)
        mask_known[_y == known_y[0]]=1
        mask_known[_y == known_y[1]]=1
        
        accumulated_samples_after.extend(_X)
        gt_class_after.extend(mask_known)
        
    else:

        mask_known = np.zeros((_X.shape[0])).astype(bool)
        mask_known[_y == known_y[0]]=1
        mask_known[_y == known_y[1]]=1
        
        accumulated_samples_before.extend(_X)
        gt_class_before.extend(mask_known)

                        

_accumulated_samples_before = np.array(accumulated_samples_before)
_gt_class_before = np.array(gt_class_before)

_accumulated_samples_after= np.array(accumulated_samples_after)
_gt_class_after = np.array(gt_class_after)


ax[0].scatter(_accumulated_samples_before[:,0], _accumulated_samples_before[:,1], alpha=0.15, s=5, c=_gt_class_before, cmap='coolwarm', marker='x')
ax[1].scatter(_accumulated_samples_after[:,0], _accumulated_samples_after[:,1], alpha=0.15, s=5, c=_gt_class_after, cmap='coolwarm', marker='x')

ax[0].set_title('chunks %i : %i' % (0, stream.max_chunk/2))
ax[1].set_title('chunks %i : %i' % (stream.max_chunk/2, stream.max_chunk))

for aa in ax:
    aa.grid(ls=':')
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)
    # aa.set_xlim(xmin,xmax)
    # aa.set_ylim(ymin,ymax)

    
plt.tight_layout()
plt.savefig('foo.png')


            

