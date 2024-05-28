from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from generators.NIC import NIC_Stream
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from sklearn.neighbors import KernelDensity

n_classes = 8
n_features = 2

X, y = make_blobs(n_samples=100000, centers=n_classes, n_features = 2, cluster_std=1.5, random_state=17688)

stream = NIC_Stream(X, y, min_classes=2, chunk_size=400)
kde = KernelDensity(kernel='gaussian')
clf = GaussianNB()

__X, __y = [], []

o_bac = []
c_bac = []
num_outer = []

accumulated_samples = []
gt_class = []
pred_class = []

fig, ax = plt.subplots(2,3,figsize=(15,10))
plt.suptitle('Stationary example | 2 known + 6 unknown | fit on 10 chunks')
# 
xmin, xmax = np.min(X[:,0]), np.max(X[:,0])
ymin, ymax = np.min(X[:,1]), np.max(X[:,1])
xx, yy = np.meshgrid(
    np.linspace(xmin, xmax,150),
    np.linspace(ymin, ymax,150)
)
mesh = np.array([xx.flatten(), yy.flatten()]).T

for chunk in range(stream.max_chunk):
    _X, _y = stream.get_chunk()
    print(np.unique(_y, return_counts=True))
    
    #train with 10 chunks
    if chunk<10:
        __X.append(_X)
        __y.append(_y)
        known_y = np.unique(_y)
    
    if chunk==10:
        print('known_y', known_y)
        __X = np.array(__X).reshape((-1,n_features))
        __y = np.array(__y).flatten()   
        
        mask_known = np.zeros((__X.shape[0])).astype(bool)
        mask_known[__y == known_y[0]]=1
        mask_known[__y == known_y[1]]=1
             
        kde.fit(__X[mask_known], __y[mask_known])
        score_mesh = kde.score_samples(mesh)
        score_mesh = np.exp(score_mesh)
        
        # oblicz próg w oparciu znanych
        pred_proba_known = kde.score_samples(__X[mask_known])
        pred_proba_known = np.exp(pred_proba_known)
        th = np.mean(pred_proba_known) - np.std(pred_proba_known)
        print(th)
        # exit()
                
        clf.fit(__X[mask_known], __y[mask_known])

    
    if chunk>10:
   
        mask_known = np.zeros((_X.shape[0])).astype(bool)
        mask_known[_y == known_y[0]]=1
        mask_known[_y == known_y[1]]=1
        
        
        #classification (closed)
        clf_preds = clf.predict(_X[mask_known])
        clf_preds_all = clf.predict(_X)
        c_bac.append(balanced_accuracy_score(_y[mask_known], clf_preds))
              
        #open
        pred_proba = kde.score_samples(_X)
        pred_proba = np.exp(pred_proba)
        
        pred_known = (pred_proba>th).astype(int)
        print(np.unique(pred_known, return_counts=True))

        # store samples
        accumulated_samples.extend(_X)
        
        #known
        gt_class = np.full((_X.shape[0]), 0.2).astype(float)
        gt_class[_y == known_y[0]]=-1
        gt_class[_y == known_y[1]]=1
        
        pred_class = np.copy(clf_preds_all).astype(float)
        pred_class[pred_class==known_y[0]] = -1
        pred_class[pred_class==known_y[1]] = 1
        pred_class[pred_known==0] = 0.2
                
        o_bac.append(balanced_accuracy_score(mask_known, pred_known))
        print(o_bac[-1])
                        
        _accumulated_samples = np.array(_X)
        _gt_class = gt_class
        _pred_class = pred_class

        print(_accumulated_samples.shape)
        print(_gt_class.shape)
        print(_pred_class.shape)     

        ax[0,0].scatter(_accumulated_samples[:,0], _accumulated_samples[:,1], alpha=0.95, s=10, c=_gt_class, cmap='coolwarm', marker='o')
        ax[0,1].scatter(_accumulated_samples[:,0], _accumulated_samples[:,1], alpha=0.95, s=10, c=_pred_class, cmap='coolwarm', marker='o', )
        ax[0,2].scatter(mesh[:,0], mesh[:,1], alpha=0.5, s=15, c=score_mesh, cmap='coolwarm')

        ax[0,0].set_xlabel('$x_1$')
        ax[0,0].set_ylabel('$x_2$')
        ax[0,1].set_xlabel('$x_1$')
        ax[0,2].set_xlabel('$x_1$')
        
        aax = plt.subplot(2,1,2)
        aax.plot(median_filter(o_bac,5, mode='nearest'), color='blue', label='Open (KDE)')
        aax.plot(median_filter(c_bac,5, mode='nearest'), color='red', label='Closed (GNB)')
        aax.set_xlim(0,stream.max_chunk-10)
        aax.legend(frameon=False)
        aax.grid(ls=':')
        aax.set_ylabel('balanced accuracy')
        aax.vlines((stream.max_chunk/2)-10, 0.5, 1, color='gray', ls=':')
        aax.set_ylim(0.5,1)
    

        for aa in ax[0]:
            aa.grid(ls=':')
            aa.spines['top'].set_visible(False)
            aa.spines['right'].set_visible(False)
            aa.set_xlim(xmin,xmax)
            aa.set_ylim(ymin,ymax)
        for aa in ax[1]:
            aa.set_xticks([])
            aa.set_yticks([])
            
        aax.spines['top'].set_visible(False)
        aax.spines['right'].set_visible(False)
        aax.set_xlabel('chunk')
        
        plt.tight_layout()
        plt.savefig('foo.png')
        # plt.savefig('mov/%04d.png' % (chunk-10))

        for aa in ax.ravel():
            aa.cla()
            aax.cla()


            

