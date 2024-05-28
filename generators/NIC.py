import numpy as np

np.random.seed(233)

class NIC_Stream:
    def __init__(self, X,y, n_experiences=10, chunk_size=200, min_classes=2):
        self.X = X
        self.y = y
        self.n_experiences = n_experiences
        self.min_classes = min_classes
        
        self.mask_used = np.zeros_like(self.y)
      
        self.chunk=0
        self.chunk_size=chunk_size
        
        self.max_chunk = int(len(self.y)/chunk_size)
        self.exp_len = []
        
        self.prepare()
        
    
    def prepare(self):
        total_classes = len(np.unique(self.y))
        
        p = np.unique(self.y, return_counts=True)[1]
        
        #niech najpierw każda klasa dostanie przypisanie exp
        class_exp = np.random.choice(self.n_experiences, total_classes)

        while np.sum(class_exp==0) != self.min_classes:
            # sprawdzić, czy się da
            if (total_classes<self.min_classes):
                raise RuntimeError("min_classes greater than total_classes")
            
            #za dużo czy za mało
            if np.sum(class_exp==0) < self.min_classes:
                #wymusić minimum
                random_idx = np.random.choice(total_classes)
                class_exp[random_idx]=0
            if np.sum(class_exp==0) > self.min_classes:
                #usunąć nadmiar
                random_exp = np.random.choice(self.n_experiences)
                first_idx = np.argwhere(class_exp==0).flatten()[0]
                class_exp[first_idx]=random_exp
                
        classes_in_experiences = []
        for e_id in range(self.n_experiences):
            selected = list(np.argwhere(class_exp==e_id).flatten())
            if e_id != 0:
                selected.extend(classes_in_experiences[-1])
            
            classes_in_experiences.append(selected)
            
        ce = [item for row in classes_in_experiences for item in row]
        _ue, _ce = np.unique(ce, return_counts=True)
        e = np.zeros((total_classes))
        e[_ue] = _ce
        
        p = (p/e).astype(int) # Jak w experience pojawi się klasa to ile wylosować obiektów
        
        # Experiences
        all_exp_indexes = []
        for e_id, e in enumerate(classes_in_experiences):
            exp_indexes = []
            
            for ce in e:
                possible_mask = np.ones_like(self.y)
                possible_mask[self.y != ce] = 0
                possible_mask[self.mask_used == 1] = 0
                select_from = np.argwhere(possible_mask==1).flatten()
                
                selection = np.random.choice(select_from, p[ce], replace=False)
                
                self.mask_used[selection] = 1
                exp_indexes.append(selection)
                
            # flatten in experience and shuffle
            exp_indexes_flat = np.array([item for row in exp_indexes for item in row])
            rand_perm = np.random.permutation(len(exp_indexes_flat))

            exp_indexes_flat = exp_indexes_flat[rand_perm]
            
            all_exp_indexes.append(exp_indexes_flat)
            self.exp_len.append(len(exp_indexes_flat))
        
        self.order = np.array([item for row in all_exp_indexes for item in row])
        
    # build a stream from experiences
    def get_chunk(self):
        if self.chunk > self.max_chunk:
            return 
        
        start = self.chunk_size*self.chunk
        end =  start + self.chunk_size
        
        indexes = self.order[start:end]
        
        self.chunk+=1
        
        return self.X[indexes], self.y[indexes]       
        
