import numpy as np 
import matplotlib
from matplotlib import pyplot as plt
import copy 

def compare_with_prototypes(data, prototypes, highlights, cols, idx_dict, exclude=[], cmap='viridis', save=False):
    """
    data(tensor): sl x dim input tensor 
    prototypes(list of tensors): list of sl x dim prototype tensor 
    highlights(list of list of tuples): pair of ranges for every prototype 
    cols(index): column names 
    idx_dict(dict): dictionary assigning which dimension is of which type (discrete, cont or mask)
    """
    assert len(prototypes) == len(highlights)
    
    data = data.detach().numpy() # patients, sl, dim
    prototypes = [prot.detach().numpy() for prot in prototypes]
    
    cont = idx_dict['CONT']
    cont_mask = idx_dict['CONT_MASK']
    disc_mask = idx_dict['DISC_MASK']
    
    disc_data = disc_transform(data, idx_dict['DISC'])
    disc_prot = [disc_transform(prot, idx_dict['DISC']) for prot in prototypes]
    
    # Plotting 
    axes_rows = cont.shape[0] + disc_mask.shape[0] - len(exclude)  
    fig, axes = plt.subplots(axes_rows, 2 * len(prototypes), sharey='row',figsize=(25, 20), squeeze=False)
    cmap = plt.get_cmap(cmap, len(prototypes))
    colors = [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in range(len(prototypes))]
    current = 0
    
    ## Plot continuous features
    for i in np.arange(cont.shape[0]):
        if cols[idx_dict['CONT'][i]] not in exclude:  
            for j in np.arange(len(prototypes)): 
                # Data 
                a = copy.deepcopy(data[:,cont[i]])
                plot_one(axes[current, j*2], a, data[:,cont_mask[i]], cols[idx_dict['CONT'][i]], highlights[j][0], colors[j])
                
                # Prototype
                a = copy.deepcopy(prototypes[j][:,cont[i]])
                plot_one(axes[current, j*2+1], a, prototypes[j][:,cont_mask[i]], cols[idx_dict['CONT'][i]], highlights[j][1], colors[j])
                
            current += 1

        
    ## Plot discrete features 
    ### TODO: Maybe a bar plot would be better for this purpose 
    ### TODO: change xlabels into the descriptions
    for i in np.arange(disc_mask.shape[0]):
        colname = cols[idx_dict['DISC_MASK'][0]][:-5]
        if colname not in exclude:  
            for j in np.arange(len(prototypes)): 
                # Data 
                a = copy.deepcopy(disc_data[:,i])
                plot_one(axes[current, j*2], a, data[:,disc_mask[i]], colname, highlights[j][0], colors[j])
                
                # Prototype
                a = copy.deepcopy(disc_prot[j][:,i])
                plot_one(axes[current, j*2+1], a, prototypes[j][:,disc_mask[i]], colname, highlights[j][1], colors[j])
                
            current += 1
            
    plt.tight_layout()
    if save: 
        fig.savefig(save)
    
        
def disc_transform(a, disc_idx): 
    new = []
    for idx in disc_idx:
        temp = np.array(a)[:,idx] * np.arange(1, len(idx) + 1)
        new.append(temp.sum(axis=1))
    return np.array(new).transpose(1,0)


def plot_one(ax, a, mask, title, highlight, color):
    ax.plot(a, linestyle='dotted')
    a[np.where(mask < 1)] = np.nan
    ax.plot(a, linestyle='solid')
    ax.set_title(title)
    ax.axvspan(highlight[0], highlight[1], color=color, alpha=0.3)