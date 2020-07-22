import numpy as np 
import matplotlib
import matplotlib.plotly as plt 
import copy 

def visualise_tensors(tensors, cols, idx_dict):
    a = tensors.detach().numpy() # patients, sl, dim
    cont = idx_dict['CONT']
    cont_mask = idx_dict['CONT_MASK']
    disc_mask = idx_dict['DISC_MASK']
    
    disc_a = disc_transform(a, idx_dict['DISC'])
    
    # Plot continuous features 
    for i in np.arange(cont.shape[0]):
        plt.figure()
        plot_one_fea(a[:,:,cont[i]], a[:,:,cont_mask[i]], cols[idx_dict['CONT'][i]])

    # Plot discrete features 
    # TODO: Maybe a bar plot would be better for this purpose 
    # TODO: change xlabels into the descriptions
    for i in np.arange(disc_mask.shape[0]):
        plt.figure()
        colname = cols[idx_dict['DISC_MASK'][0]][:-5]
        plot_one_fea(disc_a[:,:,i], a[:,:,disc_mask[i]], colname)
    
def disc_transform(a, disc_idx): 
    new = []
    for idx in disc_idx:
        temp = np.array(a)[:,:,idx] * np.arange(1, len(idx) + 1)
        new.append(temp.sum(axis=2))
    return np.array(new).transpose(1,2,0)
    
def plot_one_fea(a, mask, title, cmap='Spectral'):
    a = copy.deepcopy(a)
    cmap = plt.get_cmap(cmap, a.shape[0])
    colors = [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in range(a.shape[0])]
    
    for i in range(len(colors)): 
        plt.plot(a[i], linestyle='dotted', color=colors[i])
    a[np.where(mask < 1)] = np.nan
    for i in range(len(colors)): 
        plt.plot(a[i], linestyle='solid', color=colors[i])
    
    plt.title(title)