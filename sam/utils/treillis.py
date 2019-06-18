"""Treillis."""
import numpy as np
from copy import deepcopy
from tqdm import tqdm

def decompress_list(l):
    out = []

    for i in l:
        if type(i) == int:
            out = l
            break
        if type(i) == list:
         
            if type(i[0]) == list:
                for j in i:
                    out.append(j)
            else:
                out.append(i)
    return out


def treillis(adj_m, a, b, path=[], maxlength=5):
    """ Find all the acyclic paths between vars.
    """
    if len(path) != 0:
        cause = path[-1]
        if cause == b:
            return path
    else:
        cause = a
        path = [a]
        
    output = []

    for i in np.nonzero(adj_m[cause, :])[0]:
        if i not in path and len(path) < maxlength:
            output.append(treillis(adj_m, a, b, path+[i], maxlength))
    if len(output) == 0:
        return None
    output = [j for j in output if j is not None]
  

    return decompress_list(output) if len(output)>0 else None
    
    
def compute_total_effect(adj_m, gradient, maxlength=5):

    row, col = adj_m.shape
    
    total_effect_m = np.zeros((gradient.shape[0], row, col))
    
    for i in range(adj_m.shape[0]):    
        for j in range(adj_m.shape[1]):
        
            all_path = treillis(adj_m, i, j, [], maxlength)

            total_effect = np.zeros((gradient.shape[0], 1))      
            
            if(all_path is not None):   
                for path in all_path:
            
                    path_effect = np.ones((gradient.shape[0], 1))     
                
                    for k in range(len(path)-1):
                
                        cause = path[k]
                        effect = path[k+1]
                   
                        path_effect = path_effect*np.reshape(gradient[:, cause,effect], (gradient.shape[0],1))
                  
                    total_effect += path_effect

            total_effect_m[:, i,j] =  list(total_effect)
             
    return total_effect_m     
                 


