import numpy as np

def gf2_gaus_elim(gf2_matrix: np.array) -> np.array:
    """
    Function that performs Gaussian elimination over GF2(2)
    GF is the initialism of Galois field, another name for finite fields.

    GF(2) may be identified with the two possible values of a bit and to the boolean values true and false.

    pseudocode: http://dde.binghamton.edu/filler/mct/hw/1/assignment.pdf

    Args:
        gf2_matrix (np.array): GF(2) binary matrix to preform Gaussian elimination over
    Returns:
        gf2_matrix_rref (np.array): reduced row echelon form of M
    """
    gf2_matrix_rref = gf2_matrix.copy()
    m_rows, n_cols = gf2_matrix_rref.shape

    row_i = 0
    col_j = 0

    while row_i < m_rows and col_j < n_cols:

        if sum(gf2_matrix_rref[row_i:, col_j]) == 0:
            # case when col_j all zeros
            # No pivot in this column, pass to next column
            col_j += 1
            continue

        # find index of row with first "1" in the vector defined by column j (note previous if statement removes all zero column)
        k = np.argmax(gf2_matrix_rref[row_i:, col_j]) + row_i
        # + row_i gives correct index (as we start search from row_i!)

        # swap row k and row_i (row_i now has 1 at top of column j... aka: gf2_matrix_rref[row_i, col_j]==1)
        gf2_matrix_rref[[k, row_i]] = gf2_matrix_rref[[row_i, k]]
        # next need to zero out all other ones present in column j (apart from on the i_row!)
        # to do this use row_i and use modulo addition to zero other columns!

        # make a copy of j_th column of gf2_matrix_rref, this includes all rows (0 -> M)
        Om_j = np.copy(gf2_matrix_rref[:, col_j])

        # zero out the i^th position of vector Om_j (this is why copy needed... to stop it affecting gf2_matrix_rref)
        Om_j[row_i] = 0
        # note this was orginally 1 by definition...
        # This vector now defines the indices of the rows we need to zero out
        # by setting ith position to zero - it stops the next steps zeroing out the i^th row (which we need as our pivot)


        # next from row_i of rref matrix take all columns from j->n (j to last column)
        # this is vector of zero and ones from row_i of gf2_matrix_rref
        i_jn = gf2_matrix_rref[row_i, col_j:]
        # we use i_jn to zero out the rows in gf2_matrix_rref[:, col_j:] that have leading one (apart from row_i!)
        # which rows are these? They are defined by that Om_j vector!

        # the matrix to zero out these rows is simply defined by the outer product of Om_j and i_jn
        # this creates a matrix of rows of i_jn terms where Om_j=1 otherwise rows of zeros (where Om_j=0)
        Om_j_dependent_rows_flip = np.einsum('i,j->ij', Om_j, i_jn, optimize=True)
        # note flip matrix is contains all m rows ,but only j->n columns!

        # perfrom bitwise xor of flip matrix to zero out rows in col_j that that contain a leading '1' (apart from row i)
        gf2_matrix_rref[:, col_j:] = np.bitwise_xor(gf2_matrix_rref[:, col_j:], Om_j_dependent_rows_flip)

        row_i += 1
        col_j += 1

    return gf2_matrix_rref


def gf2_basis_for_gf2_rref(gf2_matrix_in_rreform: np.array) -> np.array:
    """
    Function that gets the kernel over GF2(2) of ow reduced  gf2 matrix!

    uses method in: https://en.wikipedia.org/wiki/Kernel_(linear_algebra)#Basis

    Args:
        gf2_matrix_in_rreform (np.array): GF(2) matrix in row reduced form
    Returns:
        basis (np.array): basis for gf2 input matrix that was in row reduced form
    """
    rows_to_columns = gf2_matrix_in_rreform.T
    eye = np.eye(gf2_matrix_in_rreform.shape[1], dtype=int)

    # do column reduced form as row reduced form
    rrf = gf2_gaus_elim(np.hstack((rows_to_columns, eye.T)))

    zero_rrf = np.where(~rrf[:, :gf2_matrix_in_rreform.shape[0]].any(axis=1))[0]
    basis = rrf[zero_rrf, gf2_matrix_in_rreform.shape[0]:]

    return basis

#########################################################################
#### For now uses the legacy code for identifying noncontextual sets ####
###################### TODO graph techniques! ###########################
#########################################################################

from datetime import datetime
from datetime import timedelta

# Takes two Pauli operators specified as strings (e.g., 'XIZYZ') and determines whether they commute:
def commute(x,y):
    assert len(x)==len(y), print(x,y)
    s = 1
    for i in range(len(x)):
        if x[i]!='I' and y[i]!='I' and x[i]!=y[i]:
            s = s*(-1)
    if s==1:
        return 1
    else:
        return 0

# Input: S, a list of Pauli operators specified as strings.
# Output: a boolean indicating whether S is contextual or not.
def contextualQ(S,verbose=False):
    # Store T all elements of S that anticommute with at least one other element in S (takes O(|S|**2) time).
    T=[]
    Z=[] # complement of T
    for i in range(len(S)):
        if any(not commute(S[i],S[j]) for j in range(len(S))):
            T.append(S[i])
        else:
            Z.append(S[i])
    # Search in T for triples in which exactly one pair anticommutes; if any exist, S is contextual.
    for i in range(len(T)): # WLOG, i indexes the operator that commutes with both others.
        for j in range(len(T)):
            for k in range(j,len(T)): # Ordering of j, k does not matter.
                if i!=j and i!=k and commute(T[i],T[j]) and commute(T[i],T[k]) and not commute(T[j],T[k]):
                    if verbose:
                        return [True,None,None]
                    else:
                        return True
    if verbose:
        return [False,Z,T]
    else:
        return False

def greedy_dfs(ham,cutoff,criterion='weight'):
    
    weight = {k:abs(ham[k]) for k in ham.keys()}
    possibilities = [k for k, v in sorted(weight.items(), key=lambda item: -item[1])] # sort in decreasing order of weight
    
    best_guesses = [[]]
    stack = [[[],0]]
    start_time = datetime.now()
    delta = timedelta(seconds=cutoff)
    
    i = 0
    
    while datetime.now()-start_time < delta and stack:
        
        while i < len(possibilities):
#             print(i)
            next_set = stack[-1][0]+[possibilities[i]]
#             print(next_set)
#             iscontextual = contextualQ(next_set)
#             print('  ',iscontextual,'\n')
            if not contextualQ(next_set):
                stack.append([next_set,i+1])
            i += 1
        
        if criterion == 'weight':
            new_weight = sum([abs(ham[p]) for p in stack[-1][0]])
            old_weight = sum([abs(ham[p]) for p in best_guesses[-1]])
            if new_weight > old_weight:
                best_guesses.append(stack[-1][0])
                # print(len(stack[-1][0]))
                # print(stack[-1][0],'\n')
            
        if criterion == 'size' and len(stack[-1][0]) > len(best_guesses[-1]):
            best_guesses.append(stack[-1][0])
            # print(len(stack[-1][0]))
            # print(stack[-1][0],'\n')
            
        top = stack.pop()
        i = top[1]
    
    return best_guesses

def pauli_mult(p,q):
    assert(len(p)==len(q))
    sgn=1
    out=''
    for i in range(len(p)):
        if p[i]=='I':
            out+=q[i]
        elif q[i]=='I':
            out+=p[i]
        elif p[i]=='X':
            if q[i]=='X':
                out+='I'
            elif q[i]=='Y':
                out+='Z'
                sgn=sgn*1j
            elif q[i]=='Z':
                out+='Y'
                sgn=sgn*-1j
        elif p[i]=='Y':
            if q[i]=='Y':
                out+='I'
            elif q[i]=='Z':
                out+='X'
                sgn=sgn*1j
            elif q[i]=='X':
                out+='Z'
                sgn=sgn*-1j
        elif p[i]=='Z':
            if q[i]=='Z':
                out+='I'
            elif q[i]=='X':
                out+='Y'
                sgn=sgn*1j
            elif q[i]=='Y':
                out+='X'
                sgn=sgn*-1j
    return [out,sgn]

def to_indep_set(G_w_in):
    G_w = G_w_in
    G_w_keys = [[str(g),1] for g in G_w.keys()]
    G_w_keys_orig = [str(g) for g in G_w.keys()]
    generators = []
    for i in range(len(G_w_keys[0][0])):
        # search for first X,Y,Z in ith position
        fx=None
        fy=None
        fz=None
        j=0
        while fx==None and j<len(G_w_keys):
            if G_w_keys[j][0][i]=='X' and not any(G_w_keys[j][0]==g[0] for g in generators):
                fx=G_w_keys[j]
            j+=1
        j=0
        while fy==None and j<len(G_w_keys):
            if G_w_keys[j][0][i]=='Y' and not any(G_w_keys[j][0]==g[0] for g in generators):
                fy=G_w_keys[j]
            j+=1
        j=0
        while fz==None and j<len(G_w_keys):
            if G_w_keys[j][0][i]=='Z' and not any(G_w_keys[j][0]==g[0] for g in generators):
                fz=G_w_keys[j]
            j+=1
        # multiply to eliminate all other nonidentity entries in ith position
        if fx!=None:
            generators.append(fx)
            for j in range(len(G_w_keys)):
                if G_w_keys[j][0][i]=='X': # if any other element of G_w has 'X' in the ith position...
                    # multiply it by fx
                    G_w[G_w_keys_orig[j]]=G_w[G_w_keys_orig[j]]+[fx]
                    sgn=G_w_keys[j][1]*fx[1]
                    G_w_keys[j]=pauli_mult(G_w_keys[j][0],fx[0])
                    G_w_keys[j][1]=G_w_keys[j][1]*sgn
        
        if fz!=None:
            generators.append(fz)
            # if any other element of G_w has 'Z' in the ith position...
            for j in range(len(G_w_keys)):
                if G_w_keys[j][0][i]=='Z': 
                    # multiply it by fz
                    G_w[G_w_keys_orig[j]]=G_w[G_w_keys_orig[j]]+[fz] # update the factor list for G_w_keys[j]
                    sgn=G_w_keys[j][1]*fz[1] # update the sign for G_w_keys[j]
                    G_w_keys[j]=pauli_mult(G_w_keys[j][0],fz[0]) # multiply G_w_keys[j] by fz...
                    G_w_keys[j][1]=G_w_keys[j][1]*sgn # ... and by the associated sign.
        
        if fx!=None and fz!=None:
            for j in range(len(G_w_keys)):
                if G_w_keys[j][0][i]=='Y': # if any other element of G_w has 'Y' in the ith position...
                    # multiply it by fx and fz
                    G_w[G_w_keys_orig[j]]=G_w[G_w_keys_orig[j]]+[fx,fz]
                    sgn=G_w_keys[j][1]*fx[1]
                    G_w_keys[j]=pauli_mult(G_w_keys[j][0],fx[0])
                    G_w_keys[j][1]=G_w_keys[j][1]*sgn
                    sgn=G_w_keys[j][1]*fz[1]
                    G_w_keys[j]=pauli_mult(G_w_keys[j][0],fz[0])
                    G_w_keys[j][1]=G_w_keys[j][1]*sgn
        # If both fx and fz are not None, then at this point we are done with this position.
        # Otherwise, there may be remaining 'Y's at this position:
        elif fy!=None:
            generators.append(fy)
            # if any other element of G_w has 'Y' in the ith position...
            for j in range(len(G_w_keys)):
                if G_w_keys[j][0][i]=='Y': 
                    # multiply it by fy
                    G_w[G_w_keys_orig[j]]=G_w[G_w_keys_orig[j]]+[fy]
                    sgn=G_w_keys[j][1]*fy[1]
                    G_w_keys[j]=pauli_mult(G_w_keys[j][0],fy[0])
                    G_w_keys[j][1]=G_w_keys[j][1]*sgn
    for j in range(len(G_w_keys)):
        G_w[G_w_keys_orig[j]]=G_w[G_w_keys_orig[j]]+[G_w_keys[j]]
    
    return generators, G_w