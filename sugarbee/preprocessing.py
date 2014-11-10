def compute_laplacian_matrix(V, E, norm=True, construct="e-neighbor")
    D = None
    W = None
    if construct == "e-neighbor" :
    elif construct == "k-nearest" :
    elif construct == "fully" :
    else
        assert(0)
    if norm==True:
        L = D^(-0.5) * (D-W) * D^(-0.5)
    else :
        L = D - W
    k = max(eigen);

    return L
