import numpy as np

def get_inv_weiss(ws, h_imp, hyb, eta):
    """
    Compute inverse non-interacting Weiss field:
        G0^{-1}(ω) = (ω + iη) I − h_imp − Δ(ω)

    Parameters
    ----------
    ws   : array_like, shape (n_w,)
        Real frequency grid.
    h_imp: array_like, shape (n_imp, n_imp)
        One-body impurity Hamiltonian.
    hyb  : array_like, shape (n_w, n_imp, n_imp) or None
        Hybridization function Δ(ω). If None -> isolated impurity (HIA).
    eta  : float
        Positive broadening.

    Returns
    -------
    inv_G0 : ndarray, shape (n_w, n_imp, n_imp)
    """

    ws   = np.asarray(ws,   dtype=np.float64)
    h_imp = np.asarray(h_imp, dtype=np.complex128)
    n_imp = h_imp.shape[0]

    # Construct (w + iη) I for each ω:
    # shape (n_w, n_imp, n_imp)
    zI = (ws + 1j * eta)[:, None, None] * np.eye(n_imp, dtype=np.complex128)[None, :, :]

    if hyb is None:
        return zI - h_imp[None, :, :]
    else:
        hyb = np.asarray(hyb, dtype=np.complex128)
        return zI - h_imp[None, :, :] - hyb


def invert_G(G):
    """
    Invert G(w) for each w slice independently, with w as leading index.

    Parameters
    ----------
    G : array_like, shape (n_w, n_orb, n_orb)
        Impurity Green's function G(w).

    Returns
    -------
    inv_G : ndarray, shape (n_w, n_orb, n_orb)
        Matrix inverse G(ω)^{-1} for each frequency.
    """
    G = np.asarray(G, dtype=np.complex128)
    n_w, n_orb, _ = G.shape

    # Identity for each ω: shape (n_w, n_orb, n_orb)
    I = np.eye(n_orb, dtype=np.complex128)
    rhs = np.broadcast_to(I, (n_w, n_orb, n_orb))

    # This solves, for each w_i:
    #     G(w_i) X(w_i) = I
    # which is mathematically equivalent to X(w_i) = G(w_i)^{-1},
    # but is numerically safer than forming the inverse explicitly.
    inv_G = np.linalg.solve(G, rhs)

    return inv_G

def get_sigma(inv_G0,inv_G):
    """
    Compute the self energy as 
    ∑(w) = inv(G0(w)) - inv(G(w))
    """
    return inv_G0 - inv_G


