# hybfit/hybfit_poles.py

import numpy as np
import numpy.linalg as npl
from . import utils
from ..io_utils import vprint 

class HybFitPoles:
    """
    Fit a pole representation to a matrix-valued hybridization Δ(ω) using
    a block Lanczos algorithm. Handles scalar hybridizations as a special case.
    """

    def __init__(self, omega_grid, delta_complex, n_lanczos_blocks, n_target_poles, logfile = None):

        self.logfile = logfile

        self.omega = np.asarray(omega_grid, dtype=float)

        delta_in = np.asarray(delta_complex, dtype=np.complex128)
        if delta_in.ndim == 1:
            self.delta_input = delta_in[:, np.newaxis, np.newaxis]
        elif delta_in.ndim == 3:
            self.delta_input = delta_in
        else:
            raise ValueError("delta_complex must be 1D or 3D")

        self.n_omega, self.m_orb, _ = self.delta_input.shape
        if self.omega.shape[0] != self.n_omega:
            raise ValueError("omega_grid and delta_complex length mismatch")

        self.block_size = self.m_orb
        self.n_lanczos_blocks = int(n_lanczos_blocks)
        self.n_target_poles = int(n_target_poles)


        # Final results
        self.eps_merged = None
        self.R_merged = None

        vprint(3,"HybFitPoles initialized:",filename=self.logfile)
        vprint(3,f"  Orbital dimensions M = {self.m_orb}",filename=self.logfile)
        vprint(3,f"  Lanczos blocks = {self.n_lanczos_blocks}",filename=self.logfile)
        vprint(3,f"  Target poles = {self.n_target_poles}",filename=self.logfile)

    def run_fit(self, warp_kind="atan", warp_w0=0.1):
        """
        Execute the full fitting pipeline: Lanczos -> Pole Extraction -> Merging.
        """
        vprint(3,"--- Fitting poles via Lanczos ---",filename=self.logfile)
        self._fit_lanczos()
        self._extract_poles_from_T()
        
        vprint(3,"--- Merging poles ---",filename=self.logfile)
        self._merge_poles_block(kind=warp_kind, w0=warp_w0)
        
        vprint(3,"--- Pole Fit Done ---",filename=self.logfile)
        vprint(3,"Final merged poles:",filename=self.logfile)
        for i in range(len(self.eps_merged)):
            c = np.sqrt(max(np.real(np.trace(self.R_merged[i])), 0.0))
            vprint(3,f"  pole {i}: e = {self.eps_merged[i]:+.6f}, sqrt Tr(R) = {c:.6f}")
        return self

    def _fit_lanczos(self):
        """Dispatcher for scalar vs. block Lanczos algorithm."""
        if self.m_orb == 1:
            vprint(3,"1) Using scalar Lanczos for M=1 case.",filename=self.logfile)
            self._scalar_lanczos()
        else:
            vprint(3,f"1) Using block Lanczos for M={self.m_orb} case.",filename=self.logfile)
            self._block_lanczos()

    def _scalar_lanczos(self):
        # build measure
        dx = np.diff(self.omega)
        w = np.empty_like(self.omega)
        w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
        w[0] = 0.5 * dx[0]; w[-1] = 0.5 * dx[-1]
        
        rho_meas = -np.imag(self.delta_input[:, 0, 0]) / np.pi
        mu = rho_meas * w
        
        N = min(self.n_lanczos_blocks, self.n_omega)
        x = self.omega
        
        # Lanczos algorithm
        def ip(a,b): return float(np.dot(mu, a*b))
        def nrm(a): return np.sqrt(max(ip(a,a), 1e-300))
        
        v_im1 = np.zeros_like(x); v = np.ones_like(x); v /= nrm(v)
        al, be = [], []
        beta_im1 = 0.0
        for j in range(N):
            w_vec = x*v
            a = ip(v,w_vec); al.append(a)
            w_vec = w_vec - a*v - beta_im1*v_im1
            beta = nrm(w_vec); 
            if j < N-1: be.append(beta)
            v_im1, v = v, w_vec/(beta + 1e-300)
            beta_im1 = beta
        
        T = np.zeros((N,N))
        i = np.arange(N); T[i,i] = al
        j = np.arange(N-1); T[j, j+1] = be; T[j+1, j] = be
        
        self.T_lanczos = T
        # For consistency with block method, define M0_sqrt
        self.M0_sqrt_lanczos = np.array([[np.sqrt(mu.sum())]])

    def _block_lanczos(self):
        dx = np.diff(self.omega)
        w = np.empty_like(self.omega)
        w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
        w[0] = 0.5 * dx[0]; w[-1] = 0.5 * dx[-1]
        
        rho_meas = -np.imag(self.delta_input) / np.pi
        mu_grid = [rho_meas[i] * w[i] for i in range(self.n_omega)]
        
        A_blocks, B_blocks, _, M0 = self._block_lanczos_alg(
            x_vals=self.omega, weight_mats=mu_grid,
            K=self.n_lanczos_blocks, b=self.block_size
        )
        
        if B_blocks and len(B_blocks) != len(A_blocks) - 1:
            B_blocks = B_blocks[:max(0, len(A_blocks) - 1)]

        self.T_lanczos = self._build_block_tridiagonal(A_blocks, B_blocks)
        self.M0_sqrt_lanczos = self._sym_sqrt_psd(M0)
        vprint(3,f"   T shape = {self.T_lanczos.shape}")

    def _extract_poles_from_T(self):
        vprint(3,"2) Extracting poles and residues from T",filename=self.logfile)
        if self.T_lanczos is None or self.T_lanczos.size == 0:
            self.eps_lanczos, self.R_lanczos = np.array([]), []
            return

        evals, evecs = npl.eigh(self.T_lanczos)
        self.eps_lanczos = evals

        b = self.block_size
        n_T = self.T_lanczos.shape[0]
        E1 = np.zeros((n_T, b))
        E1[:b, :b] = np.eye(b)

        R = []
        for j in range(n_T):
            uj = evecs[:, j:j+1]
            vj = self.M0_sqrt_lanczos.T @ E1.T @ uj
            Rj = vj @ vj.conj().T
            R.append(Rj)
        self.R_lanczos = R
        vprint(3,f"   poles extracted = {len(self.eps_lanczos)}",filename=self.logfile)

    def _merge_poles_block_(self, kind, w0):
        vprint(3,f"3) Merging to {self.n_target_poles} poles using warp = {kind}, w0 = {w0}",filename=self.logfile)
        self.eps_merged, self.R_merged = self._merge_poles_block_warped(
            self.eps_lanczos, self.R_lanczos, self.n_target_poles, kind, w0, self.logfile
        )

    def _merge_poles_block(self, kind, w0):
        vprint(3,"merge poles, kind = ",kind,filename=self.logfile)
        eps = self.eps_lanczos
        W = eps.max() - eps.min()
        E_keep = 0.005*W
        vprint(3,f"W={W},E_keep = {E_keep}",filename=self.logfile)
        vprint(3,f"3) Merging to {self.n_target_poles} poles with Appendix F + low-E bias",filename=self.logfile)
        self.eps_merged, self.R_merged = self._merge_poles_block_appendixF(
            self.eps_lanczos, self.R_lanczos, self.n_target_poles,
            cleanup_negative=False, cull_outliers=True,
            bias="fd", w0=E_keep, p=2.0, Tstar=E_keep, gamma=0.0,
            use_warp_spacing=True, warp_kind=kind  # reuse your 'kind' for spacing only
    )
        
    
    @staticmethod
    def _sym_sqrt_psd(A, tol=1e-12):
        A = 0.5 * (A + A.T.conj())
        w, Q = npl.eigh(A)
        w = np.clip(w, 0.0, None)
        return Q @ np.diag(np.sqrt(w)) @ Q.T.conj()

    @classmethod
    def _merge_poles_block_warped(cls, eps, R_list, n_target, kind, w0,filename=None):
        phi, inv = cls._make_warp(kind, w0,filename=filename)
        eps = np.asarray(eps, float).copy()
        R = list(R_list)
        order = np.argsort(eps)
        eps = eps[order]; R = [R[i] for i in order]
        z = phi(eps)
        Tr_R = np.array([np.real(np.trace(Ri)) for Ri in R])
        while len(eps) > n_target:
            costs = (Tr_R[:-1] * Tr_R[1:] / (Tr_R[:-1] + Tr_R[1:] + 1e-300)) * (z[:-1] - z[1:])**2
            i = int(np.argmin(costs))
            R_new = R[i] + R[i+1]
            Tr_new = Tr_R[i] + Tr_R[i+1]
            z_new = (Tr_R[i]*z[i] + Tr_R[i+1]*z[i+1]) / (Tr_new + 1e-300) if Tr_new > 1e-300 else 0.5*(z[i]+z[i+1])
            e_new = inv(z_new)
            eps = np.concatenate([eps[:i], [e_new], eps[i+2:]])
            R = R[:i] + [R_new] + R[i+2:]
            Tr_R = np.concatenate([Tr_R[:i], [Tr_new], Tr_R[i+2:]])
            z = np.concatenate([z[:i], [z_new], z[i+2:]])
        return eps, R

    @staticmethod
    def _make_warp(kind="atan", w0=0.1,filename=None):
        vprint(3,"make warp : kind = ",kind,filename=filename)
        if kind == "atan": return (lambda x: np.arctan(x / w0), lambda y: w0 * np.tan(y))
        if kind == "asinh": return (lambda x: np.arcsinh(x / w0), lambda y: w0 * np.sinh(y))
        if kind == "const": return (lambda x: x, lambda y: y)
        return (lambda x: x, lambda y: y)

    @classmethod
    def _merge_poles_block_appendixF(cls, eps, R_list, n_target,
                                    cleanup_negative=False, cull_outliers=True, W_est=None,
                                    bias="fd", w0=0.01, p=2.0, Tstar=None, gamma=0.5,
                                    use_warp_spacing=True, warp_kind="atanh"):
        """
        Appendix F with low-energy emphasis via biased selection.
        bias: "none" | "power" | "fd"
        w0, p: parameters for h_power
        Tstar: parameter for h_fd (if None uses w0)
        gamma: 0..1 controls spacing factor ℓ_k^gamma
        use_warp_spacing: if True, compute ℓ_k in z=phi(eps) space
        warp_kind: "asinh" | "atan" | "const" for spacing only
        """
        vprint(3,f"bias = {bias}, w0 = {w0}, p = {p}, gamma = {gamma}, warp_kind = {warp_kind}",filename=self.logfile)
        tiny = 1e-300
        eps = np.asarray(eps, float).copy()
        R = [np.array(Ri, dtype=np.complex128, copy=True) for Ri in R_list]

        order = np.argsort(eps)
        eps = eps[order]; R = [R[i] for i in order]

        def weights(R): return np.array([float(np.real(np.trace(Ri))) for Ri in R], dtype=float)

        # spacing warp used only for the selection metric
        def make_phi(kind, w0):
            if kind == "atan":  return lambda x: np.arctan(x / w0)
            if kind == "asinh": return lambda x: np.arcsinh(x / w0)
            return lambda x: x
        phi = make_phi(warp_kind, max(w0, 1e-12))
        Tstar = Tstar if Tstar is not None else w0

        def h_abs_e(e):
            x = abs(e)
            if bias == "power":
                return 1.0 + (x/max(w0,1e-12))**p
            if bias == "fd":
                #return 1.0 + np.exp(x/max(Tstar,1e-12))
                arg = abs(e)/max(Tstar,1e-12)
                return 1.0 + np.exp(np.clip(arg, 0.0, 50.0))
            return 1.0
        

        def eliminate_at(k, eps, R):
            a = eps; w = weights(R)
            akm1, ak, akp1 = a[k-1], a[k], a[k+1]
            wkms, wk, wkps = w[k-1], w[k], w[k+1]
            denom = max(akp1 - akm1, tiny)
            fac_L = (akp1 - ak)/denom
            fac_R = (ak   - akm1)/denom
            # new energies from first-moment conservation
            akm1_num = wkms*akm1*(akp1 - akm1) + wk*ak*(akp1 - ak)
            akm1_den = wkms*(akp1 - akm1) + wk*(akp1 - ak) + tiny
            akm1_new = akm1_num/akm1_den
            akp1_num = wkps*akp1*(akp1 - akm1) + wk*ak*(ak - akm1)
            akp1_den = wkps*(akp1 - akm1) + wk*(ak   - akm1) + tiny
            akp1_new = akp1_num/akp1_den
            # matrix residues with same partition
            Rkm1_new = R[k-1] + fac_L*R[k]
            Rkp1_new = R[k+1] + fac_R*R[k]
            eps2 = np.concatenate([a[:k-1], [akm1_new, akp1_new], a[k+2:]])
            R2 = R[:k-1] + [Rkm1_new, Rkp1_new] + R[k+2:]
            # enforce strict monotonicity if needed
            if k-2 >= 0 and not (eps2[k-2] < eps2[k-1]):
                eps2[k-1] = np.nextafter(eps2[k-2], np.inf)
            if k   < len(eps2)-1 and not (eps2[k-1] < eps2[k]):
                eps2[k] = np.nextafter(eps2[k-1], np.inf)
            return eps2, R2

        # optional: cull absurd outliers first
        if cull_outliers and len(eps) > n_target:
            w = weights(R)
            if W_est is None:
                lo, hi = (np.percentile(eps, [1, 99]) if len(eps) > 20 else (eps.min(), eps.max()))
                W_est = max(hi - lo, 1.0)
            keep = np.ones_like(eps, dtype=bool)
            keep &= ~(np.abs(eps) > 10*W_est) | (w > 1e-7)
            if keep.sum() >= n_target:
                eps = eps[keep]; R = [Ri for Ri, k in zip(R, keep) if k]
                order = np.argsort(eps); eps = eps[order]; R = [R[i] for i in order]

        # negative-weight cleanup if requested
        if cleanup_negative:
            while True:
                w = weights(R)
                candidates = np.where(w[1:-1] < 0)[0] + 1
                if candidates.size == 0 or len(eps) <= max(n_target, 2): break
                # remove most negative first
                k = int(candidates[np.argmin(w[candidates])])
                eps, R = eliminate_at(k, eps, R)

        # main reduction with biased selection
        while len(eps) > max(n_target, 2):
            w = weights(R)
            interior = np.arange(1, len(eps)-1)
            # compute selection scores
            if use_warp_spacing:
                z = phi(eps)
                dL = z[interior] - z[interior-1]
                dR = z[interior+1] - z[interior]
            else:
                dL = eps[interior] - eps[interior-1]
                dR = eps[interior+1] - eps[interior]
            ell = np.minimum(dL, dR)
            h = np.array([h_abs_e(eps[i]) for i in interior], dtype=float)
            rho_loc = w[interior-1] + w[interior] + w[interior+1]

            # prioritize any negative weight if present
            neg_mask = w[interior] <= 0
            if np.any(neg_mask):
                idxs = interior[neg_mask]
                k = int(idxs[np.argmin(w[idxs])])
            else:
                # selection score
                S = (w[interior] / h) * np.maximum(ell, tiny)**gamma * np.maximum(rho_loc, tiny)**0.0
                k = int(interior[np.argmin(S)])

            eps, R = eliminate_at(k, eps, R)

        #return eps, R
        # handle the small-N tail exactly
        if len(eps) > n_target:
            # here len(eps) is 2 and n_target is 1
            assert len(eps) == 2 and n_target == 1, "unexpected small-N case"
            w = weights(R)
            if use_warp_spacing:
                z = phi(eps)
                z_new = (w[0]*z[0] + w[1]*z[1]) / (w[0] + w[1] + tiny)
                # need inverse warp corresponding to selection space
                if warp_kind == "atan":
                    inv = lambda y: w0 * np.tan(y)
                elif warp_kind == "asinh":
                    inv = lambda y: w0 * np.sinh(y)
                else:
                    inv = lambda y: y
                e_new = float(inv(z_new))
            else:
                e_new = float((w[0]*eps[0] + w[1]*eps[1]) / (w[0] + w[1] + tiny))
            R_new = R[0] + R[1]
            eps = np.array([e_new], dtype=float)
            R = [R_new]

        return eps, R

    @staticmethod
    def _block_lanczos_alg(x_vals, weight_mats, K, b, tol=1e-12):
        S, M = len(x_vals), weight_mats[0].shape[0]
        def blk_ip_grid(Phi, Psi, mu_list):
            mu = np.stack(mu_list, axis=0)
            acc = np.einsum('smi,smn,snj->ij', Phi.conj(), mu, Psi, optimize=True)
            return 0.5*(acc + acc.conj().T)
        def _sqrt_and_pinv_from_gram(G):
            s, U = npl.eigh(0.5 * (G + G.conj().T))
            s_clipped = np.clip(s, 0.0, None)
            thr = max(tol, (np.max(s_clipped) if s_clipped.size else 0.0) * 1e-12)
            mask = s_clipped > thr
            sqrt_s, pisqrt_s = np.zeros_like(s_clipped), np.zeros_like(s_clipped)
            sqrt_s[mask] = np.sqrt(s_clipped[mask])
            pisqrt_s[mask] = 1.0 / sqrt_s[mask]
            B = U @ np.diag(sqrt_s) @ U.T.conj()
            B_pinv = U @ np.diag(pisqrt_s) @ U.T.conj()
            return B, B_pinv, int(np.sum(mask))
        M0 = sum(weight_mats)
        M0 = 0.5 * (M0 + M0.conj().T)
        Phi0_raw = np.zeros((S, M, b), dtype=np.complex128)
        for s in range(S): Phi0_raw[s] = np.eye(M, b, dtype=np.complex128)
        G0 = blk_ip_grid(Phi0_raw, Phi0_raw, weight_mats)
        _, B0_pinv, _ = _sqrt_and_pinv_from_gram(G0)
        Phi_i = Phi0_raw @ B0_pinv
        Phi_im1 = np.zeros_like(Phi_i)
        A_blocks, B_blocks, Phi_list = [], [], [Phi_i.copy()]
        B_im1 = np.zeros((b, b), dtype=np.complex128)
        for k in range(K):
            XPhi = x_vals[:, None, None] * Phi_i
            Ak = blk_ip_grid(Phi_i, XPhi, weight_mats)
            A_blocks.append(0.5 * (Ak + Ak.conj().T))
            W = XPhi - Phi_i @ Ak - Phi_im1 @ B_im1.T.conj()
            for _ in range(2):
                for P in Phi_list: W -= P @ blk_ip_grid(P, W, weight_mats)
            G = blk_ip_grid(W, W, weight_mats)
            Bk, Bk_pinv, eff_rank = _sqrt_and_pinv_from_gram(G)
            if eff_rank == 0 or k == K - 1: break
            B_blocks.append(Bk)
            Phi_ip1 = W @ Bk_pinv
            Phi_im1, Phi_i = Phi_i, Phi_ip1
            Phi_list.append(Phi_i.copy())
            B_im1 = Bk
        return A_blocks, B_blocks, Phi_list, M0

    @staticmethod
    def _build_block_tridiagonal(A_blocks, B_blocks):
        if not A_blocks: return np.zeros((0, 0), dtype=float)
        b = A_blocks[0].shape[0]
        K_eff = len(B_blocks) + 1
        n = K_eff * b
        T = np.zeros((n, n), dtype=np.complex128)
        for k in range(K_eff):
            T[k*b:(k+1)*b, k*b:(k+1)*b] = A_blocks[k]
            if k < len(B_blocks):
                Bk = B_blocks[k]
                T[k*b:(k+1)*b, (k+1)*b:(k+2)*b] = Bk
                T[(k+1)*b:(k+2)*b, k*b:(k+1)*b] = Bk
        return np.real_if_close(0.5 * (T + T.conj().T))