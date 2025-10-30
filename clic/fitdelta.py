# block_hybridization_fitter.py (v3.1)

import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)


class HybridizationFitter:
    """
    Fit a pole representation to a matrix valued hybridization Δ(ω).

    Input
      omega_grid: shape (N,)
      delta_complex: shape (N,) or (N, M, M)
      n_lanczos_blocks: number of Lanczos blocks to construct
      n_target_poles: number of poles after merging

    Pipeline
      1) Build a discrete measure μ_k = [−Im Δ(ω_k)/π] * w_k with trapezoid weights w_k
      2) Block Lanczos on the grid using μ_k to get a block Jacobi matrix T and M0
      3) Extract poles ε_j and PSD Hermitian residues R_j from T
      4) Merge adjacent poles using a warped metric controlled by kind and w0
      5) Reconstruct Δ on the grid and report errors
    """

    def __init__(self, omega_grid, delta_complex, n_lanczos_blocks, n_target_poles):
        self.omega = np.asarray(omega_grid, dtype=float)

        delta_in = np.asarray(delta_complex, dtype=np.complex128)
        if delta_in.ndim == 1:
            # promote to (N,1,1)
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

        # results
        self.T_lanczos = None
        self.M0_sqrt_lanczos = None
        self.delta_lanczos = None
        self.eps_lanczos = None
        self.R_lanczos = None
        self.eps_merged = None
        self.R_merged = None
        self.delta_merged = None

        print("HybridizationFitter initialized")
        print(f"  M = {self.m_orb}")
        print(f"  Lanczos blocks = {self.n_lanczos_blocks}")
        print(f"  Target poles = {self.n_target_poles}")

    # ------------- public API -------------

    def run_fit(self, warp_kind="atan", warp_w0=0.1, eta_broadening=0.005):
        print("\n--- Fitting and merging ---")
        self._fit_block_lanczos()
        self._extract_poles_from_T()

        z = self.omega + 1j * eta_broadening
        self.delta_lanczos = self._delta_from_poles(z, self.eps_lanczos, self.R_lanczos)

        self._merge_poles_block(kind=warp_kind, w0=warp_w0)

        self.delta_merged = self._delta_from_poles(z, self.eps_merged, self.R_merged)

        print("--- Done ---")
        print("\nFinal merged poles")
        for i in range(len(self.eps_merged)):
            c = np.sqrt(max(np.real(np.trace(self.R_merged[i])), 0.0))
            print(f"  pole {i}: e = {self.eps_merged[i]:+.6f}, sqrt Tr(R) = {c:.6f}")
        return self

    def analyze(self, eta_broadening=0.005):
        if self.delta_merged is None:
            raise RuntimeError("run_fit must be called first")

        print("\n--- Analysis ---")
        eL_im = self._rel_l2_mat(np.imag(self.delta_input), np.imag(self.delta_lanczos))
        eL_re = self._rel_l2_mat(np.real(self.delta_input), np.real(self.delta_lanczos))
        print(f"Lanczos ({len(self.eps_lanczos)} poles): Im = {eL_im:.3e}, Re = {eL_re:.3e}")

        eM_im = self._rel_l2_mat(np.imag(self.delta_input), np.imag(self.delta_merged))
        eM_re = self._rel_l2_mat(np.real(self.delta_input), np.real(self.delta_merged))
        print(f"Merged  ({self.n_target_poles} poles): Im = {eM_im:.3e}, Re = {eM_re:.3e}")

        chi = self._cost_function_mat(self.eps_merged, self.R_merged, self.omega,
                                      self.delta_input, eta_broadening)
        print(f"Chi squared (Frobenius): {chi:.6e}")
        print("--- End analysis ---")
        return self

    def plot_results(self):
        if self.delta_merged is None:
            print("Call run_fit first")
            return

        sel = (0, 0)
        tr_input = np.trace(self.delta_input, axis1=1, axis2=2)
        tr_lanczos = np.trace(self.delta_lanczos, axis1=1, axis2=2)
        tr_merged = np.trace(self.delta_merged, axis1=1, axis2=2)

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

        axes[0, 0].plot(self.omega, np.real(self.delta_input[:, sel[0], sel[1]]), 'k-', label='Input')
        axes[0, 0].plot(self.omega, np.real(self.delta_lanczos[:, sel[0], sel[1]]), 'r--', label='Lanczos')
        axes[0, 0].plot(self.omega, np.real(self.delta_merged[:, sel[0], sel[1]]), 'b:', lw=2, label='Merged')
        axes[0, 0].set_ylabel(f"Re Δ[{sel[0]},{sel[1]}]")
        axes[0, 0].set_title(f"Element [{sel[0]},{sel[1]}]")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[1, 0].plot(self.omega, np.imag(self.delta_input[:, sel[0], sel[1]]), 'k-')
        axes[1, 0].plot(self.omega, np.imag(self.delta_lanczos[:, sel[0], sel[1]]), 'r--')
        axes[1, 0].plot(self.omega, np.imag(self.delta_merged[:, sel[0], sel[1]]), 'b:', lw=2)
        axes[1, 0].set_xlabel("ω")
        axes[1, 0].set_ylabel(f"Im Δ[{sel[0]},{sel[1]}]")
        axes[1, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(self.omega, np.real(tr_input), 'k-', label='Input')
        axes[0, 1].plot(self.omega, np.real(tr_lanczos), 'r--', label='Lanczos')
        axes[0, 1].plot(self.omega, np.real(tr_merged), 'b:', lw=2, label='Merged')
        axes[0, 1].set_ylabel("Re Tr Δ")
        axes[0, 1].set_title("Trace")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 1].plot(self.omega, np.imag(tr_input), 'k-')
        axes[1, 1].plot(self.omega, np.imag(tr_lanczos), 'r--')
        axes[1, 1].plot(self.omega, np.imag(tr_merged), 'b:', lw=2)
        axes[1, 1].set_xlabel("ω")
        axes[1, 1].set_ylabel("Im Tr Δ")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # ------------- helpers and internals -------------

    @staticmethod
    def load_delta_from_files(file_re, file_im, col_re=5, col_im=5):
        data_re = np.loadtxt(file_re)
        data_im = np.loadtxt(file_im)
        omega_re, re_vals = data_re[:, 0], data_re[:, col_re]
        omega_im, im_vals = data_im[:, 0], data_im[:, col_im]
        if not np.allclose(omega_re, omega_im, rtol=1e-12, atol=1e-14):
            raise ValueError("omega grids do not match")
        return omega_re.astype(float), (re_vals + 1j * im_vals).astype(np.complex128)

    def _fit_block_lanczos(self):
        print(f"1) Block Lanczos with trapezoid weights. Blocks = {self.n_lanczos_blocks}")

        # trapezoid weights on possibly non uniform grid
        dx = np.diff(self.omega)
        w = np.empty_like(self.omega)
        w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
        w[0] = 0.5 * dx[0]
        w[-1] = 0.5 * dx[-1]

        rho_meas = -np.imag(self.delta_input) / np.pi
        mu_grid = [rho_meas[i] * w[i] for i in range(self.n_omega)]

        if self.block_size == -1:
            # collapse to scalar Gauss quadrature exactly like your script
            mu = np.array([float(M[0,0]) for M in mu_grid], dtype=np.float64)
            N = min(self.n_lanczos_blocks, self.n_omega)
            eps, wdiag, T = self._scalar_lanczos_quadrature(self.omega, mu, N)
            # lift back to block form
            self.eps_lanczos = eps
            self.R_lanczos = [np.array([[wj]], dtype=np.float64) for wj in wdiag]
            self.T_lanczos = T
            self.M0_sqrt_lanczos = np.array([[np.sqrt(mu.sum())]], dtype=np.float64)
            return

        A_blocks, B_blocks, _, M0 = self._block_lanczos_alg(
            x_vals=self.omega,
            weight_mats=mu_grid,
            K=self.n_lanczos_blocks,
            b=self.block_size
        )

        # guard: trim B if needed and build T of size (len(B)+1)*b
        if B_blocks and len(B_blocks) != len(A_blocks) - 1:
            B_blocks = B_blocks[:max(0, len(A_blocks) - 1)]

        self.T_lanczos = self._build_block_tridiagonal(A_blocks, B_blocks)
        self.M0_sqrt_lanczos = self._sym_sqrt_psd(M0)
        print(f"   T shape = {self.T_lanczos.shape}")

    def _extract_poles_from_T(self):
        if self.T_lanczos is None:
            raise RuntimeError("run _fit_block_lanczos first")

        print("2) Extracting poles and residues from T")
        if self.T_lanczos.size == 0:
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
            uj = evecs[:, j:j+1]                     # (n_T,1)
            vj = self.M0_sqrt_lanczos.T @ E1.T @ uj  # (b,1)
            Rj = vj @ vj.conj().T                    # PSD Hermitian
            R.append(Rj)
        self.R_lanczos = R
        print(f"   poles extracted = {len(self.eps_lanczos)}")

    def _merge_poles_block(self, kind="atan", w0=0.1):
        if self.eps_lanczos is None:
            raise RuntimeError("run _extract_poles_from_T first")
        print(f"3) Merging to {self.n_target_poles} poles using warp = {kind}, w0 = {w0}")
        self.eps_merged, self.R_merged = self._merge_poles_block_warped(
            self.eps_lanczos, self.R_lanczos, self.n_target_poles, kind=kind, w0=w0
        )

    @staticmethod
    def _delta_from_poles(z, eps, residues):
        z_arr = np.atleast_1d(np.asarray(z, dtype=np.complex128))
        if len(eps) == 0:
            # derive M if possible, else default to 1
            M = residues[0].shape[0] if len(residues) > 0 else 1
            out = np.zeros(z_arr.shape + (M, M), dtype=np.complex128)
            return out[0] if np.isscalar(z) else out

        M = residues[0].shape[0]
        out = np.zeros(z_arr.shape + (M, M), dtype=np.complex128)
        for j, ej in enumerate(eps):
            denom = (z_arr - ej)[..., None, None]
            out += residues[j][None, :, :] / denom
        return out[0] if np.isscalar(z) else out

    @staticmethod
    def _sym_sqrt_psd(A, tol=1e-12):
        A = 0.5 * (A + A.T.conj())
        w, Q = npl.eigh(A)
        w = np.clip(w, 0.0, None)
        return Q @ np.diag(np.sqrt(w)) @ Q.T.conj()

    @classmethod
    def _merge_poles_block_warped(cls, eps, R_list, n_target, kind="atan", w0=0.1):
        phi, inv = cls._make_warp(kind, w0)

        eps = np.asarray(eps, float).copy()
        R = list(R_list)

        order = np.argsort(eps)
        eps = eps[order]
        R = [R[i] for i in order]

        z = phi(eps)
        Tr_R = np.array([np.real(np.trace(Ri)) for Ri in R])

        while len(eps) > n_target:
            tr_prod = Tr_R[:-1] * Tr_R[1:]
            tr_sum = Tr_R[:-1] + Tr_R[1:]
            # variance loss in warped space with scalarization Tr(R)
            costs = (tr_prod / (tr_sum + 1e-300)) * (z[:-1] - z[1:])**2
            i = int(np.argmin(costs))

            R_new = R[i] + R[i+1]
            Tr_new = Tr_R[i] + Tr_R[i+1]

            if Tr_new < 1e-300:
                e_new = 0.5 * (eps[i] + eps[i+1])
            else:
                e_new = (eps[i] * Tr_R[i] + eps[i+1] * Tr_R[i+1]) / Tr_new

            z_new = phi(e_new)

            eps = np.concatenate([eps[:i], [e_new], eps[i+2:]])
            R = R[:i] + [R_new] + R[i+2:]
            Tr_R = np.concatenate([Tr_R[:i], [Tr_new], Tr_R[i+2:]])
            z = np.concatenate([z[:i], [z_new], z[i+2:]])

        return eps, R

    @staticmethod
    def _make_warp(kind="atan", w0=0.1):
        if kind == "atan":
            return (lambda x: np.arctan(x / w0), lambda y: w0 * np.tan(y))
        if kind == "asinh":
            return (lambda x: np.arcsinh(x / w0), lambda y: w0 * np.sinh(y))
        return (lambda x: x, lambda y: y)

    @staticmethod
    def _block_lanczos_alg(x_vals, weight_mats, K, b, tol=1e-12):
        """
        Robust block Lanczos with full reorthogonalization and rank aware steps.
        Inner product is <Φ,Ψ> = sum_i Φ_i^† μ_i Ψ_i, Hermitian symmetrized.
        """
        x_vals = np.asarray(x_vals, float)
        S = len(x_vals)
        M = weight_mats[0].shape[0]

        def blk_ip_grid(Phi, Psi, mu_list):
            # shapes: Phi, Psi -> (S,M,b); mu_list -> (S,M,M)
            mu = np.stack(mu_list, axis=0)                         # (S,M,M)
            acc = np.einsum('smi,smn,snj->ij', 
                            Phi.conj(), mu, Psi, optimize=True)    # (b,b)
            return 0.5*(acc + acc.conj().T)

        def _sqrt_and_pinv_from_gram(G):
            Gs = 0.5 * (G + G.conj().T)
            s, U = npl.eigh(Gs)
            s = np.clip(s, 0.0, None)
            smax = np.max(s) if s.size else 0.0
            thr = max(tol, smax * 1e-12)
            s_clipped = np.where(s > thr, s, 0.0)

            sqrt_s = np.zeros_like(s_clipped)
            pisqrt_s = np.zeros_like(s_clipped)
            mask = s_clipped > 0.0
            sqrt_s[mask] = np.sqrt(s_clipped[mask])
            pisqrt_s[mask] = 1.0 / sqrt_s[mask]

            B = U @ np.diag(sqrt_s) @ U.T.conj()
            B_pinv = U @ np.diag(pisqrt_s) @ U.T.conj()
            eff_rank = int(np.sum(mask))
            return B, B_pinv, eff_rank

        # M0
        M0 = np.zeros((M, M), dtype=np.complex128)
        for Wi in weight_mats:
            M0 += Wi
        M0 = 0.5 * (M0 + M0.conj().T)

        # starting block: identities orthonormalized
        Phi0_raw = np.zeros((S, M, b), dtype=np.complex128)
        for s in range(S):
            Phi0_raw[s] = np.eye(M, b, dtype=np.complex128)

        G0 = blk_ip_grid(Phi0_raw, Phi0_raw, weight_mats)
        _, B0_pinv, _ = _sqrt_and_pinv_from_gram(G0)
        Phi_i = Phi0_raw @ B0_pinv
        Phi_im1 = np.zeros_like(Phi_i)

        A_blocks, B_blocks, Phi_list = [], [], [Phi_i.copy()]
        B_im1 = np.zeros((b, b), dtype=np.complex128)

        for k in range(K):
            XPhi = x_vals[:, None, None] * Phi_i
            Ak = blk_ip_grid(Phi_i, XPhi, weight_mats)
            Ak = 0.5 * (Ak + Ak.conj().T)
            A_blocks.append(Ak)

            W = XPhi - Phi_i @ Ak - Phi_im1 @ B_im1.T.conj()

            # full reorthogonalization, two passes
            for _ in range(2):
                for P in Phi_list:
                    C = blk_ip_grid(P, W, weight_mats)
                    W = W - P @ C

            G = blk_ip_grid(W, W, weight_mats)
            Bk, Bk_pinv, eff_rank = _sqrt_and_pinv_from_gram(G)

            if eff_rank == 0:
                break

            if k < K - 1:
                B_blocks.append(Bk)
                Phi_ip1 = W @ Bk_pinv
                Phi_im1, Phi_i = Phi_i, Phi_ip1
                Phi_list.append(Phi_i.copy())
                B_im1 = Bk
            else:
                break

        return A_blocks, B_blocks, Phi_list, M0

    @staticmethod
    def _build_block_tridiagonal(A_blocks, B_blocks):
        if not A_blocks:
            return np.zeros((0, 0), dtype=float)

        b = A_blocks[0].shape[0]
        K_eff = len(B_blocks) + 1
        A_blocks = A_blocks[:K_eff]

        n = K_eff * b
        T = np.zeros((n, n), dtype=np.complex128)

        for k in range(K_eff):
            T[k*b:(k+1)*b, k*b:(k+1)*b] = A_blocks[k]
            if k < len(B_blocks):
                Bk = B_blocks[k]
                T[k*b:(k+1)*b, (k+1)*b:(k+2)*b] = Bk
                T[(k+1)*b:(k+2)*b, k*b:(k+1)*b] = Bk

        # return Hermitian real if blocks were real
        T = 0.5 * (T + T.conj().T)
        return np.real_if_close(T)

    @staticmethod
    def _scalar_lanczos_quadrature(x, mu, N):
        x = np.asarray(x, float); mu = np.asarray(mu, float)
        N = min(N, x.size)
        def ip(a,b): return float(np.dot(mu, a*b))
        def nrm(a): return np.sqrt(max(ip(a,a), 1e-300))
        v_im1 = np.zeros_like(x); v = np.ones_like(x); v /= nrm(v)
        al, be = [], []
        beta_im1 = 0.0
        for j in range(N):
            w = x*v
            a = ip(v,w); al.append(a)
            w = w - a*v - beta_im1*v_im1
            beta = nrm(w); 
            if j < N-1: be.append(beta)
            v_im1, v = v, w/(beta + 1e-300)
            beta_im1 = beta
        T = np.zeros((N,N))
        i = np.arange(N)
        T[i,i] = al
        j = np.arange(N-1)
        T[j, j+1] = be
        T[j+1, j] = be
        evals, evecs = npl.eigh(T)
        wdiag = mu.sum() * (evecs[0,:]**2)
        return evals, wdiag, T

    @staticmethod
    def _rel_l2_mat(F_true, F_rec):
        diff = F_true - F_rec
        num = np.sum(np.abs(diff)**2)
        den = np.sum(np.abs(F_true)**2) + 1e-300
        return np.sqrt(num / den)

    @classmethod
    def _cost_function_mat(cls, e, R, omega, target_delta, delta):
        model_delta = cls._delta_from_poles(omega + 1j * delta, e, R)
        diff = target_delta - model_delta
        return float(np.sum(np.abs(diff)**2))


if __name__ == "__main__":
    FILE_RE = "real-hyb-Ce4f.dat"
    FILE_IM = "imag-hyb-Ce4f.dat"
    N_LANCZOS_BLOCKS = 101
    N_TARGET_POLES = 3
    ETA_INPUT = 0.005

    try:
        omega_data, delta_data = HybridizationFitter.load_delta_from_files(FILE_RE, FILE_IM)
        fitter = HybridizationFitter(
            omega_grid=omega_data,
            delta_complex=delta_data,
            n_lanczos_blocks=N_LANCZOS_BLOCKS,
            n_target_poles=N_TARGET_POLES
        )
        fitter.run_fit(eta_broadening=ETA_INPUT).analyze(eta_broadening=ETA_INPUT)
        fitter.plot_results()
    except FileNotFoundError:
        print(f"Error. Missing {FILE_RE} or {FILE_IM}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()