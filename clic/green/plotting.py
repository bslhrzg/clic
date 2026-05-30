import numpy as np
import matplotlib.pyplot as plt

def plot_spectral_function(ws, A_w, impurity_indices, title, filename=None, 
                           toev=False):
    """
    Plots and save the impurity spectral function A(w) and saves it to a file.
    """
    print(f"impurity_indices = {impurity_indices}")
    if len(np.shape(A_w)) == 3:
        dos = np.sum(A_w[:, impurity_indices, impurity_indices], axis=1).real
    elif len(np.shape(A_w)) == 2:
        dos = np.sum(A_w[:, impurity_indices], axis=1).real

    ry2ev = 1
    if toev :
        ry2ev = 13.6057039763
    np.savetxt('A_w.dat', np.c_[ws * ry2ev,dos])

    plt.figure(figsize=(8, 5))
    plt.plot(ws, dos, label="Total Impurity DOS")
    plt.title(title)
    plt.xlabel("Frequency (ω)")
    plt.ylabel("A(ω) (arb. units)")
    plt.grid(True)
    plt.legend()
    
    if filename:
        plt.savefig(filename, dpi=300)
        print(f"Plot saved to '{filename}'")
    else:
        plt.show()

    plt.close()