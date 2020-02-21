"""Deals with the simulated experiments of Section 7.1."""
import numpy as np
import matplotlib.pyplot as plt


def get_sol(alpha=0, beta=0):
    """Returns a function that assigns theta_p* to p."""
    if alpha == 0 and beta == 0:
        def sol(p):
            return 0.5
    elif alpha == 1/2 and beta == 1/2:
        def sol(p):
            return ((1-p)**2)/(p**2 + (1-p)**2)
    elif alpha == 1 and beta == 1:
        def sol(p):
            return 1-p
    elif alpha == 2 and beta == 2:
        def sol(p):
            return ((1-p)**(1/2))/(p**(1/2) + (1-p)**(1/2))
    return sol

def get_risk_eval(alpha=0, beta=0):
    """Returns a function that assigns R_P(theta) to theta."""
    def risk_fun(theta, p):
        return p*theta**(1+alpha) + (1-p)*(1-theta)**(1+beta)
    return risk_fun

def plot_mat(alpha=0, beta=0, n_res=100, rel_risk=True, outname="aa"):
    """Plots the values of the excess risk of Figure 5."""
    sol_fun = get_sol(alpha=alpha, beta=beta)
    eval_fun = get_risk_eval(alpha=alpha, beta=beta)
    mat = np.zeros((n_res + 1, n_res + 1))
    val_p = np.linspace(0, 1, n_res + 1)
    for i_test, p_test in enumerate(val_p):
        ref_risk = eval_fun(sol_fun(p_test), p_test)
        for i_train, p_train in enumerate(val_p):
            train_risk = eval_fun(sol_fun(p_train), p_test)
            if rel_risk:
                mat[i_train, i_test] = np.log((train_risk - ref_risk)/ref_risk)
            else:
                mat[i_train, i_test] = train_risk - ref_risk

    plt.figure(figsize=(3, 3))
    mpb = plt.imshow(mat, origin="lower", cmap="gray", extent=[0, 1, 0, 1])
    plt.xlabel("$p'$")
    plt.ylabel("$p$")
    plt.title(r"$\alpha = {}, \beta = {}$".format(alpha, beta))
    plt.clim(0, 1)
    plt.tight_layout()
    plt.savefig("{}.pdf".format(outname), format="pdf")
    # plt.colorbar()

    fig, ax = plt.subplots(figsize=(5, 2))
    plt.colorbar(mpb, ax=ax, orientation='horizontal')
    ax.remove()
    plt.tight_layout()
    plt.savefig("sim_exp/cbar.pdf", bbox_inches='tight')

def plot_dist(alpha=0, beta=0, n_res=100, outname="aa"):
    """Plots the distributions f_+ and f_- from alpha and beta."""
    def pos_dist(x):
        return (1+alpha)*x**alpha
    def neg_dist(x):
        return (1+beta)*(1-x)**beta

    val_x = np.linspace(0, 1, n_res + 1)
    plt.figure(figsize=(3, 3))
    plt.plot(val_x, [pos_dist(x) for x in val_x],
             color="green", label=r"$\sigma=+1$")
    plt.plot(val_x, [neg_dist(x) for x in val_x],
             color="red", label=r"$\sigma=-1$")
    plt.xlabel("$x$")
    plt.ylabel(r"$f_\sigma(x)$")
    plt.title(r"$\alpha = {}, \beta = {}$".format(alpha, beta))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("{}.pdf".format(outname), format="pdf")

if __name__ == "__main__":
    for v in [0, 1/2, 1, 2]:
        plot_mat(alpha=v, beta=v, rel_risk=False,
                 outname="sim_exp/fig_{}_{}".format("abs", v))
        # plot_mat(alpha=v, beta=v, rel_risk=True,
        #         outname="sim_exp/fig_{}_{}".format("rel", v))
        plot_dist(alpha=v, beta=v,
                  outname="sim_exp/fig_{}_{}".format("dist", v))
