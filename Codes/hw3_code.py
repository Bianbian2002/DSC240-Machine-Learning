from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def sample_gaussian(mu, Sigma, n=100, seed=0):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean=mu, cov=Sigma, size=n)

def plot_boundary_and_samples(score_fn, XA, XB, title="", save_path=None):
    # score_fn(x) > 0 => predict A
    X = np.vstack([XA, XB])
    x_min, x_max = X[:,0].min()-1.0, X[:,0].max()+1.0
    y_min, y_max = X[:,1].min()-1.0, X[:,1].max()+1.0

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = np.array([score_fn(pt) for pt in grid]).reshape(xx.shape)

    plt.figure()
    plt.scatter(XA[:,0], XA[:,1], marker='o', label="Class A")
    plt.scatter(XB[:,0], XB[:,1], marker='x', label="Class B")
    plt.contour(xx, yy, zz, levels=[0.0])  # boundary where score=0
    plt.title(title)
    plt.legend()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.show()

# TODO: define mu1, mu2, priors, and Sigma1/Sigma2 per part.
# TODO: define score_fn(x) = log π1 + log N(x|μ1,Σ1) - [log π2 + log N(x|μ2,Σ2)]

def log_gaussian_pdf(x, mu, Sigma):
    x = np.asarray(x)
    mu = np.asarray(mu)
    d = x.shape[0]
    sign, logdet = np.linalg.slogdet(Sigma)
    inv = np.linalg.inv(Sigma)
    quad = (x - mu).T @ inv @ (x - mu)
    return -0.5 * (d*np.log(2*np.pi) + logdet + quad)

def make_score_fn(mu1, Sigma1, pi1, mu2, Sigma2, pi2):
    def score(x):
        return (np.log(pi1) + log_gaussian_pdf(x, mu1, Sigma1)
                - (np.log(pi2) + log_gaussian_pdf(x, mu2, Sigma2)))
    return score


def main():
    # Q3(a) parameters: mu1=[0,0], mu2=[1,1], Sigma1=Sigma2=I, pi1=pi2=0.5
    mu1 = np.array([0.0, 0.0])
    mu2 = np.array([1.0, 1.0])
    Sigma1 = np.eye(2)
    Sigma2 = np.eye(2)
    pi1 = 0.5
    pi2 = 0.5

    score_fn = make_score_fn(mu1, Sigma1, pi1, mu2, Sigma2, pi2)
    XA = sample_gaussian(mu1, Sigma1, n=100, seed=0)
    XB = sample_gaussian(mu2, Sigma2, n=100, seed=1)
    out_path = Path(__file__).resolve().parents[1] / "Figures" / "hw3" / "q3a_boundary.png"
    plot_boundary_and_samples(
        score_fn,
        XA,
        XB,
        title="Q3(a): equal priors, equal covariances",
        save_path=out_path,
    )

    # Q3(b) parameters: mu1=[0,0]^T, mu2=[1,1]^T, pi1=p=0.8, pi2=0.2
    pi1 = 0.8
    pi2 = 0.2

    score_fn = make_score_fn(mu1, Sigma1, pi1, mu2, Sigma2, pi2)
    XA = sample_gaussian(mu1, Sigma1, n=100, seed=2)
    XB = sample_gaussian(mu2, Sigma2, n=100, seed=3)
    out_path = Path(__file__).resolve().parents[1] / "Figures" / "hw3" / "q3b_boundary.png"
    plot_boundary_and_samples(
        score_fn,
        XA,
        XB,
        title="Q3(b): unequal priors (p=0.8)",
        save_path=out_path,
    )

    # Q3(c) parameters: Sigma1=sigma^2 I, Sigma2=I, p=0.5, sigma^2=0.25
    sigma2 = 0.25
    Sigma1 = sigma2 * np.eye(2)
    Sigma2 = np.eye(2)
    pi1 = 0.5
    pi2 = 0.5

    score_fn = make_score_fn(mu1, Sigma1, pi1, mu2, Sigma2, pi2)
    XA = sample_gaussian(mu1, Sigma1, n=100, seed=4)
    XB = sample_gaussian(mu2, Sigma2, n=100, seed=5)
    out_path = Path(__file__).resolve().parents[1] / "Figures" / "hw3" / "q3c_boundary.png"
    plot_boundary_and_samples(
        score_fn,
        XA,
        XB,
        title="Q3(c): Sigma1=sigma^2 I, sigma^2=0.25",
        save_path=out_path,
    )

    # Q3(d) parameters: mu1=mu2=[0,0]^T, p=0.5, sigma^2=0.25
    mu1 = np.array([0.0, 0.0])
    mu2 = np.array([0.0, 0.0])
    sigma2 = 0.25
    Sigma1 = sigma2 * np.eye(2)
    Sigma2 = np.eye(2)
    pi1 = 0.5
    pi2 = 0.5

    score_fn = make_score_fn(mu1, Sigma1, pi1, mu2, Sigma2, pi2)
    XA = sample_gaussian(mu1, Sigma1, n=100, seed=6)
    XB = sample_gaussian(mu2, Sigma2, n=100, seed=7)
    out_path = Path(__file__).resolve().parents[1] / "Figures" / "hw3" / "q3d_boundary.png"
    plot_boundary_and_samples(
        score_fn,
        XA,
        XB,
        title="Q3(d): mu1=mu2, sigma^2=0.25",
        save_path=out_path,
    )


if __name__ == "__main__":
    main()