import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)


def make_matheron_demo(a=4, b=3, n_ens=8, length_scale=0.4, seed=0):
    rng = np.random.default_rng(seed)
    d = a + b

    # 1D coordinates for x and y blocks (with a gap to visually separate them)
    x_pos = np.linspace(0.0, 1.0, a)
    y_pos = np.linspace(2.0, 3.0, b)
    pos   = np.concatenate([x_pos, y_pos])  # shape (d,)

    # Squared-exponential covariance on this line
    D = pos[:, None] - pos[None, :]
    C = np.exp(-(D**2) / (2 * length_scale**2))
    C += 1e-6 * np.eye(d)  # jitter

    # Prior ensemble: N(0, C)
    L = np.linalg.cholesky(C)
    Z_prior = rng.standard_normal(size=(n_ens, d)) @ L.T  # shape (N, d)

    # Choose observation y* as the y-block of one prior member (makes the update easy to see)
    y_star = Z_prior[0, a:].copy()

    # Empirical Matheron update for full [x,y]
    X = Z_prior[:, :a]
    Y = Z_prior[:, a:]
    N = n_ens

    AX = (X - X.mean(axis=0, keepdims=True)) / np.sqrt(N - 1)
    AY = (Y - Y.mean(axis=0, keepdims=True)) / np.sqrt(N - 1)

    C_xy = AX.T @ AY                    # (a, b)
    C_yy = AY.T @ AY + 1e-5 * np.eye(b) # (b, b) + nugget
    C_zy = np.vstack([C_xy, C_yy])      # (a+b, b)

    K = C_zy @ np.linalg.inv(C_yy)      # (a+b, b)

    Delta = y_star[None, :] - Y         # (N, b)
    Z_post = Z_prior + Delta @ K.T      # (N, a+b)

    return pos, Z_prior, Z_post, a, b, y_star


def plot_matheron_three_panel(pos, Z_prior, Z_post, a, b, y_star):
    n_ens, d = Z_prior.shape
    assert d == a + b

    colors = plt.cm.tab10(np.linspace(0, 1, n_ens))
    sample_ids = np.arange(n_ens)

    fig = plt.figure(figsize=(12, 4))
    ax_prior  = fig.add_subplot(1, 3, 1, projection='3d')
    ax_update = fig.add_subplot(1, 3, 2, projection='3d')
    ax_post   = fig.add_subplot(1, 3, 3, projection='3d')

    # Shared limits
    zmin = min(Z_prior.min(), Z_post.min())
    zmax = max(Z_prior.max(), Z_post.max())
    zpad = 0.1 * (zmax - zmin + 1e-9)
    zmin -= zpad
    zmax += zpad

    def setup_axes(ax, title):
        ax.set_xlim(pos.min() - 0.2, pos.max() + 0.2)
        ax.set_ylim(-1, n_ens)
        ax.set_zlim(zmin, zmax)
        # ax.set_xlabel("grid coordinate (x-block | y-block)")
        # ax.set_ylabel("ensemble member index")
        # ax.set_zlabel("value")
        ax.set_title(title)
        # Suppress tick labels (values are arbitrary)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        # Useful viewing angle
        ax.view_init(elev=25, azim=-60)

        # Optional: vertical line marking the boundary between x and y
        boundary = pos[a-1] + (pos[a] - pos[a-1]) / 2 if a < d else pos[a-1]
        ax.plot([boundary, boundary], [-1, n_ens], [zmin, zmin],
                "k:", alpha=0.4)

    # Panel 1: prior ensemble
    setup_axes(ax_prior, "Prior ensemble")
    for i, c in zip(sample_ids, colors):
        ax_prior.plot(
            pos,
            np.full_like(pos, i),
            Z_prior[i, :],
            "-o",
            color=c,
            lw=1,
            ms=3,
        )

    # Show observed y* as a reference "ridge" at pseudo-index -0.5
    ax_prior.plot(
        pos[a:],
        np.full(b, -0.5),
        y_star,
        "k--",
        lw=2,
        label="$y^*$"
    )
    ax_prior.legend(loc="upper left")

    # Panel 2: update vectors (prior→posterior displacement)
    setup_axes(ax_update, "Matheron displacement")
    for i, c in zip(sample_ids, colors):
        for j, xj in enumerate(pos):
            ax_update.plot(
                [xj, xj],
                [i, i],
                [Z_prior[i, j], Z_post[i, j]],
                color=c,
                alpha=0.7,
                lw=1,
            )

    # Panel 3: posterior ensemble
    setup_axes(ax_post, "Posterior ensemble")
    for i, c in zip(sample_ids, colors):
        ax_post.plot(
            pos,
            np.full_like(pos, i),
            Z_post[i, :],
            "-o",
            color=c,
            lw=1,
            ms=3,
        )

    # Overplot y* again (now all ensemble y-blocks coincide with this line)
    ax_post.plot(
        pos[a:],
        np.full(b, -0.5),
        y_star,
        "k--",
        lw=2,
        label="$y^*$"
    )
    ax_post.legend(loc="upper left")

    # Use subplots_adjust instead of tight_layout for reliable 3D plot margins
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.12, top=0.95, wspace=0.25)
    return fig


def plot_matheron_stage_extruded(pos, Z_prior, Z_post, a, b, y_star):
    n_ens, d = Z_prior.shape

    colors = plt.cm.tab10(np.linspace(0, 1, n_ens))
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")

    zmin = min(Z_prior.min(), Z_post.min())
    zmax = max(Z_prior.max(), Z_post.max())
    zpad = 0.1 * (zmax - zmin + 1e-9)
    zmin -= zpad
    zmax += zpad

    ax.set_xlim(pos.min() - 0.2, pos.max() + 0.2)
    ax.set_ylim(-0.2, 1.2)   # stage from 0 to 1
    ax.set_zlim(zmin, zmax)

    # ax.set_xlabel("grid coordinate (x-block | y-block)")
    # ax.set_ylabel("stage (0 = prior, 1 = posterior)")
    # ax.set_zlabel("value")
    ax.set_title("Matheron update as path in stage")

    # Suppress tick labels (values are arbitrary)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Boundary between x and y blocks
    boundary = pos[a-1] + (pos[a] - pos[a-1]) / 2 if a < d else pos[a-1]
    ax.plot([boundary, boundary], [0, 1], [zmin, zmin], "k:", alpha=0.4)

    # Ensemble curves
    for i, c in enumerate(colors):
        # prior at stage 0
        ax.plot(
            pos,
            np.zeros_like(pos),
            Z_prior[i, :],
            "-",
            color=c,
            alpha=0.7,
        )
        # posterior at stage 1
        ax.plot(
            pos,
            np.ones_like(pos),
            Z_post[i, :],
            "-",
            color=c,
            alpha=0.7,
        )

        # Optional: vertical connectors per grid point
        for j, xj in enumerate(pos):
            ax.plot(
                [xj, xj],
                [0, 1],
                [Z_prior[i, j], Z_post[i, j]],
                color=c,
                alpha=0.4,
                lw=0.7,
            )

    # Highlight y* trajectory across stage
    ax.plot(
        pos[a:],
        np.zeros(b),
        Z_prior[0, a:],  # prior y from member 0 (which we used as y*)
        "k--",
        lw=1.5,
        label="prior y (member 0)",
    )
    ax.plot(
        pos[a:],
        np.ones(b),
        y_star,
        "k-",
        lw=2,
        label="$y^*$ (posterior y)",
    )

    ax.legend(loc="upper left")
    ax.view_init(elev=25, azim=-60)
    # Use subplots_adjust instead of tight_layout for reliable 3D plot margins
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.92)
    return fig
