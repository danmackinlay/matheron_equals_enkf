#!/usr/bin/env python3
# Copyright (c) 2025 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
#
# All rights reserved.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate illustrative Matheron/EnKF ensemble update plots.

This script creates small, didactic figures showing how a Matheron update
acts on a partitioned state vector [x, y] via an ensemble mapping.

It is intended for:
- inclusion in the paper as a figure; and
- exploratory use from the command line / Quarto notebooks.

Typical usage:

    uv run python da_gp/scripts/plot_matheron.py --mode three-panel
    uv run python da_gp/scripts/plot_matheron.py --mode both --out figures/matheron.pdf

The underlying construction lives in da_gp/src/matheron_viz.py.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from da_gp.src.figstyle import setup_figure_style
from da_gp.src.logging_setup import get_logger, setup_logging
from da_gp.src.matheron_viz import (
    make_matheron_demo,
    plot_matheron_stage_extruded,
    plot_matheron_three_panel,
)

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate illustrative Matheron/EnKF ensemble update plots"
    )

    # Logging options (consistent with other scripts)
    parser.add_argument(
        "--log-level",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        default="WARNING",
        help="Set the logging level (default: WARNING)",
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        help="Use JSON formatting for logs",
    )

    # Output configuration
    parser.add_argument(
        "--out",
        default="figures/matheron_3panel.pdf",
        help=(
            "Output figure path. For --mode=both, this is treated as a stem and "
            "suffixes '_3panel' and '_stage' are added before the extension "
            "(default: figures/matheron_3panel.pdf)"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively instead of closing them",
    )

    # Demo configuration
    parser.add_argument(
        "--a",
        type=int,
        default=4,
        help="Size of x-block (default: 4)",
    )
    parser.add_argument(
        "--b",
        type=int,
        default=3,
        help="Size of y-block (default: 3)",
    )
    parser.add_argument(
        "--n-ens",
        type=int,
        default=8,
        help="Ensemble size (default: 8)",
    )
    parser.add_argument(
        "--length-scale",
        type=float,
        default=0.4,
        help="Length scale for the toy covariance kernel (default: 0.4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible toy ensemble (default: 0)",
    )

    # Plot mode
    parser.add_argument(
        "--mode",
        choices=["three-panel", "stage-extruded", "both"],
        default="three-panel",
        help=(
            "Which visualization to generate: 'three-panel' (prior/update/posterior), "
            "'stage-extruded' (prior→posterior as a stage dimension), or 'both' "
            "(generates two files) (default: three-panel)"
        ),
    )

    # Styling
    parser.add_argument(
        "--colorblind-friendly",
        action="store_true",
        help="Use color-blind friendly palette via da_gp.figstyle",
    )

    return parser


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _split_out_path(out: Path) -> tuple[Path, str, str]:
    """Split an output path into (directory, stem, suffix).

    If there is no suffix, '.pdf' is assumed.
    """
    if out.suffix:
        return out.parent, out.stem, out.suffix
    return out.parent, out.name, ".pdf"


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    # Logging
    setup_logging(args.log_level, json=args.log_json)

    # Styling (JMLR-like; colors optionally adjusted)
    setup_figure_style(colorblind_friendly=args.colorblind_friendly)

    # Build toy [x, y] ensemble and its posterior via empirical Matheron update
    logger.info(
        "Constructing Matheron demo with a=%d, b=%d, n_ens=%d, length_scale=%.3f, seed=%d",
        args.a,
        args.b,
        args.n_ens,
        args.length_scale,
        args.seed,
    )

    pos, Z_prior, Z_post, a, b, y_star = make_matheron_demo(
        a=args.a,
        b=args.b,
        n_ens=args.n_ens,
        length_scale=args.length_scale,
        seed=args.seed,
    )

    out_path = Path(args.out)
    out_dir, stem, suffix = _split_out_path(out_path)
    _ensure_parent_dir(out_dir)

    # Generate requested figure(s)
    if args.mode in ("three-panel", "both"):
        logger.info("Generating three-panel Matheron ensemble figure")
        fig_3panel = plot_matheron_three_panel(pos, Z_prior, Z_post, a, b, y_star)

        if args.mode == "three-panel":
            save_path = out_dir / f"{stem}{suffix}"
        else:
            save_path = out_dir / f"{stem}_3panel{suffix}"

        fig_3panel.savefig(save_path, bbox_inches="tight", dpi=450)
        logger.info("Saved three-panel figure to %s", save_path)

        png_path = save_path.with_suffix(".png")
        fig_3panel.savefig(png_path, bbox_inches="tight", dpi=450)
        logger.info("Saved three-panel PNG to %s", png_path)

    if args.mode in ("stage-extruded", "both"):
        logger.info("Generating stage-extruded Matheron ensemble figure")
        fig_stage = plot_matheron_stage_extruded(pos, Z_prior, Z_post, a, b, y_star)

        if args.mode == "stage-extruded":
            save_path = out_dir / f"{stem}{suffix}"
        else:
            save_path = out_dir / f"{stem}_stage{suffix}"

        fig_stage.savefig(save_path, bbox_inches="tight", dpi=450)
        logger.info("Saved stage-extruded figure to %s", save_path)

        png_path = save_path.with_suffix(".png")
        fig_stage.savefig(png_path, bbox_inches="tight", dpi=450)
        logger.info("Saved stage-extruded PNG to %s", png_path)

    if args.show:
        plt.show()
    else:
        plt.close("all")

    logger.info("Matheron plotting complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
