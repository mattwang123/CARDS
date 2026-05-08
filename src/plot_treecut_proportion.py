"""
TreeCut variant of proportion plotting pipeline.

This reuses src/plot_proportion.py end-to-end, but switches DATASET to 'treecut'.
All CLI flags from plot_proportion.py are supported (cache controls, viz-only, etc.).
"""

import plot_proportion as pp


def main():
    # Reuse exactly the same plotting logic with TreeCut dataset.
    pp.DATASET = "treecut"
    args = pp.parse_plot_args()
    pp.run_relative_scatter(args)


if __name__ == "__main__":
    main()

