"""Hamming-1 graphs: thin covers over `hamming_n_graph`, which does any distance.

Every name here keeps the signature it had before that module existed, so existing callers need
no change. The three builders pin `distance=1, exact=True`; at distance 1 exact is a no-op, so
they are the same graphs as before. ⚠️ Edges now also carry `hamming_distance`, which is 1.

The plotter and the connected-component extractor were never Hamming-1 specific and now live in
`hamming_n_graph` as `plot_seq_graph` and `extract_connected_components`; both are re-exported
here, `plot_seq_graph` under its old name.
"""
from .hamming_n_graph import (  # noqa: F401
    build_and_plot_seq_hamming_graph,
    build_seq_hamming_graph,
    build_seq_hamming_graph_and_extract_ccs,
    extract_connected_components,
    plot_seq_graph,
    prepend_other_column_to_seq,
)

#: The plotter's old name. It draws a graph; nothing about it is Hamming-1 specific.
plot_seq_ham1_graph = plot_seq_graph


def build_seq_ham1_graph(pdf_tcrs, *args, **kwargs):
    """`build_seq_hamming_graph` at distance 1. See it for the arguments."""
    return build_seq_hamming_graph(pdf_tcrs, *args, distance=1, exact=True, **kwargs)


def build_seq_ham1_graph_and_extract_ccs(pdf_tcrs, *args, **kwargs):
    """`build_seq_hamming_graph_and_extract_ccs` at distance 1. See it for the arguments."""
    return build_seq_hamming_graph_and_extract_ccs(
        pdf_tcrs, *args, distance=1, exact=True, **kwargs
    )


def build_and_plot_seq_ham1_graph(pdf_tcrs, *args, **kwargs):
    """`build_and_plot_seq_hamming_graph` at distance 1. See it for the arguments."""
    return build_and_plot_seq_hamming_graph(pdf_tcrs, *args, distance=1, exact=True, **kwargs)
