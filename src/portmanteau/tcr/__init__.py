"""T-cell receptor vocabulary: gene names, HLA alleles and antigen peptides.

The three modules here answer the question "are these two records talking about the same thing?"
for the three axes a TCR record is identified on. They are deliberately separate from
`portmanteau.sequence`, which works on the CDR3 string itself and knows nothing about naming.

- **`genes`** - V and J gene calls to allele-free IMGT names, and to families.
- **`hla`** - allele names to a single standard spelling, with field counts and MHC class.
- **`peptides`** - grouping epitope records that are the same antigen written differently.

`portmanteau.tcr.publicdb` reads the public TCR-antigen databases and leans on all three.

Needs the `tcr` extra (tidytcells).
"""
