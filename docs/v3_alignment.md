# V3 cross-dataset feature alignment

UNSW-NB15 has 47 features; CIC-IDS-2017 (our schema) has 18 scalars + 3 histograms per direction.

Alignment:
- 18 scalars: take UNSW-NB15's closest analog columns (see src/data/unsw_nb15.py:REQUIRED_COLS)
- 3 histograms: UNSW-NB15 has no proto/service histograms in the same form.
  For V3, we substitute a single "proto" histogram mapped from UNSW-NB15's `proto` column.
- The remaining 86 columns of F_entity=104 are zero-padded.

This is documented as a *known limitation* of V3 in the model card.
