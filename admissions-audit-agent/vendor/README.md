# vendor/

Place the internal `au_grouper.py` module here before building the image. It is not part of this
repository and is excluded from version control (see `.gitignore`).

The adapter in `src/audit_agent/grouper/au_drg.py` expects this interface:

| Name | Used as |
|---|---|
| `Groupy(lorica_stack_name=...)` | grouper service client, created once at start-up |
| `get_metadata(path)` | returns `(drg_df, drg_desc)` from the DRG reference data in the model artifact's `drg/` directory |
| `drg_grouper_test(row, groupy, use_alt_diag=True, new_diag_col="new_diag_poa")` | returns a Series with `regroup_drg`, `cost_weight`, `original_cost_weight` |
| `calculate_possible_recovery(df, drg_df, drg_desc)` | adds `drg_rw`, `regroup_drg_value`, `possible_recovery`, `drg_desc`, `regroup_drg_desc` |

The `groupy` package it imports is installed from `requirements/grouper.txt`.

To run without a grouper (verdicts and proposed corrections only), build with
`--build-arg REQUIRE_GROUPER=false` and deploy with `AUDIT_GROUPER_BACKEND=disabled`.
