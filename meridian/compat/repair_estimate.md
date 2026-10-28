# Compatibility matrix: repair_estimate

| revision | release | first seen | effective | changes | contract | notice | behaviour |
|---|---|---|---|---|---|---|---|
| v7 | v7 | 2026-06-01 | 2026-06-01 | none | breaking: does not promise notes as a required string; states no revocation horizon | onboarding | - |
| v7.1 | v7 | 2026-10-10 | 2026-10-17 | term changed revocation_seconds (None to 90): safe | breaking: does not promise notes as a required string | 7 days | - |
| v7.2 | v7 | 2026-10-12 | 2026-10-19 | output added notes: safe | compatible | 7 days | - |
| v8 | v7 | 2026-10-05 | 2026-11-02 | input added vin (required): breaking | breaking: requires vin, which this seam does not send | 28 days | - |
| v7.3 | v7 | 2026-12-07 | 2026-12-21 | input removed vin: safe; term changed key_seconds (None to 3600): safe | compatible | 14 days | - |
| v7.4 | v7 | 2027-01-04 | 2027-01-18 | none | compatible | 14 days | in band, bv:3ec17ceb844a |
