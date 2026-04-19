============================================================
📊  BENCHMARK SUMMARY
============================================================

🏁  SPEED RANKING — ALL (tok/s)
  #    name                                     tok/s
  ---- ----------------------------------- ----------
  1    gen5                                     82.69
  2    gen7                                     69.99
  3    gen5_non_early_stop                      65.96
  4    gen7_non_early_stop                      39.21
  5    gen1                                     33.58
  6    gen3                                     29.09

📏  LENGTH RANKING — ALL (tokens)
  #    name                                    tokens
  ---- ----------------------------------- ----------
  1    gen1                                     86.19
  2    gen3                                     74.23
  3    gen5_non_early_stop                      66.85
  4    gen5                                     52.15
  5    gen7                                     27.46
  6    gen7_non_early_stop                      27.46

============================================================
📋  GROUP AVERAGES
============================================================

⚡  Avg Speed by group (tok/s)
  group                             tok/s  n
  ---------------------------- ----------  -
  normal                            53.84  (4)
  non_early_stop                    52.80  (4)
  paged                             34.28  (4)
  non_early_stop_paged              19.58  (4)

📐  Avg Length by group (tokens)
  group                            tokens  n
  ---------------------------- ----------  -
  normal                            60.01  (4)
  non_early_stop                    47.15  (4)
  non_early_stop_paged              47.15  (4)
  paged                             39.81  (4)

============================================================
🔬  SUB-DIMENSION COMPARISONS
============================================================

⚡  Speed: early_stop vs non_early_stop
  dimension                           tok/s  n
  ------------------------------ ----------  -
  early_stop                          44.06  8
  non_early_stop                      36.19  8

📐  Length: early_stop vs non_early_stop
  dimension                          tokens  n
  ------------------------------ ----------  -
  early_stop                          49.91  8
  non_early_stop                      47.15  8

⚡  Speed: paged vs non_paged
  dimension                           tok/s  n
  ------------------------------ ----------  -
  non_paged                           53.32  8
  paged                               26.93  8

📐  Length: paged vs non_paged
  dimension                          tokens  n
  ------------------------------ ----------  -
  non_paged                           53.58  8
  paged                               43.48  8

⚡  Speed: no_cache vs kv_cache
  dimension                           tok/s  n
  ------------------------------ ----------  -
  kv_cache gen5-8                     45.71  8
  no_cache gen1-4                     34.53  8

📐  Length: no_cache vs kv_cache
  dimension                          tokens  n
  ------------------------------ ----------  -
  no_cache gen1-4                     53.58  8
  kv_cache gen5-8                     43.48  8

⚡  Speed: score_fn1 vs score_fn2
  dimension                           tok/s  n
  ------------------------------ ----------  -
  score_fn1 gen1,2,5,6                45.57  8
  score_fn2 gen3,4,7,8                34.67  8

📐  Length: score_fn1 vs score_fn2
  dimension                          tokens  n
  ------------------------------ ----------  -
  score_fn1 gen1,2,5,6                63.75  8
  score_fn2 gen3,4,7,8                33.31  8
