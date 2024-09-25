| solver        |     scores |   median_time |   median_fixed_time |   %_solved |   rel_loss_med_no0 |   rel_loss_max |
|:--------------|-----------:|--------------:|--------------------:|-----------:|-------------------:|---------------:|
| dwave         |  658.355   |    196.041    |          3600       |      0.375 |          0.0127898 |     inf        |
| genosolver    |  798.264   |      0.656266 |          3600       |      0.25  |          0.0157122 |       0.140453 |
| gurobi        | 3482.3     |   3615.94     |          3600       |      1     |          0         |       0        |
| gurobi_10s    |    9.75917 |     10.0665   |            10.0665  |      1     |          0         |       0        |
| gurobi_1s     |    1       |      1.01926  |             1.01926 |      1     |          0         |       0        |
| gurobi_dwtime |  117.094   |    196.721    |           196.721   |      1     |          0         |       0        |

The following table shows results without calculating problems on which dwave did not find an emedding.

| solver        |     scores |   median_time |   median_fixed_time |   %_solved |   rel_loss_med_no0 |   rel_loss_max |
|:--------------|-----------:|--------------:|--------------------:|-----------:|-------------------:|---------------:|
| dwave         |  372.897   |     78.7965   |          1838.66    |   0.5      |         0.00981997 |      0.0127898 |
| genosolver    |  483.28    |      0.535263 |          3600       |   0.333333 |         0.0107207  |      0.0166945 |
| gurobi        | 3460.49    |   3610.48     |          3600       |   1        |         0          |      0         |
| gurobi_10s    |    9.70528 |     10.0913   |            10.0913  |   1        |         0          |      0         |
| gurobi_1s     |    1       |      1.04131  |             1.04131 |   1        |         0          |      0         |
| gurobi_dwtime |   83.243   |     79.1093   |            79.1093  |   1        |         0          |      0         |