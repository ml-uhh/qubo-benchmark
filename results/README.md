| solver        |   scores |   median_time |   median_fixed_time |   %_solved |   rel_loss_med_no0 |   rel_loss_max |
|:--------------|---------:|--------------:|--------------------:|-----------:|-------------------:|---------------:|
| dwave         | 37.8439  |    2.66597    |          2.66597    |   0.703422 |         0.00780107 |    inf         |
| genosolver    | 48.6047  |    0.637172   |          0.665848   |   0.640684 |         0.00249872 |      0.292341  |
| gurobi        |  1.72593 |    0.00419545 |          0.00419545 |   1        |         0          |      0         |
| gurobi_10s    |  1       |    0.00462675 |          0.00462675 |   0.980989 |         0.00859915 |      0.0194489 |
| gurobi_1s     |  1.16911 |    0.00566649 |          0.00566649 |   0.973384 |         0.0361583  |      0.0722535 |
| gurobi_dwtime |  1.36289 |    0.0241255  |          0.0241255  |   0.984791 |         0.004402   |      0.0165828 |

The following table shows results without calculating problems on which dwave did not find an emedding.

| solver        |   scores |   median_time |   median_fixed_time |   %_solved |   rel_loss_med_no0 |   rel_loss_max |
|:--------------|---------:|--------------:|--------------------:|-----------:|-------------------:|---------------:|
| dwave         | 23.4232  |    2.59682    |          2.59682    |   0.772443 |         0.00477022 |    516         |
| genosolver    | 31.9389  |    0.570529   |          0.594134   |   0.701461 |         0.00382848 |      0.292341  |
| gurobi        |  1.27816 |    0.00371814 |          0.00371814 |   1        |         0          |      0         |
| gurobi_10s    |  1       |    0.0039866  |          0.0039866  |   0.979123 |         0.00859915 |      0.0194489 |
| gurobi_1s     |  1.14915 |    0.00465155 |          0.00465155 |   0.97286  |         0.0390561  |      0.0722535 |
| gurobi_dwtime |  1.13235 |    0.0235181  |          0.0235181  |   0.983299 |         0.004402   |      0.0165828 |