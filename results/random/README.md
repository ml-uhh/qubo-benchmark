| solver        |     scores |   median_time |   median_fixed_time |   %_solved |   rel_loss_med_no0 |   rel_loss_max |
|:--------------|-----------:|--------------:|--------------------:|-----------:|-------------------:|---------------:|
| dwave         | 2288.83    |    55.7289    |        3600         |   0.26087  |         24         |     inf        |
| genosolver    |  143.009   |     0.648464  |           0.672691  |   0.652174 |          0.0189787 |       0.140453 |
| gurobi        |  295.769   |     0.0645044 |           0.0645044 |   1        |          0         |       0        |
| gurobi_10s    |    8.05758 |     0.0645852 |           0.0645852 |   1        |          0         |       0        |
| gurobi_1s     |    1       |     0.0644872 |           0.0644872 |   1        |          0         |       0        |
| gurobi_dwtime |   39.8517  |     0.0817502 |           0.0817502 |   1        |          0         |       0        |

The following table shows results without calculating problems on which dwave did not find an emedding.

| solver        |     scores |   median_time |   median_fixed_time |   %_solved |   rel_loss_med_no0 |   rel_loss_max |
|:--------------|-----------:|--------------:|--------------------:|-----------:|-------------------:|---------------:|
| dwave         | 2041.61    |    36.8988    |        3600         |        0.3 |         21         |    516         |
| genosolver    |  109.66    |     0.638954  |           0.664436  |        0.7 |          0.0157122 |      0.0302847 |
| gurobi        |  259.664   |     0.0390663 |           0.0390663 |        1   |          0         |      0         |
| gurobi_10s    |    7.97025 |     0.0367451 |           0.0367451 |        1   |          0         |      0         |
| gurobi_1s     |    1       |     0.0415363 |           0.0415363 |        1   |          0         |      0         |
| gurobi_dwtime |   31.4862  |     0.0538851 |           0.0538851 |        1   |          0         |      0         |