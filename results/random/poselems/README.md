| solver        |       scores |   median_time |   median_fixed_time |   %_solved |   rel_loss_med_no0 |   rel_loss_max |
|:--------------|-------------:|--------------:|--------------------:|-----------:|-------------------:|---------------:|
| dwave         | 152918       |    35.2711    |        3600         |          0 |            29.8755 |            inf |
| genosolver    |     27.4739  |     0.656181  |           0.656181  |          1 |             0      |              0 |
| gurobi        |      1.09268 |     0.0263891 |           0.0263891 |          1 |             0      |              0 |
| gurobi_10s    |      1       |     0.0156469 |           0.0156469 |          1 |             0      |              0 |
| gurobi_1s     |      1.08787 |     0.0221906 |           0.0221906 |          1 |             0      |              0 |
| gurobi_dwtime |      1.66072 |     0.0311682 |           0.0311682 |          1 |             0      |              0 |

The following table shows results without calculating problems on which dwave did not find an emedding.

| solver        |       scores |   median_time |   median_fixed_time |   %_solved |   rel_loss_med_no0 |   rel_loss_max |
|:--------------|-------------:|--------------:|--------------------:|-----------:|-------------------:|---------------:|
| dwave         | 195403       |    34.2712    |        3600         |          0 |            26.9377 |            516 |
| genosolver    |     34.9415  |     0.649143  |           0.649143  |          1 |             0      |              0 |
| gurobi        |      1.13371 |     0.0180019 |           0.0180019 |          1 |             0      |              0 |
| gurobi_10s    |      1       |     0.0126816 |           0.0126816 |          1 |             0      |              0 |
| gurobi_1s     |      1.12692 |     0.01623   |           0.01623   |          1 |             0      |              0 |
| gurobi_dwtime |      1.83342 |     0.0290444 |           0.0290444 |          1 |             0      |              0 |