# Results

## Initial Test Result

Dataset Shape: X: (7500, 784), y: (7500,)
Running 20 rounds of experiments...

### AVERAGE RESULTS ACROSS ALL ROUNDS

Mean Metrics:
             sklearn     numpy       mlx   pytorch
ARI         0.370824  0.368586  0.377660  0.376219
NMI         0.487760  0.489389  0.493728  0.495406
Silhouette  0.059053  0.060935  0.063618  0.063425
Time (s)    0.163333  2.777906  1.102932  0.605396

Standard Deviation:
             sklearn     numpy       mlx   pytorch
ARI         0.028960  0.023885  0.021521  0.023364
NMI         0.017778  0.014492  0.011171  0.013058
Silhouette  0.009398  0.008716  0.007861  0.008616
Time (s)    0.086759  1.325928  0.825973  0.305303
