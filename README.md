SARCOS
======

Machine learning methods applied to the [SARCOS dataset](http://www.gaussianprocess.org/gpml/data/), treated as a multivariate regression problem.

**Note: As discovered by [@rajshah4](https://github.com/rajshah4), there is a large amount of leakage between the training and test sets. This code and results are currently left as-is for reference, but should not be considered representative of results with a proper training/test split.**

| Method                           | MSE      | # Params    | # Avg. Path Params |
| -------------------------------- | -------- |-------------|--------------------|
| Linear regression                | 10.69263 | 154         | N/A                |
| Decision tree                    | 3.70763  | 319,591     | 24.6               |
| Neural network (1 hidden layer)  | 2.83472  | 7,431       | N/A                |
| Neural network (5 hidden layers) | 2.65670  | 270,599     | N/A                |
| Random forest                    | 2.39401  | 141,540,436 | 16,771.0           |
| Neural network (3 hidden layers) | 2.12862  | 139,015     | N/A                |
| Gradient boosted trees           | 1.44412  | 988,256     | 6,807.7            |
