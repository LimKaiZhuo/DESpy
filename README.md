# DESpy

_Optimal Allocation of Energy Storage and Solar Photovoltaic Systems with Residential Demand Scheduling_

The objectives of this project is to determine the optimal energy demand scheduling (DS) and the optimal allocation of
electrical storage devices (ES) and solar photovoltaics (PV), such that it minimizes the total electrical generation cost (TEC)
as well as the peak-to-average ratio (PAR) for a residential population. Three levels of optimization are conducted:

- *Solo Optimization* at the individual customer level using convex programming (CVXPY library and GUROBI solver);

- *Base Optimization* at the residential grid level, in which for each iteration each customer sequentially determines his
energy consumption vector (ECV) based on the exogenous demand (from other consumers) at prior iterations (Game Theory); and

- *GA Optimization* where the allocation of ESDs amongst the population is determined using genetic algorithms (DEAP library).
