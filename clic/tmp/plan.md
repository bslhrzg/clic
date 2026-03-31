So the idea here is to replace the following call in this directory : 

model = Model.from_hybridization..
solver_settings = SolverParameters(...)
solver = FockSpaceSolver ...
result = solver.solve(..)
..
gf_config = GreenFunctionConfig(..)
..

First let's forget about the green functions. 
We want a single structure containing ALL relevant parameters, GLOBALLY defined and accessible 

We want a simpler object for the FockSpace thingy, ie a structure, not an object with methods

We want a fully functional oriented code for the hybridization fit 
we can focus on cost minimization for now 

----

ok so i have clicvars
i have refactored hybfit for cost only diag only 
now i need to get rid of the results class
solver_new for one Nelec, return a simple dict 
i need to replace thermal_gs by a simple dict as well 
