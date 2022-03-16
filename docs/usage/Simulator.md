# Simulator

PyMSM can be used as a simulator\sampler for saved models  

See COVID hospitalization example  
  
We can save a model for later use, and then configure a simulator to generate simulated paths
```python
from pymsm.simulation import MultiStateSimulator

mssim = MultiStateSimulator(
    competing_risks_models_list,
    terminal_states=[5, 6],
    update_covariates_fn=covid_update_covariates_function,
    covariate_names=covariate_cols,
)
```  
  
And now we can sample paths from this simulator  
```python
# Run MC for a sample single patient
sim_paths = multi_state_model.run_monte_carlo_simulation(
    sample_covariates=pd.Series({"is_male":0, "age":75, "was_severe": 0}),
    origin_state=3,
    current_time=2,
    n_random_samples=5,
    max_transitions=10,
    print_paths=True,
    n_jobs=-1
)
```
