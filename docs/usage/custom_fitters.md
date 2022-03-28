## Custom fitters
PyMSM allows configuration of custom event-specific-fitters.  
See EventSpecificFitter class which is an abstract class which specifies the API which needs to be implemented.  

Some custom fitters are available off-the-shelf such as Survival trees [Ishwaran 2008]  


## Survival trees  
An example of using Survival Trees as custom event-specific-fitters in a Multistate model:  

```python hl_lines="2 8"
from pymsm.multi_state_competing_risks_model import MultiStateModel
from pymsm.survival_tree_fitter import SurvivalTreeWrapper

multi_state_model = MultiStateModel(
    dataset, 
    terminal_states, 
    default_update_covariates_function,
    event_specific_fitter=SurvivalTreeWrapper
    )
```
