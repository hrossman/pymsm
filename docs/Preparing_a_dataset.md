## Preparing a dataset for multistate modeling with PyMSM  

The first step of any multistate model is to provide the sample data of paths and covariates.  

There are 2 types of dataset formats which can serve as an input:

1) a list of `PathObject`s  
2) a pandas data frame in the format used to fit the `CompetingRiskModel` class  

## 1. A list of `PathObject`s
Best to see an example:

```py
# Load Rotterdam example data
from pymsm.datasets import prep_rotterdam
dataset, _ = prep_rotterdam()

# Print types
print('dataset type: {}'.format(type(dataset)))
print('elements type: {}'.format(type(dataset[0])))
```

The dataset is a list of elements from class PathObject. Each PathObject in the list corresponds to a single sample’s (i.e “patient’s”) observed path.

Let’s look at one such object in detail:

```py
# Display paths and covariates of one sample (#1314)
sample_path = dataset[1314]
sample_path.print_path()
```

```
dataset type: <class 'list'>  
elemnets type: <class 'pymsm.multi_state_competing_risks_model.PathObject'>  
Sample id: 1326  
States: [1, 2, 3]  
Transition times: [873.999987, 1672.0000989999999]  
Covariates:  
year      1990  
age         44  
meno         0  
grade        3  
nodes       17  
pgr         40  
er           7  
hormon       0  
chemo        1  
Name: 1314, dtype: object  
```


## 2. A pandas dataframe

a pandas data frame in the format used to fit the `CompetingRiskModel` class. Let's see one:

```py
# Load EBMT dataset in long format
from pymsm.datasets import prep_ebmt_long
competing_risk_dataset, covariate_cols, state_labels = prep_ebmt_long()
competing_risk_dataset.head()
```

The `competing_risk_dataset` has to include the following columns:  
```
'sample_id',
'origin_state',
'target_state',
'time_entry_to_origin',
'time_transition_to_target'  
```

which are self-explanatory, as well as any other covariate columns.
