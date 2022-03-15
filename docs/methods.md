# Methods

!!! error "Under construction"

  
`PyMSM` fits and predicts based on a multistate model supplied by the user.   
Some examples of multistate models:  

1) A simple competing-risk form with three states, $A,B,C$ and two possible transitions, $A \rightarrow B$ and $A \rightarrow C$ .  

``` mermaid
stateDiagram-v2
    A --> B
    A --> C
```

2) An illness-death model with the same three states but three possible transitions $A \rightarrow B$, $A \rightarrow C$ and $B \rightarrow C$ . 

``` mermaid
stateDiagram-v2
    A --> B
    A --> C
    B --> C
```

3) Or any other more involved multistate framework (add figures of these models). 


The description of the content of `PyMSM` would be easier to digest under a certain setting.  Thus, to set the stage, we adopt the multistate model of [Roimi et. al. (2021)](https://academic.oup.com/jamia/article/28/6/1188/6105188). Specifically, assume a multistate model consists of four states $A,B,C,D$ and six possible transitions:

$$
A \rightarrow B \,\,\,\,\,\,       A \rightarrow C   \,\,\,\,\,\,     A \rightarrow D   \,\,\,\,\,\,    B \rightarrow A \,\,\,\,\,\,    B \rightarrow D \,\,\,\,\,\,   C \rightarrow A \, .
$$  

``` mermaid
stateDiagram-v2
    A --> B
    A --> C
    A --> D
    B --> A
    B --> D
    C --> A
```


Each transition is characterizes by a transition-specific hazard function, also known as a cause-specific hazard function,

$$
	\lambda_{A,B} (t|Z) \,\,\, \lambda_{A,C} (t|Z) \,\,\, 	\lambda_{A,D} (t|Z) \,\,\, \lambda_{B,A} (t|Z)  \,\,\, \lambda_{B,D} (t|Z) \,\,\,  \lambda_{C,A} (t|Z) \,
$$  

for $t > 0$ and $Z$ vector of covariates. Although $Z$ is shared by the six models above,  it does not imply that identical covariates must be used in these models. For example, in Cox models with   transition-dependent   regression coefficient vectors,  one can set any specific coefficient to 0 for excluding  the corresponding covariate.  

 Let $J_C$ and $J_N$ denote the current and next states, respectively, and $T$ denotes the transition time. Assume the journey of an observation in the system described by the multistate model starts at state $j^*$ with vector of baseline covariates $W$. Let $Z(t)$ be a time-dependent vector of covariates, where  

$$
Z(t)=(W^T,\widetilde{W}(t))
$$

and $\widetilde{W}(t)$ is a time-dependent vector of covariates known at the entrance to the new state. Let $K_{j^*}$ be the set of possible states that can be reached directly from state $j^*$. Then, the conditional probability of transition $j^* \rightarrow j$, $j \in K_{j^*}$, by time $t$ given $Z(0)=Z$ is given by  

$$
\Pr(T \leq t, J_N=j|J_C=j^*,Z(0)=Z) = \int_0^t \lambda_{j^*,j}(u|Z)\exp\left\{-\sum_{k=1}^{|K_{j^*}|} \Lambda_{j^*,k}(u-|Z) \right\}du
$$  
