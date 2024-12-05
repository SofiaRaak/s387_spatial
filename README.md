# Dynamic model of NMDAR-dependent Ago2 phosphorylation

Computational model to investigate dynamics of Ago2 S387 phosphorylation in dendritic spines

## Model background

Parameters were obtained through fitting to experimental data in [Rajgor _et al._ 2018](https://www.embopress.org/doi/full/10.15252/embj.201797943)

## Model structure

The model was designed as a system of ordinary differential equations:

```math
\frac{d[R]}{dt} = k_{-1}*[pR] - NMDA*k_1*[R]\frac{d[pR]}{dt} = NMDA*k_1*[R] - k_{-1}*[pR]
```

```math
\frac{d[A]}{{dt}} = k_{-2}*[pA] - k_2*[A]*[pR]
```

```math
\frac{d[pA]}{dt} = (k_2*[A]*[pR] + k_{-3}*[pAG]) - (k_{-2}*[pA] + k_3*[pA]*[G])
```

```math
\frac{d[G]}{dt} = k_{-3}*[pAG] -  k_3*[pA]*[G]
```

```math
\frac{d[pAG]}{dt} = k_3*[pA]*[G] - k_{-3}*[pAG]
```

where 

* $NMDA$ denotes the relative strength of an NMDA-mediated signal in the synapse
* $[R]$ is the concentration of inactive Akt1
* $[pR]$ is the concentration of phosphorylated, active Akt1
* $[A]$ is the concentration of unphosphorylated Ago2
* $[pA]$ is the concentration of phosphorylated Ago2
* $[G]$ is the concentration of free GW182
* $[pAG]$ is the concentration of pAgo2:GW182 complex
* $k_1$ is the phosphorylation rate of Akt1
* $k_{-1}$ is the dephosphorylation rate of Akt1
* $k_2$ is the phosphorylation rate of Ago2
* $k_{-2} is the dephosphorylation rate of Ago2
* $k_3$ is the rate of pAgo2:GW182 complex formation
* $k_{-3}$ is the rate of pAgo2:complex dissociation
