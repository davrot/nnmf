Refinement of the approach for deep NNMF networks shown in:

```
Competitive performance and superior noise robustness of a non-negative deep convolutional spiking network
David Rotermund, Alberto Garcia-Ortiz, Kaus R. Pawelzik
https://www.biorxiv.org/content/10.1101/2023.04.22.537923v1
```

Now a normal ADAM optimiser will work.  

The BP learning rule is taken from here (it was derived for a spike-based SbS system, but it works exactly the same for NNMF): 

```
Back-Propagation Learning in Deep Spike-By-Spike Networks
David Rotermund and Klaus R. Pawelzik
https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2019.00055/full
```

