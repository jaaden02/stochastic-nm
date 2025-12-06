# stochastic-nm


## Methods for PDE

1. Crank Nicolson -- Finite Differences  Filip 
2. Monte Carlo Jasper
3. Chang-Cooper Jasper
4. Spectral Method Pietro


### Notes on what seems important to them:

- justifying the choices we make i.e. step size, forward vs backward differences
- should we do adaptive mesh refinement?
- we need to explain the dimensions of the plot and what the numbers mean

# Slides

  1) ### Determistic Article:
     
     Why this model
     
     Explaining equations
     
  3) ### Plots of deterministic w different starting conditions:
     
     brief, also numerical methods here !eliminating some paths cause they go to zero
  
  5) Adding noise:
   
     explain why a stochastic, and why only on phi

     how to add this to the original equations

     and numerical
 
  7) Show average in manypaths stochastic

     2 different D

     choose aboute the final D, physical meaning and plankton layer
  
  9) FP theory
  
     from SDEs to FP
  
  10) Our langevin to FP
  
     splitting the differential operator advection + diffusion
  
  11) Crank - Nicolson
  
     just show the generic scheme, briefly proprietes of CN and why we use it
  
  12) Chang Cooper
  
     same but more theory to explain it
     
     cite the article
  
  13) Comparing all 4 results
  
      we already said that SDE and FP should be the same

      deterministic + langevin + CN * CC
  
  15) Conclusions
   
      comparing the three different methods

      use langevin lol

      if we can conclusion on the physics
  
  11... Appendix
