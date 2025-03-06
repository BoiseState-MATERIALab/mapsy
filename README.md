# Surface Topology Maps


## Goal Applications
1. Automatically identify adsorption sites to perform DFT simulations
2. Predict adsorption energies at different sites using available DFT simulations
3. Potentially, iterate until convergence (geometry optimization or transition states)
4. Use simulations on one substrate-adsorbate configuration for coverage effects (multiple adsorbates)
5. Use simulations on one substrate-adsorbate configuration for other adsorbates (catalytic pathways)

## 1. Automatic Identification of Adsorption Sites
Algorithm:
1. Generate the Contact Space (CS, aka grid-data)
2. Compute descriptors for CS
3. Feature selection (let's start with PCA, we may want to figure out what works the best)
4. Select N points (clustering or select points farthest apart?)
Validation 
1. How far apart are the energies of these points? For different hyperparameters and different values of N
2. Is there a correlation between the energy and the distance in descriptor space.
I would try to run the validation tests on grid-data energies from interpolation of filtered-data. This allows to do faster tests and test more options. Once we have better ideas, we should test the same validation but running DFT calculations on the selected points.

### Auxiliary Goals
1.1. Decide which features are most efficient in selecting inequivalent adsorption sites. Unsupervised or supervised. If we have too many features of the same type, we may overlook differences in adsorption sites according to features that are a minority, but important. We can decrease the number of features in an unsupervised fashion, maybe? 
1.2. Which algorithm (PCA)? Which hyperparameters (number of PCA components)?

## 2. Predict Adsorption Energies Using Available DFT Simulations
Algorithm: N of available simulations (in a hypotetical simulated-data file)
1. Generate the Contact Space (CS, aka grid-data)
2. Generate descriptors for CS (this can in principle be different from 1.)
3. Feature selection (NF smaller than N of available simulations)
4. Import the Simulated Space (SS, aka simulated-data)
5. Generate descriptors for SS 
6. Feature selection for SS
7. Regression or non-linear regression with error-bar on SS
8. Prediction on CS
9. From predictions, identify 1) regions of lower energy 2) regions of high inaccuracy
Validation:
1. in 8. we do prediction on the benchmark-data RMSE on Energy
2. in principles we can repeat this analysis for any N data points from the initial 100 points of the benchmark data
3. ideally we want to do the analysis on N points generated with algorithm 1. 

### Auxiliary Goals
2.1. If we have many DFT simulations, we may use this to create smooth-looking maps of the energy as a function of space. This may help reduce noise due to non-fully converged relax calculations (fancy interpolation). We may use this to assess the accuracy of the other algorithms. (ALMOST DONE, MISSING ERROR BARS)
2.2. Which regression algorithm works better? Gaussian processes (decision forests?). Which hyperparameters? 

There may still be a purpose for supervised learning: 
Prupose 1: if we can have non-element specific features (e.g. fukui functions, electronic density, etc.). This would require 1-3 DFT calculations on the substrate, but it is still ok. 
1.1.1. Are these features consistent across different substrates? Supervised. It would be better if the features are not dependent on the elements of the substrate (they can depend on the adsorbate). We could try using electrons-based or DFT-based descriptors.
1.1.2. Are these features connected with the type of adsorbate? Supervised. It would be better if the features are not dependent on the elements of the substrate (they can depend on the adsorbate). We could try using electrons-based or DFT-based descriptors.
Purpose 2: if we have data for 1 adsorbate and we want to run simulations for N adsorbates or for a slightly different adsorbate, there may still be a use for supervised learning.