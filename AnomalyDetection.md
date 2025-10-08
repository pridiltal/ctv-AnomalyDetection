---
name: AnomalyDetection
topic: Anomaly Detection
maintainer: Priyanga Dilini Talagala, Rob J. Hyndman
email: pritalagala@gmail.com
version: 2025-10-07
source: https://github.com/pridiltal/ctv-AnomalyDetection
---

This CRAN Task View provides a comprehensive list of R packages for anomaly detection. Anomaly detection problems have many different facets, and detection techniques are influenced by factors such as how anomalies are defined, the type of input data, and the expected output. These variations lead to diverse problem formulations, requiring different analytical approaches. This Task View aims to help users navigate the available tools by organizing them based on their applicability to different data types and detection methodologies.

Anomalies are often referred to by alternative names such as outliers, novelties, odd values, extreme values, faults, and aberrations, depending on the application domain. This Task View considers all these variations and categorizes relevant R packages accordingly. The overview covers methods applicable to univariate, multivariate, spatial, temporal, and functional data, ensuring users can identify suitable tools for various analytical needs. R packages that do not primarily focus on anomaly detection but offer substantial functionalities for anomaly detection have also been included.

Packages where anomaly detection is only a minor feature with very limited functions have been excluded. Additionally, tools that are outdated, redundant, or lack sufficient support have not been considered.

To facilitate navigation, the Task View is structured into well-defined sections, including Univariate Outlier Detection, Multivariate Detection (further categorized into density-based, distance-based, clustering-based, angle-based, and decision tree-based methods), Temporal Data, Spatial and Spatio-Temporal Data, Functional Data, and other specialized approaches.

### Univariate Outlier Detection

- Univariate outlier detection methods focus on values in a single feature space. Package `r pkg("univOutl")` includes various methods for detecting univariate outliers, e.g., the Hidiroglou-Berthelot method. Methods to deal with skewed distribution are also included in this package.
- The `r pkg("dixonTest")` package provides Dixon's ratio test for outlier detection in small and normally distributed samples.
- The `r pkg("hotspots")` package supports univariate outlier detection by identifying values that are disproportionately high based on both the deviance of any given value from a statistical distribution and its similarity to other values.
- The `r pkg("outliers")` package provides a collection of tests commonly used for identifying outliers. For most functions the input is a numeric vector. If the argument is a data frame, then the outlier is calculated for each column by `sapply()`. The same behavior is applied by `apply()` when the matrix is given.
- The `r pkg("extremevalues")` package offers outlier detection and plot functions for univariate data. In this work a value in the data is an outlier when it is unlikely to be drawn from the estimated distribution.
- The `r pkg("funModeling")` package provides tools for outlier detection using top/bottom X%, Tukey’s boxplot definition, and Hampel’s method.
- The `r pkg("alphaOutlier")` package provides Alpha-Outlier regions (as proposed by Davies and Gather (1993)) for well-known probability distributions.

### Multivariate Outlier Detection

Under *a multivariate, high-dimensional, or multidimensional scenario,* where the focus is on n (\>2)-dimensional space, all attributes might be of the same type or might be a mixture of different types, such as categorical or numerical, which has a direct impact on the implementation and scope of the algorithm. The problems of anomaly detection in high-dimensional data are threefold, involving detection of (a) global anomalies, (b) local anomalies, and (c) microclusters or clusters of anomalies. Global anomalies are very different from the dense area with respect to their attributes. In contrast, a local anomaly is only an anomaly when it is distinct from, and compared with, its local neighbourhood. Microclusters, or clusters of anomalies, may cause masking problems. The following categorization of multivariate outlier detection techniques is based on the underlying methodological principle, including density-based outlier detection, distance-based outlier detection, clustering-based outlier detection, angle-based outlier detection, and decision tree–based approaches.

#### Multivariate Outlier Detection: Density-based outlier detection

- *Local Outlier Factor (LOF)* is an algorithm for detecting anomalous data points by measuring the local deviation of a given data point with respect to its neighbours. This algorithm, with some variations, is supported by many packages. The `r pkg("DescTools")` package provides functions for outlier detection using LOF and Tukey’s boxplot definition. Functions `LOF()` and `GLOSH` in package `r pkg("dbscan")` provide density-based anomaly detection methods using a kd-tree to speed up kNN search. Parallel implementation of LOF, which uses multiple CPUs to significantly speed up the LOF computation for large datasets, is available in the `r pkg("Rlof")` package. Package `r pkg("bigutilsr")` provides utility functions for outlier detection in large-scale data. It includes the LOF and outlier detection method based on departure from the histogram.
- The `r pkg("SMLoutliers")` package provides an implementation of the Local Correlation Integral method (Lof: Identifying density-based local outliers) for outlier detection in multivariate data, which consists of numeric values.
- The `r pkg("ldbod")` package provides flexible functions for computing local density-based outlier scores. It allows for subsampling of input data or a user-specified reference data set to compute outlier scores against, so both unsupervised and semi-supervised outlier detection can be done.
- The `r pkg("kernlab")` package provides kernel-based machine learning methods, including one-class Support Vector Machines for *novelty* detection.
- The `r pkg("amelie")` package implements anomaly detection as binary classification for multivariate
- The estimated density ratio function in the `r pkg("densratio")` package can be used in many applications such as anomaly detection, change-point detection, and covariate shift adaptation.
- The `r pkg("lookout")` package detects outliers using leave-one-out kernel density estimates and extreme value theory. The bandwidth for kernel density estimates is computed using persistent homology, a technique in topological data analysis. It also has the capability to explore the birth and the cessation of outliers with changing bandwidth and significance levels via `persisting_outliers().`
- The Weighted BACON (blocked adaptive computationally efficient outlier nominators) algorithms in `r pkg("wbacon")` implement a weighted variant of the BACON algorithms for multivariate outlier detection and robust linear regression. The methods assume that the typical data follows an elliptically contoured distribution.

#### Multivariate Outlier Detection: Distance-based outlier detection

- The `r pkg("HDoutliers")` package provides an implementation of an algorithm for univariate and multivariate outlier detection that can handle data with mixed categorical and continuous variables and the outlier masking problem.
- The `r pkg("stray")` package implements an algorithm for detecting anomalies in high-dimensional data that addresses the limitations of the 'HDoutliers' algorithm. An approach based on extreme value theory is used for the anomalous threshold calculation.
- The `r pkg("Routliers")` package provides robust methods to detect univariate (Median Absolute Deviation method) and multivariate outliers (Mahalanobis-Minimum Covariance Determinant method).
- The `r pkg("modi")` package implements Mahalanobis distance or depth-based algorithms for multivariate outlier detection in the presence of missing values (incomplete survey data).
- The `r pkg("CerioliOutlierDetection")` package implements the iterated RMCD method of Cerioli (2010) for multivariate outlier detection via robust Mahalanobis distances.
- The `r pkg("rrcovHD")` package performs outlier identification using robust multivariate methods based on robust Mahalanobis distances and principal component analysis.
- The  `r pkg("mvoutlier")` package also provides various robust methods based on multivariate outlier detection capabilities. This includes a Mahalanobis-type method with an adaptive outlier cutoff value, a method incorporating local neighbourhood, and a method for compositional data.
- The function `dm.mahalanobis()` in the `r pkg("DJL")` package implements the Mahalanobis distance measure for outlier detection. In addition to the basic distance measure, boxplots are provided with potential outlier(s) to give an insight into the early stage of the data cleansing task.
- The `r pkg("mvout")` package detects multivariate outliers using robust Mahalanobis distances based on the Minimum Covariance Determinant (MCD) estimator.
- The `r pkg("outlierMBC")` package implements sequential outlier identification for Gaussian mixture models.  Outliers are detected by comparing observed Mahalanobis distances with the theoretical distribution. It also provides an extension for Gaussian linear cluster-weighted models using studentized residuals. The method emphasizes model-based, distance-driven identification of anomalies.

#### Multivariate Outlier Detection: Clustering-based outlier detection

- The `r pkg("kmodR")` package presents a unified approach for simultaneously clustering and discovering outliers in high-dimensional data. Their approach is formalized as a generalization of the k-MEANS problem.
- The `r pkg("odetector")` package detects multivariate outliers using soft partitioning clustering algorithms such as Fuzzy C-means and its variants. Observations with low typicality degrees are flagged as outliers.


#### Multivariate Outlier Detection: Angle-based outlier detection

- The `r pkg("abodOutlier")` package performs angle-based outlier detection on high-dimensional data. A complete, a randomized, and a KNN-based method are available.

#### Multivariate Outlier Detection: Decision tree based approaches

- Explainable outlier detection method through decision tree conditioning is facilitated by the `r pkg("outliertree")` package.
- The `r pkg("bagged.outliertrees")` package provides an explainable unsupervised outlier detection method based on an ensemble implementation of the existing OutlierTree procedure in the `r pkg("outliertree")` package. The implementation takes advantage of bootstrap aggregating (bagging) to improve robustness by reducing the possible masking effect and subsequent high variance (similarly to Isolation Forest), hence the name "Bagged OutlierTrees".
- The `r pkg("isotree")` package provides fast and multi-threaded implementation of Extended Isolation Forest, Fair-Cut Forest, SCiForest (a.k.a. Split-Criterion iForest), and regular Isolation Forest for isolation-based outlier detection, clustered outlier detection, distance or similarity approximation, and imputation of missing values based on random or guided decision tree splitting. It also supports categorical data.
- The `r pkg("outForest")` package provides a random forest-based implementation for multivariate outlier detection.  In this method each numeric variable is regressed onto all other variables by a random forest. If the scaled absolute difference between the observed value and the out-of-bag prediction of the corresponding random forest is suspiciously large, then a value is considered an outlier.
- The `r pkg("solitude")` package provides an implementation of isolation forest, which detects anomalies in cross-sectional tabular data purely based on the concept of isolation without employing any distance or density measures.
- The `r pkg("bulkQC")` package includes the `ind_multi()` function for detecting multivariate outliers using Isolation Forests.


#### Multivariate Outlier Detection: Other approaches


- The  `r pkg("abnormality")` package measures a subject's abnormality with respect to a reference population. A methodology is introduced to address this bias to accurately measure overall abnormality in high-dimensional spaces. It can be applied to datasets in which the number of observations is less than the number of features/variables, and it can be abstracted to practically any number of domains or dimensions.
- The `r pkg("ICSOutlier")` package performs multivariate outlier detection using invariant coordinates and offers different methods to choose the appropriate components. The current implementation targets data sets with only a small percentage of outliers, but future extensions are under preparation.
- The `r pkg("sGMRFmix")` package provides an anomaly detection method for multivariate noisy sensor data using sparse Gaussian Markov random field mixtures. It can compute variable-wise anomaly scores.
- Artificial neural networks for anomaly detection are implemented in the `r pkg("ANN2")` package.
- The `r pkg("probout")` package estimates unsupervised outlier probabilities for multivariate numeric
- The `r pkg("mrfDepth")` package provides tools to compute depth measures and implementations of related tasks such as outlier detection, data exploration, and classification of multivariate, regression, and functional data.
- The `r pkg("evtclass")` package provides two classifiers for open set recognition and novelty detection based on extreme value theory.
- Cellwise outliers are entries in the data matrix that are substantially higher or lower than what could be expected based on the other cells in its column as well as the other cells in its row, taking the relations between the columns into account. Package `r pkg("cellWise")` provides tools for detecting cellwise outliers and robust methods to analyze data that may contain them.
- The Projection Congruent Subset (PCS) is a method for finding multivariate outliers by searching for a subset that minimizes a criterion. PCS is supported by the `r pkg("FastPCS")` package.
- The  `r pkg("outlierensembles")` package provides ensemble functions for outlier/anomaly detection. In addition to some existing ensemble methods for outlier detection, it also provides an item response theory-based ensemble method.


### Temporal Data

- The problems of anomaly detection for temporal data are 3-fold: (a) the detection of contextual anomalies (point anomalies) within a given series; (b) the detection of anomalous subsequences within a given series; and (c) the detection of anomalous series within a collection of series.
- The `r pkg("trendsegmentR")` package performs the detection of point anomalies and linear trend changes for univariate time series by implementing the bottom-up unbalanced wavelet transformation.
- The `r pkg("anomaly")` package implements Collective And Point Anomaly (CAPA), Multi-Variate Collective And Point Anomaly (MVCAPA), Proportion Adaptive Segment Selection (PASS), and Bayesian Abnormal Region Detector (BARD) methods for the detection of anomalies in time series data.
- The `r pkg("anomalize")` package enables a "tidy" workflow for detecting anomalies in data. The main functions are `time_decompose()`, `anomalize()`, and `time_recompose()`.
- The `detectAO()` and `detectIO()` functions in the `r pkg("TSA")` package support detecting additive outliers and innovative outliers in time series data.
- The `r pkg("washeR")` package performs time series outlier detection using a nonparametric test. An input can be a data frame (grouped time series: phenomenon+date+group+values) or a vector (single time series).
- The `r pkg("tsoutliers")` package implements the Chen-Liu approach for detection of time series outliers such as innovational outliers, additive outliers, level shifts, temporary changes, and seasonal level shifts.
- The `r pkg("seasonal")` package provides an easy-to-use interface to X-13-ARIMA-SEATS, the seasonal adjustment software by the US Census Bureau. It offers full access to almost all options and outputs of X-13, including outlier detection.
- The `r pkg("npphen")` package implements basic and high-level functions for detection of anomalies in vector data (numerical series/time series) and raster data (satellite-derived products). Processing of very large raster files is supported.
- The `r pkg("ACA")` package offers an interactive function for the detection of abrupt change points or aberrations in point series.
- The `r pkg("oddstream")` package implements an algorithm for early detection of anomalous series within a large collection of streaming time series data. The model uses time series features as inputs and a density-based comparison to detect any significant changes in the distribution of the features.
- The `r pkg("pasadr")` package provides a novel stealthy-attack detection mechanism that monitors time series of sensor measurements in real time for structural changes in the process  behaviour. It has the capability of detecting both significant deviations in the process behavior and subtle attack-indicating changes, significantly raising the bar for strategic adversaries who may attempt to maintain their malicious manipulation within the noise level.
- The `r pkg("kfino")` package detects impulse-noise outliers in time series using a Kalman filter. It provides robust sequential filtering and prediction on cleaned data, implementing both Maximum Likelihood (ML) and Expectation-Maximization (EM) algorithms.
- The `r pkg("outliers.ts.oga")` package provides efficient detection and cleaning of outliers in single time series as well as in large databases of homogeneous and heterogeneous time series. It uses the Orthogonal Greedy Algorithm (OGA) for saturated linear regression models, enabling scalable detection. Version 1.1.1 includes improvements in parallelization.


### Spatial Data

- Spatial objects whose non-spatial attribute values are markedly different from those of their spatial neighbors are known as spatial outliers or abnormal spatial patterns.
- Enhanced False Discovery Rate (EFDR) is a tool to detect anomalies in an image. Package `r pkg("EFDR")` implements wavelet-based enhanced FDR for detecting signals from complete or incomplete spatially aggregated data. The package also provides elementary tools to interpolate spatially irregular data onto a grid of the required size.
- The function `spatial.outlier()` in `r pkg("depth.plot")` package helps to identify multivariate spatial outliers within a p-variate data cloud or if any p-variate observation is an outlier with respect to a p-variate data cloud.


### Spatio-Temporal Data

- The `r pkg("CoordinateCleaner")` package provides automated tools to detect and flag common spatial and temporal errors (outliers) in biological and paleontological occurrence data. It identifies problematic coordinates such as country centroids, biodiversity institution locations, or ocean points, and flags species-level outliers and rounding errors, improving data quality for ecological and conservation analyses.

### Functional Data

- The `foutliers()` function from the `r pkg("rainbow")` package provides functional outlier detection methods. Bagplots and boxplots for functional data can also be used to identify outliers, which have either the lowest depth (distance from the centre) or the lowest density, respectively.
- The `r pkg("adamethods")` package provides a collection of several algorithms to obtain archetypoids with small and large databases and with both classical multivariate data and functional data (univariate and multivariate). Some of these algorithms also allow us to detect anomalies.
- The `shape.fd.outliers()` function in the `r pkg("ddalpha")` package detects functional outliers of the first three orders, based on the order extended integrated depth for functional data.
- The `r pkg("fda.usc")` package provides tools for outlier detection in functional data (atypical curve detection) using different approaches such as the likelihood ratio test, depth measures, and quantiles of the bootstrap samples.
- The `r pkg("fdasrvf")` package supports outlier detection in functional data using the square-root velocity framework, which allows for elastic analysis of functional data through phase and amplitude separation.
- The `r pkg("fdaoutlier")` package provides a collection of functions for outlier detection in functional data analysis. Methods implemented include directional outlyingness, MS-plot, total variation depth, and sequential transformations, among others.
- The `r pkg("DeBoinR")` package detects outliers in ensembles of probability density functions using functional boxplots. The `deboinr()` function orders the functions by distance and flags outliers based on a user-defined interquartile range.
- The `r pkg("mrct")` package detects outliers in functional data using the Minimum Regularized Covariance Trace (MRCT) estimator.


### Visualization of Outlier

- The `r pkg("OutliersO3")` package provides tools to aid in the display and understanding of patterns of multivariate outliers. It uses the results of identifying outliers for every possible combination of dataset variables to provide insight into why particular cases are outliers.
- The `r pkg("Morpho")` package provides a collection of tools for geometric morphometrics and mesh processing. Apart from the core functions, it provides a graphical interface to find outliers and/or to switch mislabeled landmarks.
- The `r pkg("StatDA")` package provides visualization tools to locate outliers in environmental data.

### Pre-processing Methods for Outlier Detection

- The `r pkg("dobin")` package provides a dimension reduction technique for outlier detection using neighbours and constructs a set of basis vectors for outlier detection. It brings outliers to the forefront using fewer basis vectors.


### Specific Application Fields

#### Epidemiology

- The `r pkg("ABPS")` package provides an implementation of the Abnormal Blood Profile Score (ABPS, part of the Athlete Biological Passport program of the World Anti-Doping Agency), which combines several blood parameters into a single score in order to detect blood doping. The package also contains functions to calculate other scores used in anti-doping programs, such as the OFF-score.
- The `r pkg("surveillance")` package implements statistical methods for aberration detection in time series of counts, proportions, and categorical data, as well as for the modeling of continuous-time point processes of epidemic phenomena. The package also contains several real-world data sets and the ability to simulate outbreak data and to visualize the results of the monitoring in a temporal, spatial, or spatio-temporal fashion.
- The `r pkg("outbreaker2")` package supports Bayesian reconstruction of disease outbreaks using epidemiological and genetic information. It is applicable to various densely sampled epidemics and improves previous approaches by detecting unobserved and imported cases, as well as allowing multiple introductions of the pathogen.
- The `r pkg("outbreaks")` package provides empirical or simulated disease outbreak data, either as RData or as text files.

#### Environmental Science / Hydrology / Meteorology

- The `r pkg("precintcon")` package contains functions to analyze the precipitation intensity, concentration, and anomaly.
- The  `r pkg("wql")` package, which stands for \`water quality,' provides functions including anomaly detection to assist in the processing and exploration of data from environmental monitoring programs.
- The Grubbs‐Beck test is recommended by the federal guidelines for detection of low outliers in flood flow frequency computation in the United States. The `r pkg("MGBT")` package computes the multiple Grubbs-Beck low-outlier test on positively distributed data and utilities for non-interpretive U.S. Geological Survey annual peak-stream flow data processing.
- The  `r pkg("envoutliers")` package provides three semi-parametric methods for detection of outliers in environmental data based on kernel regression and subsequent analysis of smoothing residuals.
- The  `r pkg("extremeIndex")` computes an index measuring the amount of information brought by forecasts for extreme events, subject to calibration. This index is originally designed for weather or climate forecasts, but it may be used in other forecasting contexts.

#### Biomedical and Clinical Research Applications

- The `r pkg("survBootOutliers")` package provides concordance-based bootstrap methods for outlier detection in survival analysis.
- The `r pkg("referenceIntervals")` package provides a collection of tools, including outlier detection, to allow the medical professional to calculate appropriate reference ranges (intervals) with confidence intervals around the limits for diagnostic purposes.
- The `r pkg("bulkQC")` provides tools for quality control and outlier identification in multicenter randomized trials. It analyzes data from multiple study participants across several sites, detecting outliers at both the individual (univariate and multivariate) and site levels, with or without covariate adjustment. 
- The `r pkg("NMAoutlier")` package implements the forward search algorithm for the detection of outlying studies (studies with extreme results) in network meta-analysis.
- The `r pkg("boutliers")` package provides methods for outlier detection and influence diagnostics for meta-analysis based on bootstrap distributions of the influence statistics.

#### Genetics & Bioinformatics

- The `r pkg("pcadapt")` package provides methods to detect genetic markers involved in biological adaptation using statistical tools based on principal component analysis.
- The `r pkg("GGoutlieR")` package detects and visualizes individuals with unusual geo-genetic patterns using a K-nearest neighbor approach. It identifies outliers that deviate from the isolation-by-distance assumption and provides statistical summaries and geographic visualizations. 
- The `r pkg("MALDIrppa")` package provides methods for quality control and robust preprocessing and analysis of MALDI mass spectrometry data. 
- The `r pkg("qpcR")` package implements methods for kinetic outlier detection (KOD) in real-time polymerase chain reaction  (qPCR).

#### Seismology & Geoscience

- The Hampel filter is a robust outlier detector using Median Absolute Deviation (MAD). The `r pkg("seismicRoll")` package provides fast rolling functions for seismology, including outlier detection with a rolling Hampel filter.

#### Political Science / Election Analysis

- The `r pkg("spikes")` package provides a tool to detect election fraud from irregularities in vote-share distributions using the resampled kernel density method.


### Data Sets

- The `r pkg("anomaly")` package contains light curve time series data from the Kepler telescope.
- The `r pkg("outbreaks")` package provides empirical or simulated disease outbreak data, either as RData or as text files.


### Educational and Companion Resources

- The [OutliersLearn](../packages/OutliersLearn/index.html) package
  provides implementations of some of the most important outlier
  detection algorithms. Includes a tutorial mode option that shows a
  description of each algorithm and provides a step-by-step execution
  explanation of how it identifies outliers from the given data with the
  specified input parameters. The package covers three main types of
  approaches statistical, distance-based, and density/clustering
  approaches.
  
- The [weird](../packages/weird/index.html) package provides functions and
  data sets for Hyndman (2024), *That's Weird: Anomaly Detection Using R* 
  <https://OTexts.com/weird/>. It includes all functions, data sets, and 
  required packages needed to reproduce the examples from the book.

**Miscellaneous**

- The [CircOutlier](../packages/CircOutlier/index.html) package enables
  detection of outliers in circular-circular regression models,
  modifying its and estimating of models parameters.
- The Residual Congruent Subset (RCS) is a method for finding outliers
  in the regression setting. RCS is supported by
  [FastRCS](../packages/FastRCS/index.html) package.
- The [oclust](../packages/oclust/index.html) package provides a
  function to detect and trim outliers in Gaussian mixture model based
  clustering using methods described in Clark and McNicholas (2019).
- The [SeleMix](../packages/SeleMix/index.html) package provides
  functions for detection of outliers and influential errors using a
  latent variable model. A mixture model (Gaussian contamination model)
  based on response(s) y and a depended set of covariates is fit to the
  data to quantify the impact of errors to the estimates.
- The [compositions](../packages/compositions/index.html) package
  provides functions to detect various types of outliers in
  compositional datasets.
- The [kuiper.2samp](../packages/kuiper.2samp/index.html) package
  performs the two-sample Kuiper test to assess the anomaly of
  continuous, one-dimensional probability distributions.
- The `enpls.od()` function in [enpls](../packages/enpls/index.html)
  package performs outlier detection with ensemble partial least
  squares.
- The [faoutlier](../packages/faoutlier/index.html) package provides
  tools for detecting and summarize influential cases that can affect
  exploratory and confirmatory factor analysis models and structural
  equation models.
- The [crseEventStudy](../packages/crseEventStudy/index.html) package
  provides a robust and powerful test of abnormal stock returns in
  long-horizon event studies

</div>

### CRAN packages

|  |  |
|----|----|
| *Core:* | [HDoutliers](https://CRAN.R-project.org/package=HDoutliers), [OutliersO3](https://CRAN.R-project.org/package=OutliersO3), [weird](https://cran.r-project.org/web/packages/weird/index.html) |
| *Regular:* | [abnormality](https://CRAN.R-project.org/package=abnormality), [abodOutlier](https://CRAN.R-project.org/package=abodOutlier), [ABPS](https://CRAN.R-project.org/package=ABPS), [ACA](https://CRAN.R-project.org/package=ACA), [adamethods](https://CRAN.R-project.org/package=adamethods), [alphaOutlier](https://CRAN.R-project.org/package=alphaOutlier), [amelie](https://CRAN.R-project.org/package=amelie), [ANN2](https://CRAN.R-project.org/package=ANN2), [anomalize](https://CRAN.R-project.org/package=anomalize), [anomaly](https://CRAN.R-project.org/package=anomaly), [bagged.outliertrees](https://CRAN.R-project.org/package=bagged.outliertrees), [bigutilsr](https://CRAN.R-project.org/package=bigutilsr), [boutliers](https://CRAN.R-project.org/package=boutliers), [bulkQC](https://CRAN.R-project.org/package=bulkQC), [cellWise](https://CRAN.R-project.org/package=cellWise), [CerioliOutlierDetection](https://CRAN.R-project.org/package=CerioliOutlierDetection), [CircOutlier](https://CRAN.R-project.org/package=CircOutlier), [compositions](https://CRAN.R-project.org/package=compositions), [CoordinateCleaner](https://CRAN.R-project.org/package=CoordinateCleaner), [crseEventStudy](https://CRAN.R-project.org/package=crseEventStudy), [dbscan](https://CRAN.R-project.org/package=dbscan), [ddalpha](https://CRAN.R-project.org/package=ddalpha), [DeBoinR](https://CRAN.R-project.org/package=DeBoinR), [densratio](https://CRAN.R-project.org/package=densratio), [depth.plot](https://CRAN.R-project.org/package=depth.plot), [DescTools](https://CRAN.R-project.org/package=DescTools), [dixonTest](https://CRAN.R-project.org/package=dixonTest), [DJL](https://CRAN.R-project.org/package=DJL), [dobin](https://CRAN.R-project.org/package=dobin), [EFDR](https://CRAN.R-project.org/package=EFDR), [enpls](https://CRAN.R-project.org/package=enpls), [envoutliers](https://CRAN.R-project.org/package=envoutliers), [evtclass](https://CRAN.R-project.org/package=evtclass), [extremeIndex](https://CRAN.R-project.org/package=extremeIndex), [extremevalues](https://CRAN.R-project.org/package=extremevalues), [faoutlier](https://CRAN.R-project.org/package=faoutlier), [FastPCS](https://CRAN.R-project.org/package=FastPCS), [FastRCS](https://CRAN.R-project.org/package=FastRCS), [fda.usc](https://CRAN.R-project.org/package=fda.usc), [fdaoutlier](https://CRAN.R-project.org/package=fdaoutlier), [fdasrvf](https://CRAN.R-project.org/package=fdasrvf), [funModeling](https://CRAN.R-project.org/package=funModeling), [GGoutlieR](https://CRAN.R-project.org/package=GGoutlieR), [hotspots](https://CRAN.R-project.org/package=hotspots), [ICSOutlier](https://CRAN.R-project.org/package=ICSOutlier), [isotree](https://CRAN.R-project.org/package=isotree), [kernlab](https://CRAN.R-project.org/package=kernlab), [kfino](https://CRAN.R-project.org/package=kfino), [kmodR](https://CRAN.R-project.org/package=kmodR), [kuiper.2samp](https://CRAN.R-project.org/package=kuiper.2samp), [ldbod](https://CRAN.R-project.org/package=ldbod), [lookout](https://CRAN.R-project.org/package=lookout), [MALDIrppa](https://CRAN.R-project.org/package=MALDIrppa), [MGBT](https://CRAN.R-project.org/package=MGBT), [modi](https://CRAN.R-project.org/package=modi), [Morpho](https://CRAN.R-project.org/package=Morpho), [mrct](https://CRAN.R-project.org/package=mrct), [mrfDepth](https://CRAN.R-project.org/package=mrfDepth), [mvout](https://CRAN.R-project.org/package=mvout), [mvoutlier](https://CRAN.R-project.org/package=mvoutlier), [NMAoutlier](https://CRAN.R-project.org/package=NMAoutlier), [npphen](https://CRAN.R-project.org/package=npphen), [oclust](https://CRAN.R-project.org/package=oclust), [oddstream](https://CRAN.R-project.org/package=oddstream), [odetector](https://CRAN.R-project.org/package=odetector), [outbreaker2](https://CRAN.R-project.org/package=outbreaker2), [outbreaks](https://CRAN.R-project.org/package=outbreaks), [outForest](https://CRAN.R-project.org/package=outForest), [outlierensembles](https://CRAN.R-project.org/package=outlierensembles), [outlierMBC](https://CRAN.R-project.org/package=outlierMBC), [outliers](https://CRAN.R-project.org/package=outliers), [outliers.ts.oga](https://CRAN.R-project.org/package=outliers.ts.oga), [OutliersLearn](https://CRAN.R-project.org/package=OutliersLearn), [outliertree](https://CRAN.R-project.org/package=outliertree), [pasadr](https://CRAN.R-project.org/package=pasadr), [pcadapt](https://CRAN.R-project.org/package=pcadapt), [precintcon](https://CRAN.R-project.org/package=precintcon), [probout](https://CRAN.R-project.org/package=probout), [qpcR](https://CRAN.R-project.org/package=qpcR), [rainbow](https://CRAN.R-project.org/package=rainbow), [referenceIntervals](https://CRAN.R-project.org/package=referenceIntervals), [Rlof](https://CRAN.R-project.org/package=Rlof), [Routliers](https://CRAN.R-project.org/package=Routliers), [rrcovHD](https://CRAN.R-project.org/package=rrcovHD), [seasonal](https://CRAN.R-project.org/package=seasonal), [seismicRoll](https://CRAN.R-project.org/package=seismicRoll), [SeleMix](https://CRAN.R-project.org/package=SeleMix), [sGMRFmix](https://CRAN.R-project.org/package=sGMRFmix), [SMLoutliers](https://CRAN.R-project.org/package=SMLoutliers), [solitude](https://CRAN.R-project.org/package=solitude), [spikes](https://CRAN.R-project.org/package=spikes), [StatDA](https://CRAN.R-project.org/package=StatDA), [stray](https://CRAN.R-project.org/package=stray), [survBootOutliers](https://CRAN.R-project.org/package=survBootOutliers), [surveillance](https://CRAN.R-project.org/package=surveillance), [trendsegmentR](https://CRAN.R-project.org/package=trendsegmentR), [TSA](https://CRAN.R-project.org/package=TSA), [tsoutliers](https://CRAN.R-project.org/package=tsoutliers), [univOutl](https://CRAN.R-project.org/package=univOutl), [washeR](https://CRAN.R-project.org/package=washeR), [wbacon](https://CRAN.R-project.org/package=wbacon), [wql](https://CRAN.R-project.org/package=wql). |

### Related links

- CRAN Task View: [Cluster](Cluster.html)
- CRAN Task View: [Epidemiology](Epidemiology.html)
- CRAN Task View: [ExtremeValue](ExtremeValue.html)
- [GitHub repository for this Task
  View](https://github.com/pridiltal/ctv-AnomalyDetection)
