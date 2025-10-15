---
name: AnomalyDetection
topic: Anomaly Detection
maintainer: Priyanga Dilini Talagala, Rob J. Hyndman, Gaetano Romano
email: priyangad@uom.lk
version: 2025-10-13
source: https://github.com/pridiltal/ctv-AnomalyDetection
---

This CRAN Task View provides a comprehensive list of R packages for anomaly detection. Anomaly detection problems have many facets, and the techniques used are influenced by factors such as how anomalies are defined, the type of input data, and the expected output. These variations lead to diverse problem formulations, requiring different analytical approaches. This Task View helps users navigate the available tools by organizing them based on their applicability to different data types and detection methodologies.

Anomalies are often referred to by alternative names such as outliers, novelties, odd values, extreme values, faults, and aberrations, depending on the application domain. In this Task View, these terms are used interchangeably. The overview covers methods applicable to univariate, multivariate, spatial, temporal, and functional data, ensuring that users can identify suitable tools for various analytical needs. Packages that do not primarily focus on anomaly detection but provide substantial functionality for it are also included.

Packages where anomaly detection is only a minor feature or offers very limited functions have been excluded. Tools that are outdated, redundant, or lack sufficient support have also not been considered.

To facilitate navigation, the Task View is structured into well-defined sections, including Univariate Outlier Detection, Multivariate Detection (further categorized into density-based, distance-based, clustering-based, angle-based, and decision tree–based methods), Temporal Data, Spatial and Spatio-Temporal Data, Functional Data, and other specialized approaches. There is some overlap between the tools in this Task View and those listed in the Task Views for `r view("Cluster")`, `r view("Epidemiology")`, `r view("ExtremeValue")`, and `r view("TimeSeries")`.

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

- The `r pkg("HDoutliers",  priority = "core")` package provides an implementation of an algorithm for univariate and multivariate outlier detection that can handle data with mixed categorical and continuous variables and the outlier masking problem.
- The `r pkg("stray")` package implements an algorithm for detecting anomalies in high-dimensional data that addresses the limitations of the 'HDoutliers' algorithm. An approach based on extreme value theory is used for the anomalous threshold calculation.
- The `r pkg("Routliers")` package provides robust methods to detect univariate (Median Absolute Deviation method) and multivariate outliers (Mahalanobis-Minimum Covariance Determinant method).
- The `r pkg("modi")` package implements Mahalanobis distance or depth-based algorithms for multivariate outlier detection in the presence of missing values (incomplete survey data).
- The `r pkg("CerioliOutlierDetection")` package implements the iterated RMCD method of Cerioli (2010) for multivariate outlier detection via robust Mahalanobis distances.
- The `r pkg("rrcovHD")` package performs outlier identification using robust multivariate methods based on robust Mahalanobis distances and principal component analysis.
- The  `r pkg("mvoutlier")` package also provides various robust methods based on multivariate outlier detection capabilities. This includes a Mahalanobis-type method with an adaptive outlier cutoff value, a method incorporating local neighbourhood, and a method for compositional data.
- The function `dm.mahalanobis()` in the `r pkg("DJL")` package implements the Mahalanobis distance measure for outlier detection. In addition to the basic distance measure, boxplots are provided with potential outlier(s) to give an insight into the early stage of the data cleansing task.
- The `r pkg("mvout")` package detects multivariate outliers using robust Mahalanobis distances based on the Minimum Covariance Determinant (MCD) estimator.
- The `r pkg("outlierMBC")` package implements sequential outlier identification for Gaussian mixture models.  Outliers are detected by comparing observed Mahalanobis distances with the theoretical distribution. It also provides an extension for Gaussian linear cluster-weighted models using studentized residuals. The method emphasizes model-based, distance-driven identification of anomalies.
- The two packages `r pkg("RMSD")` and `r pkg("RMSDp")` implement Modified Stahel-Donoho (MSD) estimators for detecting outliers in elliptically distributed multivariate datasets using Mahalanobis distance. The `r pkg("RMSD")` package provides a single-core implementation, while `r pkg("RMSDp")` offers a parallelized version optimized for high-dimensional data.

#### Multivariate Outlier Detection: Clustering-based outlier detection

- The `r pkg("kmodR")` package presents a unified approach for simultaneously clustering and discovering outliers in high-dimensional data. Their approach is formalized as a generalization of the k-MEANS problem.
- The `r pkg("odetector")` package detects multivariate outliers using soft partitioning clustering algorithms such as Fuzzy C-means and its variants. Observations with low typicality degrees are flagged as outliers.
- The  `r pkg("oclust")` package provides a function to detect and trim outliers in Gaussian mixture model-based clustering using methods described in Clark and McNicholas (2019).

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
- The `r pkg("ShapleyOutlier")` package uses Shapley values and Mahalanobis distances to explain multivariate outliers and detect or impute cellwise anomalies. It implements the methods described in Mayrhofer and Filzmoser (2023).
- The `r pkg("HRTnomaly")` package provides historical, relational, and tail anomaly detection at the data-entry level. Uses distribution-free scoring, fuzzy logic, Bayesian bootstrap, and empirical likelihood tests to identify nuanced outliers that may not be easily distinguishable from other data points. Suitable for detailed cellwise anomaly detection in structured datasets.


### Temporal Data

The problems of anomaly detection for temporal data are three-fold: (a) the detection of contextual anomalies (point anomalies) within a given series; (b) the detection of anomalous subsequences within a given series; and (c) the detection of anomalous series within a collection of series.

Related algorithms for identifying structural breaks and regime shifts are discussed in the `r view("TimeSeries", "Change point detection")` section of the Time Series Task View. Change point detection methods aim to locate abrupt changes in the statistical properties of a time series, which are conceptually similar to detecting anomalous subsequences. In some contexts, such as an epidemic change that represents a temporary deviation affecting a contiguous segment of the series, this can be regarded as the detection of anomalous subsequences within a given series.

Anomaly detection methods for temporal data can be broadly divided into two categories: Offline (Batch) procedures, which analyze the complete time series retrospectively, and Online (Sequential / Real-Time) procedures, which detect anomalies as data arrives sequentially.

#### Offline (Batch) Procedures

These methods assume that the complete time series is available for analysis and are typically applied retrospectively to detect anomalies.

- The `r pkg("trendsegmentR")` package performs detection of point anomalies and linear trend changes in univariate time series using a bottom-up unbalanced wavelet transformation. It is capable of identifying both abrupt jumps and gradual trend shifts, leveraging wavelet decomposition to separate signal from noise.
- The `r pkg("anomaly")` package implements Collective And Point Anomaly (CAPA), Multi-Variate Collective And Point Anomaly (MVCAPA), Proportion Adaptive Segment Selection (PASS), and Bayesian Abnormal Region Detector (BARD) methods. These methods combine likelihood-based segmentation and Bayesian modeling to detect both localized and global anomalies in univariate and multivariate series.
- The `r pkg("anomalize")` package provides a tidy workflow using time_decompose(), anomalize(), and time_recompose(). It decomposes series into trend, seasonal, and remainder components and identifies anomalies in the remainder using robust statistical thresholds, making it compatible with tidyverse pipelines.
- The `r pkg("TSA")` package offers `detectAO()` and `detectIO()` functions to detect additive outliers (AO) and innovational outliers (IO) using classical time series models. AO detection identifies abrupt spikes, while IO detection accounts for anomalies propagating through an ARIMA model structure.
- The `r pkg("AnomalyScore")`package computes anomaly scores for multivariate time series using a k-nearest neighbors approach, with multiple distance measures available for comparison.
- The `r pkg("washeR")` package performs time series outlier detection using a nonparametric test. An input can be a data frame (grouped time series: phenomenon+date+group+values) or a vector (single time series).
- The `r pkg("tsoutliers")` package implements the Chen-Liu approach for detection of time series outliers such as innovational outliers, additive outliers, level shifts, temporary changes, and seasonal level shifts.
- The `r pkg("seasonal")` package provides an easy-to-use interface to X-13-ARIMA-SEATS, the seasonal adjustment software by the US Census Bureau. It offers full access to almost all options and outputs of X-13, including outlier detection.
- The `r pkg("npphen")` package detects phenological cycles and anomalies in vegetation using non-parametric methods. Works with time series of vegetation indices from remote sensing or field measurements, supporting both vector and large raster data.
- The `r pkg("ACA")` package provides interactive tools for detecting abrupt change points or anomalies in serial (time-ordered) data. It identifies significant shifts or aberrations within point series, helping to locate where sudden changes occur in the underlying process dynamics.
- The `r pkg("outliers.ts.oga")` package detects and cleans outliers in single or large databases of homogeneous or heterogeneous time series using the Orthogonal Greedy Algorithm (OGA) for saturated linear regression models, providing scalable detection with parallelization..
- The `r pkg("RobKF")` package implements robust Kalman filters for additive, innovative, or combined outliers in time series, based on methods by Ruckdeschel et al. (2014), Agamennoni et al. (2018), and Fisch et al. (2020).
- The `r pkg("spectralAnomaly")` package detects anomalies in time series using the spectral residual algorithm. Provides anomaly scores for threshold-based outlier detection or integration into predictive models (Ren et al., 2019).
- The `r pkg("oddnet")` package detects anomalies in temporal networks using a feature-based approach. Features are extracted for each network, modeled with time series methods, and anomalies are identified from time series residuals, accounting for temporal dependencies.

#### Online (Sequential / Real-Time) Procedures

These methods detect anomalies as new data arrives, supporting real-time or near-real-time monitoring. Many methods include an initial offline phase for model fitting or training, followed by a sequential testing or filtering phase, often using sliding windows, adaptive thresholds, or recursive models.

- The `r pkg("oddstream")` package implements an algorithm for early detection of anomalous series within a large collection of streaming time series data. The model uses time series features as inputs and a density-based comparison to detect any significant changes in the distribution of the features.
- The `r pkg("pasadr")` package monitors sensor measurements using a dual-phase approach: an initial training phase estimates baseline process behavior, followed by sequential detection of structural changes. It can detect both abrupt deviations and subtle manipulations, including stealthy attacks, by continuously updating test statistics.
- The `r pkg("kfino")`package  detects impulse-noise outliers in streaming or sequential time series data using a Kalman filter–based recursive estimator. The package supports both Maximum Likelihood (ML) and Expectation-Maximization (EM) algorithms for parameter estimation. It provides sequential filtering, prediction, and anomaly scoring at each time step, without requiring access to the full historical series, making it well suited for real-time monitoring of sensors or automated measurement systems.


### Spatial Data

- Spatial objects whose non-spatial attribute values are markedly different from those of their spatial neighbors are known as spatial outliers or abnormal spatial patterns.
- Enhanced False Discovery Rate (EFDR) is a tool to detect anomalies in an image. Package `r pkg("EFDR")` implements wavelet-based enhanced FDR for detecting signals from complete or incomplete spatially aggregated data. The package also provides elementary tools to interpolate spatially irregular data onto a grid of the required size.
- The function `spatial.outlier()` in `r pkg("depth.plot")` package helps to identify multivariate spatial outliers within a p-variate data cloud or if any p-variate observation is an outlier with respect to a p-variate data cloud.


### Spatio-Temporal Data

- The `r pkg("CoordinateCleaner")` package provides automated tools to detect and flag common spatial and temporal errors (outliers) in biological and paleontological occurrence data. It identifies problematic coordinates such as country centroids, biodiversity institution locations, or ocean points, and flags species-level outliers and rounding errors, improving data quality for ecological and conservation analyses.
- the `r pkg("scanstatistics")` package detects anomalous space-time clusters using scan statistics. Designed for prospective surveillance of data streams, it scans for ongoing clusters and supports hypothesis testing via Monte Carlo simulation.

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

- The `r pkg("OutliersO3",  priority = "core")` package provides tools to aid in the display and understanding of patterns of multivariate outliers. It uses the results of identifying outliers for every possible combination of dataset variables to provide insight into why particular cases are outliers.
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
- The  `r pkg("extremeIndex")` package computes an index measuring the amount of information brought by forecasts for extreme events, subject to calibration. This index is originally designed for weather or climate forecasts, but it may be used in other forecasting contexts.
- The  `r pkg("npphen")` package detects phenological cycles and anomalies in vegetation using nonparametric kernel-based methods, suitable for both vector and raster data, identifying deviations from expected seasonal patterns.

#### Biomedical and Clinical Research Applications

- The `r pkg("survBootOutliers")` package provides concordance-based bootstrap methods for outlier detection in survival analysis.
- The `r pkg("referenceIntervals")` package provides a collection of tools, including outlier detection, to allow the medical professional to calculate appropriate reference ranges (intervals) with confidence intervals around the limits for diagnostic purposes.
- The `r pkg("bulkQC")` provides tools for quality control and outlier identification in multicenter randomized trials. It analyzes data from multiple study participants across several sites, detecting outliers at both the individual (univariate and multivariate) and site levels, with or without covariate adjustment. 
- The `r pkg("NMAoutlier")` package implements the forward search algorithm for the detection of outlying studies (studies with extreme results) in network meta-analysis.
- The `r pkg("boutliers")` package provides methods for outlier detection and influence diagnostics for meta-analysis based on bootstrap distributions of the influence statistics.

#### Genetics and Bioinformatics

- The `r pkg("pcadapt")` package provides methods to detect genetic markers involved in biological adaptation using statistical tools based on principal component analysis.
- The `r pkg("GGoutlieR")` package detects and visualizes individuals with unusual geo-genetic patterns using a K-nearest neighbor approach. It identifies outliers that deviate from the isolation-by-distance assumption and provides statistical summaries and geographic visualizations. 
- The `r pkg("MALDIrppa")` package provides methods for quality control and robust preprocessing and analysis of MALDI mass spectrometry data. 
- The `r pkg("qpcR")` package implements methods for kinetic outlier detection (KOD) in real-time polymerase chain reaction  (qPCR).
- The `r pkg("OmicsQC")` package analyzes quality control metrics from multi-sample genomic sequencing studies to identify poor-quality samples. It transforms per-sample metrics into z-scores, models their distribution using parametric methods, and applies Cosine Similarity Outlier Detection to nominate potential outliers.
- The `r pkg("OutSeekR")` package provides an approach to outlier detection in RNA-seq and related genomic data based on five statistics. It implements an outlier test by comparing the distributions of these statistics in observed data with those from simulated null data.
- The `r pkg("phylter")` package detects and removes outliers in phylogenomics datasets by analyzing gene trees or matrices to identify species–gene outliers. It builds on the Distatis approach, a generalization of multidimensional scaling for multiple distance matrices.

#### Seismology and Geoscience

- The Hampel filter is a robust outlier detector using Median Absolute Deviation (MAD). The `r pkg("seismicRoll")` package provides fast rolling functions for seismology, including outlier detection with a rolling Hampel filter.

#### Political Science / Election Analysis

- The `r pkg("spikes")` package provides a tool to detect election fraud from irregularities in vote-share distributions using the resampled kernel density method.

#### Finance and Econometrics Applications

- The `r pkg("crseEventStudy")` package detects abnormal stock returns in long-horizon event studies using a robust standardized test. It accounts for heteroskedasticity, autocorrelation, volatility clustering, and cross-sectional correlation, ensuring reliable identification of anomalies in financial returns.

### Data Sets

- The `r pkg("anomaly")` package contains light curve time series data from the Kepler telescope.
- The `r pkg("outbreaks")` package provides empirical or simulated disease outbreak data, either as RData or as text files.
- The `r pkg("weird",  priority = "core")` package provides all the datasets used in Hyndman (2024), [That's Weird: Anomaly Detection Using R](https://OTexts.com/weird/).
- The `r pkg("SCOUTer")` package provides a method to simulate controlled outliers using principal component analysis. New observations are generated based on target values of the Squared Prediction Error (SPE) and Hotelling’s $T^2$ statistics, allowing precise creation of outliers for testing and evaluation of anomaly detection methods.


### Educational and Companion Resources

- The `r pkg("OutliersLearn")` package provides implementations of some of the most important outlier detection algorithms. Includes a tutorial mode option that shows a description of each algorithm and provides a step-by-step execution explanation of how it identifies outliers from the given data with the specified input parameters. The package covers three main types of approaches: statistical, distance-based, and density/clustering approaches.
- The `r pkg("weird")`  accompanies Hyndman (2024), [That's Weird: Anomaly Detection Using R](https://OTexts.com/weird/). It includes all datasets, functions, and supporting packages required to reproduce the examples presented in the book.
- The `r pkg("UAHDataScienceO")` package provides implementations of key outlier detection algorithms with a tutorial mode. The tutorial explains each algorithm step-by-step, showing how outliers are identified from the input data. References include Boukerche et al. (2020), Smiti (2020), and Su & Tsai (2011).


### Miscellaneous

- The `r pkg("CircOutlier")` package enables detection of outliers in circular-circular regression models, modifying it and estimating the models' parameters.
- The Residual Congruent Subset (RCS) is a method for finding outliers in the regression setting. RCS is supported by the `r pkg("FastRCS")` package.
- The `r pkg("SeleMix")` package provides functions for detecting outliers and influential observations using a latent variable approach. It fits a mixture model (Gaussian contamination model) based on response(s) y and associated covariates to quantify the impact of errors on parameter estimates.
- The `r pkg("compositions")` package provides functions to detect various types of outliers in compositional datasets.
- The `r pkg("kuiper.2samp")` package performs the two-sample Kuiper test to assess the anomaly of continuous, one-dimensional probability distributions.
- The `enpls.od()` function in the `r pkg("enpls")` package performs outlier detection with ensemble partial least squares.
- The `r pkg("faoutlier")` package provides tools for detecting and summarizing influential cases that can affect exploratory and confirmatory factor analysis models and structural equation models.


### Links

- Articles: ["Anomaly detection: A survey" in ACM Journals (2009)](https://dl.acm.org/doi/abs/10.1145/1541880.1541882)
- Articles: ["Visualizing Big Data Outliers Through Distributed Aggregation" in IEEE Transactions on Visualization and Computer Graphics (2018)](https://dl.acm.org/doi/abs/10.1145/1541880.1541882)
- Book: [That's weird! Anomaly detection using R (Hyndman; 2024)](https://otexts.com/weird/)

