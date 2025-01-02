## CRAN Task View: Anomaly Detection with R

                                                                     
--------------- --------------------------------------------------   
**Maintainer:** Priyanga Dilini Talagala, Rob J. Hyndman             
**Contact:**    pritalagala at gmail.com                             
**Version:**    2022-12-31                                           
**URL:**        <https://CRAN.R-project.org/view=AnomalyDetection>   

<div>

This CRAN task view contains a list of packages that can be used for
anomaly detection. Anomaly detection problems have many different facets
and the detection techniques can be highly influenced by the way we
define anomalies, the type of input data to the algorithm, the expected
output, etc. This leads to wide variations in problem formulations,
which need to be addressed through different analytical approaches.

Anomalies are often mentioned under several alternative names such as
outliers, novelty, odd values, extreme values, faults, aberration in
different application domains. These variants are also considered for
this task view.

**The development of this task view is fairly new and still in its early
stages and therefore subject to changes. Please send suggestions for
additions and extensions for this task view to the task view
maintainer.**

**Univariate Outlier Detection**

  - *Univariate outlier* detection methods focus on values in a single
    feature space. Package [univOutl](https://cran.r-project.org/package=univOutl)
    includes various methods for detecting univariate outliers, e.g. the
    Hidiroglou-Berthelot method. Methods to deal with skewed
    distribution are also included in this package.
  - The [dixonTest](https://cran.r-project.org/package=dixonTest) package provides
    Dixon's ratio test for outlier detection in small and normally
    distributed samples.
  - Univariate outliers detection is also supported by `outlier()`
    function in [GmAMisc](https://cran.r-project.org/package=GmAMisc) which
    implements three different methods (mean-base, median-based,
    boxplot-based).
  - The [hotspots](https://cran.r-project.org/package=hotspots) package supports
    univariate outlier detection by identifying values that are
    disproportionately high based on both the deviance of any given
    value from a statistical distribution and its similarity to other
    values.
  - The [outliers](https://cran.r-project.org/package=outliers) package provides a
    collection of tests commonly used for identifying *outliers* . For
    most functions the input is a numeric vector. If argument is a data
    frame, then outlier is calculated for each column by sapply. The
    same behavior is applied by apply when the matrix is given.
  - The [extremevalues](https://cran.r-project.org/package=extremevalues) package
    offers outlier detection and plot functions for univariate data. In
    this work a value in the data is an outlier when it is unlikely to
    be drawn from the estimated distribution.
  - The [funModeling](https://cran.r-project.org/package=funModeling) package
    provides tools for outlier detection using top/bottom X%, Tukey’s
    boxplot definition and Hampel’s method.
  - The [alphaOutlier](https://cran.r-project.org/package=alphaOutlier) package
    provides Alpha-Outlier regions (as proposed by Davies and Gather
    (1993)) for well-known probability distributions.

**Multivariate Outlier Detection**

  - Under *multivariate, high-dimensional or multidimensional scenario,*
    where the focus is on n (\>2) - dimensional space, all attributes
    might be of same type or might be a mixture of different types such
    as categorical or numerical, which has a direct impact on the
    implementation and scope of the algorithm. The problems of anomaly
    detection in high-dimensional data are threefold, involving
    detection of: (a) global anomalies, (b) local anomalies and (c)
    micro clusters or clusters of anomalies. Global anomalies are very
    different from the dense area with respect to their attributes. In
    contrast, a local anomaly is only an anomaly when it is distinct
    from, and compared with, its local neighbourhood. Micro clusters or
    clusters of anomalies may cause masking problems.

*Multivariate Outlier Detection: Density-based outlier detection*

  - The [DDoutlier](https://cran.r-project.org/package=DDoutlier) package provides a
    wide variety of distance- and density-based outlier detection
    functions mainly focusing local outliers in high-dimensional data.
  - *Local Outlier Factor (LOF)* is an algorithm for detecting anomalous
    data points by measuring the local deviation of a given data point
    with respect to its neighbours. This algorithm with some variations
    is supported by many packages. The
    [DescTools](https://cran.r-project.org/package=DescTools) package provides
    functions for outlier detection using LOF and Tukey’s boxplot
    definition. Functions `LOF()` and `GLOSH` in package
    [dbscan](https://cran.r-project.org/package=dbscan) provide density based
    anomaly detection methods using a kd-tree to speed up kNN search.
    Parallel implementation of LOF which uses multiple CPUs to
    significantly speed up the LOF computation for large datasets is
    available in [Rlof](https://cran.r-project.org/package=Rlof) package. Package
    [bigutilsr](https://cran.r-project.org/package=bigutilsr) provides utility
    functions for outlier detection in large-scale data. It includes LOF
    and outlier detection method based on departure from histogram.
  - The [SMLoutliers](https://cran.r-project.org/package=SMLoutliers) package
    provides an implementation of the Local Correlation Integral method
    (Lof: Identifying density-based local outliers) for outlier
    detection in multivariate data which consists of numeric values.
  - The [ldbod](https://cran.r-project.org/package=ldbod) package provides flexible
    functions for computing local density-based outlier scores. It
    allows for subsampling of input data or a user specified reference
    data set to compute outlier scores against, so both unsupervised and
    semi-supervised outlier detection can be done.
  - The [kernlab](https://cran.r-project.org/package=kernlab) package provides
    kernel-based machine learning methods including one-class Support
    Vector Machines for *novelty* detection.
  - The [amelie](https://cran.r-project.org/package=amelie) package implements
    anomaly detection as binary classification for multivariate
  - The estimated density ratio function in
    [densratio](https://cran.r-project.org/package=densratio) package can be used in
    many applications such as anomaly detection, change-point detection,
    covariate shift adaptation.
  - The [lookout](https://cran.r-project.org/package=lookout) package detects
    outliers using leave-one-out kernel density estimates and extreme
    value theory. The bandwidth for kernel density estimates is computed
    using persistent homology, a technique in topological data analysis.
    It also has the capability to explore the birth and the cessation of
    outliers with changing bandwidth and significance levels via
    `persisting_outliers().`
  - The Weighted BACON (blocked adaptive computationally-efficient
    outlier nominators) algorithms in
    [wbacon](https://cran.r-project.org/package=wbacon) implement a weighted variant
    of the BACON algorithms for multivariate outlier detection and
    robust linear regression. The methods assume that the typical data
    follows an elliptically contoured distribution.

*Multivariate Outlier Detection: Distance-based outlier detection*

  - The [HDoutliers](https://cran.r-project.org/package=HDoutliers) package provides
    an implementation of an algorithm for univariate and multivariate
    outlier detection that can handle data with a mixed categorical and
    continuous variables and outlier masking problem.
  - The [stray](https://cran.r-project.org/package=stray) package implements an
    algorithm for detecting anomalies in high-dimensional data that
    addresses the limitations of 'HDoutliers' algorithm. An approach
    based on extreme value theory is used for the anomalous threshold
    calculation.
  - The [Routliers](https://cran.r-project.org/package=Routliers) package provides
    robust methods to detect univariate (Median Absolute Deviation
    method) and multivariate outliers (Mahalanobis-Minimum Covariance
    Determinant method).
  - The [modi](https://cran.r-project.org/package=modi) package implements
    Mahalanobis distance or depth-based algorithms for multivariate
    outlier detection in the presence of missing values (incomplete
    survey data).
  - The
    [CerioliOutlierDetection](https://cran.r-project.org/package=CerioliOutlierDetection)
    package implements the iterated RMCD method of Cerioli (2010) for
    multivariate outlier detection via robust Mahalanobis distances.
  - The [rrcovHD](https://cran.r-project.org/package=rrcovHD) package performs
    outlier identification using robust multivariate methods based on
    robust mahalanobis distances and principal component analysis.
  - The [mvoutlier](https://cran.r-project.org/package=mvoutlier) package also
    provides various robust methods based multivariate outlier detection
    capabilities. This includes a Mahalanobis type method with an
    adaptive outlier cutoff value, a method incorporating local
    neighborhood and a method for compositional data.
  - Function `dm.mahalanobis` in [DJL](https://cran.r-project.org/package=DJL)
    package implements Mahalanobis distance measure for outlier
    detection. In addition to the basic distance measure, boxplots are
    provided with potential outlier(s) to give an insight into the early
    stage of data cleansing task.

*Multivariate Outlier Detection: Clustering-based outlier detection*

  - The [kmodR](https://cran.r-project.org/package=kmodR) package presents a unified
    approach for simultaneously clustering and discovering outliers in
    high dimensional data. Their approach is formalized as a
    generalization of the k-MEANS problem.
  - The [DMwR2](https://cran.r-project.org/package=DMwR2) package uses hierarchical
    clustering to obtain a ranking of outlierness for a set of cases.
    The ranking is obtained on the basis of the path each case follows
    within the merging steps of a agglomerative hierarchical clustering
    method.

*Multivariate Outlier Detection: Angle-based outlier detection*

  - The [abodOutlier](https://cran.r-project.org/package=abodOutlier) package
    performs angle-based outlier detection on high dimensional data. A
    complete, a randomized and a knn based methods are available.

*Multivariate Outlier Detection: Decision tree based approaches*

  - Explainable outlier detection method through decision tree
    conditioning is facilitated by
    [outliertree](https://cran.r-project.org/package=outliertree) package .
  - The
    [bagged.outliertrees](https://cran.r-project.org/package=bagged.outliertrees)
    package provides an explainable unsupervised outlier detection
    method based on an ensemble implementation of the existing
    OutlierTree procedure in
    [outliertree](https://cran.r-project.org/package=outliertree) package. The
    implementation takes advantage of bootstrap aggregating (bagging) to
    improve robustness by reducing the possible masking effect and
    subsequent high variance (similarly to Isolation Forest), hence the
    name "Bagged OutlierTrees".
  - The [isotree](https://cran.r-project.org/package=isotree) package provides fast
    and multi-threaded implementation of Extended Isolation Forest,
    Fair-Cut Forest, SCiForest (a.k.a. Split-Criterion iForest), and
    regular Isolation Forest, for isolation-based outlier detection,
    clustered outlier detection, distance or similarity approximation,
    and imputation of missing values based on random or guided decision
    tree splitting. It also supports categorical data.
  - The [outForest](https://cran.r-project.org/package=outForest) package provides a
    random forest based implementation for multivariate outlier
    detection. In this method each numeric variable is regressed onto
    all other variables by a random forest. If the scaled absolute
    difference between observed value and out-of-bag prediction of the
    corresponding random forest is suspiciously large, then a value is
    considered an outlier.
  - The [solitude](https://cran.r-project.org/package=solitude) package provides an
    implementation of Isolation forest which detects anomalies in
    cross-sectional tabular data purely based on the concept of
    isolation without employing any distance or density measures.

*Multivariate Outlier Detection: Other approaches*

  - The [abnormality](https://cran.r-project.org/package=abnormality) package
    measures a Subject's Abnormality with Respect to a Reference
    Population. A methodology is introduced to address this bias to
    accurately measure overall abnormality in high dimensional spaces.
    It can be applied to datasets in which the number of observations is
    less than the number of features/variables, and it can be abstracted
    to practically any number of domains or dimensions.
  - The [ICSOutlier](https://cran.r-project.org/package=ICSOutlier) package performs
    multivariate outlier detection using invariant coordinates and
    offers different methods to choose the appropriate components. The
    current implementation targets data sets with only a small
    percentage of outliers but future extensions are under preparation.
  - The [sGMRFmix](https://cran.r-project.org/package=sGMRFmix) package provides an
    anomaly detection method for multivariate noisy sensor data using
    sparse Gaussian Markov random field mixtures. It can compute
    variable-wise anomaly scores.
  - Artificial neural networks for anomaly detection is implemented in
    [ANN2](https://cran.r-project.org/package=ANN2) package.
  - The [probout](https://cran.r-project.org/package=probout) package estimates
    unsupervised outlier probabilities for multivariate numeric
  - The [mrfDepth](https://cran.r-project.org/package=mrfDepth) package provides
    tools to compute depth measures and implementations of related tasks
    such as outlier detection, data exploration and classification of
    multivariate, regression and functional data.
  - The [evtclass](https://cran.r-project.org/package=evtclass) package provides two
    classifiers for open set recognition and novelty detection based on
    extreme value theory.
  - The [FastHCS](https://cran.r-project.org/package=FastHCS) package implements
    robust algorithm for principal component analysis and thereby
    provide robust PCA modeling and associated outlier detection and
    diagnostic tools for high-dimensional data. PCA based outlier
    detection tools are also available via
    [FactoInvestigate](https://cran.r-project.org/package=FactoInvestigate) package.
  - *Cellwise outliers* are entries in the data matrix which are
    substantially higher or lower than what could be expected based on
    the other cells in its column as well as the other cells in its row,
    taking the relations between the columns into account. Package
    [cellWise](https://cran.r-project.org/package=cellWise) provides tools for
    detecting cellwise outliers and robust methods to analyze data which
    may contain them.
  - *The Projection Congruent Subset (PCS)* is a method for finding
    multivariate outliers by searching for a subset which minimizes a
    criterion. PCS is supported by
    [FastPCS](https://cran.r-project.org/package=FastPCS) package.
  - The [molic](https://cran.r-project.org/package=molic) package provides an
    outlier detection method for high‐dimensional contingency tables
    using decomposable graphical models
  - The [outlierensembles](https://cran.r-project.org/package=outlierensembles)
    package provides ensemble functions for outlier/anomaly detection.
    In addition to some exiting ensemble methods for outlier detcetion,
    it also provides an Item Response Theory based ensemble method.

**Temporal Data**

  - The problems of anomaly detection for temporal data are 3-fold: (a)
    the detection of contextual anomalies (point anomalies) within a
    given series; (b) the detection of anomalous subsequences within a
    given series; and (c) the detection of anomalous series within a
    collection of series
  - The [trendsegmentR](https://cran.r-project.org/package=trendsegmentR) package
    performs the detection of point anomalies and linear trend changes
    for univariate time series by implementing the bottom-up unbalanced
    wavelet transformation.
  - The [anomaly](https://cran.r-project.org/package=anomaly) package implements
    Collective And Point Anomaly (CAPA), Multi-Variate Collective And
    Point Anomaly (MVCAPA), Proportion Adaptive Segment Selection (PASS)
    and Bayesian Abnormal Region Detector (BARD) methods for the
    detection of *anomalies* in time series data.
  - The [anomalize](https://cran.r-project.org/package=anomalize) package enables a
    "tidy" workflow for detecting anomalies in data. The main functions
    are `time_decompose()`, `anomalize()`, and `time_recompose()`.
  - The [cbar](https://cran.r-project.org/package=cbar) package detect contextual
    anomalies in time-series data with Bayesian data analysis. It
    focuses on determining a normal range of target value, and provides
    simple-to-use functions to abstract the outcome.
  - The `detectAO` and `detectIO` functions in
    [TSA](https://cran.r-project.org/package=TSA) package support detecting additive
    outlier and innovative outlier in time series data.
  - The [washeR](https://cran.r-project.org/package=washeR) package performs time
    series outlier detection using non parametric test. An input can be
    a data frame (grouped time series: phenomenon+date+group+values) or
    a vector (single time series)
  - The [tsoutliers](https://cran.r-project.org/package=tsoutliers) package
    implements the Chen-Liu approach for detection of time series
    outliers such as innovational outliers, additive outliers, level
    shifts, temporary changes and seasonal level shifts.
  - The [seasonal](https://cran.r-project.org/package=seasonal) package provides
    easy-to-use interface to X-13-ARIMA-SEATS, the seasonal adjustment
    software by the US Census Bureau. It offers full access to almost
    all options and outputs of X-13, including outlier detection.
  - The [npphen](https://cran.r-project.org/package=npphen) package implements basic
    and high-level functions for detection of anomalies in vector data
    (numerical series/ time series) and raster data (satellite derived
    products). Processing of very large raster files is supported.
  - the [ACA](https://cran.r-project.org/package=ACA) package offers an interactive
    function for the detection of abrupt change-points or aberrations in
    point series.
  - The [oddstream](https://cran.r-project.org/package=oddstream) package implements
    an algorithm for early detection of anomalous series within a large
    collection of streaming time series data. The model uses time series
    features as inputs, and a density-based comparison to detect any
    significant changes in the distribution of the features.
  - The [pasadr](https://cran.r-project.org/package=pasadr) package provides a novel
    stealthy-attack detection mechanism that monitors time series of
    sensor measurements in real time for structural changes in the
    process behavior. It has the capability of detecting both
    significant deviations in the process behavior and subtle
    attack-indicating changes, significantly raising the bar for
    strategic adversaries who may attempt to maintain their malicious
    manipulation within the noise level.

**Spatial Outliers**

  - Spatial objects whose non-spatial attribute values are markedly
    different from those of their spatial neighbors are known as Spatial
    outliers or abnormal spatial patterns.
  - The [RWBP](https://cran.r-project.org/package=RWBP) package detects spatial
    outliers using a Random Walk on Bipartite Graph.
  - Enhanced False Discovery Rate (EFDR) is a tool to detect anomalies
    in an image. Package [EFDR](https://cran.r-project.org/package=EFDR) implements
    wavelet-based Enhanced FDR for detecting signals from complete or
    incomplete spatially aggregated data. The package also provides
    elementary tools to interpolate spatially irregular data onto a grid
    of the required size.
  - The function `spatial.outlier` in
    [depth.plot](https://cran.r-project.org/package=depth.plot) package helps to
    identify multivariate spatial outlier within a p-variate data cloud
    or if any p-variate observation is an outlier with respect to a
    p-variate data cloud.

**Spatio-Temporal Data**

  - Functions for error detection and correction in point data quality
    datasets that are used in species distribution modeling are
    available via [biogeo](https://cran.r-project.org/package=biogeo) package.
  - The [CoordinateCleaner](https://cran.r-project.org/package=CoordinateCleaner)
    package provides functions for flagging of common spatial and
    temporal outliers (errors) in biological and paleontological
    collection data, for the use in conservation, ecology and
    paleontology.

**Functional Data**

  - The `foutliers()` function from
    [rainbow](https://cran.r-project.org/package=rainbow) package provides
    functional outlier detection methods. Bagplots and boxplots for
    functional data can also be used to identify outliers, which have
    either the lowest depth (distance from the centre) or the lowest
    density, respectively.
  - The [adamethods](https://cran.r-project.org/package=adamethods) package provides
    a collection of several algorithms to obtain archetypoids with small
    and large databases and with both classical multivariate data and
    functional data (univariate and multivariate). Some of these
    algorithms also allow to detect anomalies.
  - The `shape.fd.outliers` function in
    [ddalpha](https://cran.r-project.org/package=ddalpha) package detects functional
    outliers of first three orders, based on the order extended
    integrated depth for functional data.
  - The [fda.usc](https://cran.r-project.org/package=fda.usc) package provides tools
    for outlier detection in functional data (atypical curves detection)
    using different approaches such as likelihood ratio test, depth
    measures, quantiles of the bootstrap samples.
  - The [fdasrvf](https://cran.r-project.org/package=fdasrvf) package supports
    outlier detection in functional data using the square-root velocity
    framework which allows for elastic analysis of functional data
    through phase and amplitude separation.
  - The [fdaoutlier](https://cran.r-project.org/package=fdaoutlier) package provides
    a collection of functions for outlier detection in functional data
    analysis. Methods implemented include directional outlyingness,
    MS-plot, total variation depth, and sequential transformations among
    others.

**Visualization of Anomalies**

  - The [OutliersO3](https://cran.r-project.org/package=OutliersO3) package provides
    tools to aid in the display and understanding of patterns of
    multivariate outliers. It uses the results of identifying outliers
    for every possible combination of dataset variables to provide
    insight into why particular cases are outliers.
  - The [Morpho](https://cran.r-project.org/package=Morpho) package provides a
    collection of tools for Geometric Morphometrics and mesh processing.
    Apart from the core functions it provides a graphical interface to
    find outliers and/or to switch mislabeled landmarks.
  - The [StatDA](https://cran.r-project.org/package=StatDA) package provides
    visualization tools to locate outliers in environmental data.

**Pre-processing Methods for Anomaly Detection**

  - The [dobin](https://cran.r-project.org/package=dobin) package provides dimension
    reduction technique for outlier detection using neighbours,
    constructs a set of basis vectors for outlier detection. It brings
    outliers to the fore-front using fewer basis vectors.

**Specific Application Fields**

*Epidemiology*

  - The [ABPS](https://cran.r-project.org/package=ABPS) package provides an
    implementation of the Abnormal Blood Profile Score (ABPS, part of
    the Athlete Biological Passport program of the World Anti-Doping
    Agency), which combines several blood parameters into a single score
    in order to detect blood doping. The package also contains functions
    to calculate other scores used in anti-doping programs, such as the
    OFF-score
  - The [surveillance](https://cran.r-project.org/package=surveillance) package
    implements statistical methods for aberration detection in time
    series of counts, proportions and categorical data, as well as for
    the modeling of continuous-time point processes of epidemic
    phenomena. The package also contains several real-world data sets,
    the ability to simulate outbreak data, and to visualize the results
    of the monitoring in a temporal, spatial or spatio-temporal fashion.
  - The [outbreaker2](https://cran.r-project.org/package=outbreaker2) package
    supports Bayesian reconstruction of disease outbreaks using
    epidemiological and genetic information. It is applicable to various
    densely sampled epidemics, and improves previous approaches by
    detecting unobserved and imported cases, as well as allowing
    multiple introductions of the pathogen.
  - The [outbreaks](https://cran.r-project.org/package=outbreaks) package provides
    empirical or simulated disease outbreak data, either as RData or as
    text files.

*Other*

  - The [precintcon](https://cran.r-project.org/package=precintcon) package contains
    functions to analyze the precipitation intensity, concentration and
    anomaly.
  - The [survBootOutliers](https://cran.r-project.org/package=survBootOutliers)
    package provides concordance based bootstrap methods for outlier
    detection in survival analysis.
  - The [pcadapt](https://cran.r-project.org/package=pcadapt) package provides
    methods to detect genetic markers involved in biological adaptation
    using statistical tools based on Principal Component Analysis.
  - The [rgr](https://cran.r-project.org/package=rgr) package supports exploratory
    data analysis with applied geochemical data, with special
    application to the estimation of background ranges and
    identification of anomalies to support mineral exploration and
    environmental studies.
  - The [NMAoutlier](https://cran.r-project.org/package=NMAoutlier) package
    implements the forward search algorithm for the detection of
    outlying studies (studies with extreme results) in network
    meta-analysis.
  - The [boutliers](https://cran.r-project.org/package=boutliers) package provides
    methods for outlier detection and influence diagnostics for
    meta-analysis based on Bootstrap distributions of the influence
    statistics.
  - The [dave](https://cran.r-project.org/package=dave) package provides a
    collection of functions for data analysis in vegetation ecology
    including outlier detection using nearest neighbour distances.
  - The [MALDIrppa](https://cran.r-project.org/package=MALDIrppa) package provides
    methods for quality control and robust pre-processing and analysis
    of MALDI mass spectrometry data.
  - The [MIPHENO](https://cran.r-project.org/package=MIPHENO) package contains
    functions to carry out processing of high throughput data analysis
    and detection of putative hits/mutants.
  - The [OutlierDM](https://cran.r-project.org/package=OutlierDM) package provides
    functions to detect outlying values such as genes, peptides or
    samples for multi-replicated high-throughput high-dimensional data.
  - The [qpcR](https://cran.r-project.org/package=qpcR) package implements methods
    for kinetic outlier detection (KOD) in real-time polymerase chain
    reaction (qPCR).
  - The [referenceIntervals](https://cran.r-project.org/package=referenceIntervals)
    package provides a collection of tools including outlier detcetion
    to allow the medical professional to calculate appropriate reference
    ranges (intervals) with confidence intervals around the limits for
    diagnostic purposes.
  - The Hampel filter is a robust outlier detector using Median Absolute
    Deviation (MAD). The
    [seismicRoll](https://cran.r-project.org/package=seismicRoll) package provides
    fast rolling functions for seismology including outlier detection
    with a rolling Hampel Filter.
  - The [spikes](https://cran.r-project.org/package=spikes) package provides tool to
    detect election fraud from irregularities in vote-share
    distributions using re-sampled kernel density method.
  - The [wql](https://cran.r-project.org/package=wql) package stands for \`water
    quality' provides functions including anomaly detection to assist in
    the processing and exploration of data from environmental monitoring
    programs.
  - The Grubbs‐Beck test is recommended by the federal guidelines for
    detection of low outliers in flood flow frequency computation in the
    United States. The [MGBT](https://cran.r-project.org/package=MGBT) computes the
    multiple Grubbs-Beck low-outlier test on positively distributed data
    and utilities for non-interpretive U.S. Geological Survey annual
    peak-stream flow data processing.
  - The [envoutliers](https://cran.r-project.org/package=envoutliers) package
    provides three semi-parametric methods for detection of outliers in
    environmental data based on kernel regression and subsequent
    analysis of smoothing residuals
  - The [rIP](https://cran.r-project.org/package=rIP) package supports detection of
    fraud in online surveys by tracing, scoring, and visualizing IP
    addresses
  - The [extremeIndex](https://cran.r-project.org/package=extremeIndex) computes an
    index measuring the amount of information brought by forecasts for
    extreme events, subject to calibration. This index is originally
    designed for weather or climate forecasts, but it may be used in
    other forecasting contexts.
  - The [clampSeg](https://cran.r-project.org/package=clampSeg) package provides
    tool to identify and idealize flickering events in filtered ion
    channel recordings.

**Data Sets**

  - The [anomaly](https://cran.r-project.org/package=anomaly) package contains
    lightcurve time series data from the Kepler telescope.
  - The [leri](https://cran.r-project.org/package=leri) package finds and downloads
    Landscape Evaporative Response Index (LERI) data, then reads the
    data into R. The LERI product measures anomalies in actual
    evapotranspiration, to support drought monitoring and early warning
    systems.
  - The [waterData](https://cran.r-project.org/package=waterData) package imports
    U.S. Geological Survey (USGS) daily hydrologic data from USGS web
    services and provides functions to calculate and plot anomalies.

**Miscellaneous**

  - The [CircOutlier](https://cran.r-project.org/package=CircOutlier) package
    enables detection of outliers in circular-circular regression
    models, modifying its and estimating of models parameters.
  - The Residual Congruent Subset (RCS) is a method for finding outliers
    in the regression setting. RCS is supported by
    [FastRCS](https://cran.r-project.org/package=FastRCS) package.
  - Package [quokar](https://cran.r-project.org/package=quokar) provides quantile
    regression outlier diagnostics with K Left Out Analysis.
  - The [oclust](https://cran.r-project.org/package=oclust) package provides a
    function to detect and trim outliers in Gaussian mixture model based
    clustering using methods described in Clark and McNicholas (2019).
  - The [semdiag](https://cran.r-project.org/package=semdiag) package implements
    outlier and leverage diagnostics for Structural equation modeling.
  - The [SeleMix](https://cran.r-project.org/package=SeleMix) package provides
    functions for detection of outliers and influential errors using a
    latent variable model. A mixture model (Gaussian contamination
    model) based on response(s) y and a depended set of covariates is
    fit to the data to quantify the impact of errors to the estimates.
  - The [compositions](https://cran.r-project.org/package=compositions) package
    provides functions to detect various types of outliers in
    compositional datasets.
  - The [kuiper.2samp](https://cran.r-project.org/package=kuiper.2samp) package
    performs the two-sample Kuiper test to assess the anomaly of
    continuous, one-dimensional probability distributions.
  - The `enpls.od()` function in [enpls](https://cran.r-project.org/package=enpls)
    package performs outlier detection with ensemble partial least
    squares.
  - The [surveyoutliers](https://cran.r-project.org/package=surveyoutliers) package
    helps manage outliers in sample surveys by calculating optimal
    one-sided winsorizing cutoffs.
  - The [faoutlier](https://cran.r-project.org/package=faoutlier) package provides
    tools for detecting and summarize influential cases that can affect
    exploratory and confirmatory factor analysis models and structural
    equation models.
  - The [crseEventStudy](https://cran.r-project.org/package=crseEventStudy) package
    provides a robust and powerful test of abnormal stock returns in
    long-horizon event
    studies

</div>

### CRAN packages:

  - [abnormality](https://cran.r-project.org/package=abnormality)
  - [abodOutlier](https://cran.r-project.org/package=abodOutlier)
  - [ABPS](https://cran.r-project.org/package=ABPS)
  - [ACA](https://cran.r-project.org/package=ACA)
  - [adamethods](https://cran.r-project.org/package=adamethods)
  - [alphaOutlier](https://cran.r-project.org/package=alphaOutlier)
  - [amelie](https://cran.r-project.org/package=amelie)
  - [ANN2](https://cran.r-project.org/package=ANN2)
  - [anomalize](https://cran.r-project.org/package=anomalize)
  - [anomaly](https://cran.r-project.org/package=anomaly)
  - [bagged.outliertrees](https://cran.r-project.org/package=bagged.outliertrees)
  - [bigutilsr](https://cran.r-project.org/package=bigutilsr)
  - [biogeo](https://cran.r-project.org/package=biogeo)
  - [boutliers](https://cran.r-project.org/package=boutliers)
  - [cbar](https://cran.r-project.org/package=cbar)
  - [cellWise](https://cran.r-project.org/package=cellWise)
  - [CerioliOutlierDetection](https://cran.r-project.org/package=CerioliOutlierDetection)
  - [CircOutlier](https://cran.r-project.org/package=CircOutlier)
  - [clampSeg](https://cran.r-project.org/package=clampSeg)
  - [compositions](https://cran.r-project.org/package=compositions)
  - [composits](https://cran.r-project.org/package=composits)
  - [CoordinateCleaner](https://cran.r-project.org/package=CoordinateCleaner)
  - [crseEventStudy](https://cran.r-project.org/package=crseEventStudy)
  - [dave](https://cran.r-project.org/package=dave)
  - [dbscan](https://cran.r-project.org/package=dbscan)
  - [ddalpha](https://cran.r-project.org/package=ddalpha)
  - [DDoutlier](https://cran.r-project.org/package=DDoutlier) (core)
  - [densratio](https://cran.r-project.org/package=densratio)
  - [depth.plot](https://cran.r-project.org/package=depth.plot)
  - [DescTools](https://cran.r-project.org/package=DescTools)
  - [dixonTest](https://cran.r-project.org/package=dixonTest)
  - [DJL](https://cran.r-project.org/package=DJL)
  - [DMwR2](https://cran.r-project.org/package=DMwR2)
  - [dobin](https://cran.r-project.org/package=dobin)
  - [EFDR](https://cran.r-project.org/package=EFDR)
  - [enpls](https://cran.r-project.org/package=enpls)
  - [envoutliers](https://cran.r-project.org/package=envoutliers)
  - [evtclass](https://cran.r-project.org/package=evtclass)
  - [extremeIndex](https://cran.r-project.org/package=extremeIndex)
  - [extremevalues](https://cran.r-project.org/package=extremevalues)
  - [FactoInvestigate](https://cran.r-project.org/package=FactoInvestigate)
  - [faoutlier](https://cran.r-project.org/package=faoutlier)
  - [FastHCS](https://cran.r-project.org/package=FastHCS)
  - [FastPCS](https://cran.r-project.org/package=FastPCS)
  - [FastRCS](https://cran.r-project.org/package=FastRCS)
  - [fda.usc](https://cran.r-project.org/package=fda.usc)
  - [fdaoutlier](https://cran.r-project.org/package=fdaoutlier)
  - [fdasrvf](https://cran.r-project.org/package=fdasrvf)
  - [forecast](https://cran.r-project.org/package=forecast)
  - [funModeling](https://cran.r-project.org/package=funModeling)
  - [GmAMisc](https://cran.r-project.org/package=GmAMisc)
  - [HDoutliers](https://cran.r-project.org/package=HDoutliers) (core)
  - [hotspots](https://cran.r-project.org/package=hotspots)
  - [ICSOutlier](https://cran.r-project.org/package=ICSOutlier)
  - [isotree](https://cran.r-project.org/package=isotree)
  - [kernlab](https://cran.r-project.org/package=kernlab)
  - [kmodR](https://cran.r-project.org/package=kmodR)
  - [kuiper.2samp](https://cran.r-project.org/package=kuiper.2samp)
  - [ldbod](https://cran.r-project.org/package=ldbod)
  - [leri](https://cran.r-project.org/package=leri)
  - [lookout](https://cran.r-project.org/package=lookout)
  - [MALDIrppa](https://cran.r-project.org/package=MALDIrppa)
  - [MGBT](https://cran.r-project.org/package=MGBT)
  - [MIPHENO](https://cran.r-project.org/package=MIPHENO)
  - [modi](https://cran.r-project.org/package=modi)
  - [molic](https://cran.r-project.org/package=molic)
  - [Morpho](https://cran.r-project.org/package=Morpho)
  - [mrfDepth](https://cran.r-project.org/package=mrfDepth)
  - [mvoutlier](https://cran.r-project.org/package=mvoutlier)
  - [NMAoutlier](https://cran.r-project.org/package=NMAoutlier)
  - [npphen](https://cran.r-project.org/package=npphen)
  - [oclust](https://cran.r-project.org/package=oclust)
  - [oddstream](https://cran.r-project.org/package=oddstream)
  - [otsad](https://cran.r-project.org/package=otsad)
  - [outbreaker2](https://cran.r-project.org/package=outbreaker2)
  - [outbreaks](https://cran.r-project.org/package=outbreaks)
  - [outForest](https://cran.r-project.org/package=outForest)
  - [OutlierDM](https://cran.r-project.org/package=OutlierDM)
  - [outlierensembles](https://cran.r-project.org/package=outlierensembles)
  - [outliers](https://cran.r-project.org/package=outliers)
  - [OutliersO3](https://cran.r-project.org/package=OutliersO3) (core)
  - [outliertree](https://cran.r-project.org/package=outliertree)
  - [pasadr](https://cran.r-project.org/package=pasadr)
  - [pcadapt](https://cran.r-project.org/package=pcadapt)
  - [precintcon](https://cran.r-project.org/package=precintcon)
  - [probout](https://cran.r-project.org/package=probout)
  - [qpcR](https://cran.r-project.org/package=qpcR)
  - [quokar](https://cran.r-project.org/package=quokar)
  - [rainbow](https://cran.r-project.org/package=rainbow)
  - [referenceIntervals](https://cran.r-project.org/package=referenceIntervals)
  - [rgr](https://cran.r-project.org/package=rgr)
  - [rIP](https://cran.r-project.org/package=rIP)
  - [Rlof](https://cran.r-project.org/package=Rlof)
  - [Routliers](https://cran.r-project.org/package=Routliers)
  - [rrcovHD](https://cran.r-project.org/package=rrcovHD)
  - [RWBP](https://cran.r-project.org/package=RWBP)
  - [seasonal](https://cran.r-project.org/package=seasonal)
  - [seismicRoll](https://cran.r-project.org/package=seismicRoll)
  - [SeleMix](https://cran.r-project.org/package=SeleMix)
  - [semdiag](https://cran.r-project.org/package=semdiag)
  - [sGMRFmix](https://cran.r-project.org/package=sGMRFmix)
  - [SMLoutliers](https://cran.r-project.org/package=SMLoutliers)
  - [solitude](https://cran.r-project.org/package=solitude)
  - [spikes](https://cran.r-project.org/package=spikes)
  - [StatDA](https://cran.r-project.org/package=StatDA)
  - [stray](https://cran.r-project.org/package=stray)
  - [survBootOutliers](https://cran.r-project.org/package=survBootOutliers)
  - [surveillance](https://cran.r-project.org/package=surveillance)
  - [surveyoutliers](https://cran.r-project.org/package=surveyoutliers)
  - [trendsegmentR](https://cran.r-project.org/package=trendsegmentR)
  - [TSA](https://cran.r-project.org/package=TSA)
  - [tsoutliers](https://cran.r-project.org/package=tsoutliers)
  - [univOutl](https://cran.r-project.org/package=univOutl)
  - [washeR](https://cran.r-project.org/package=washeR)
  - [waterData](https://cran.r-project.org/package=waterData)
  - [wbacon](https://cran.r-project.org/package=wbacon)
  - [wql](https://cran.r-project.org/package=wql)

### Related links:

  - CRAN Task View: [Cluster](Cluster.html)
  - CRAN Task View: [ExtremeValue](ExtremeValue.html)
  - [GitHub repository for this Task
    View](https://github.com/pridiltal/ctv-AnomalyDetection)
