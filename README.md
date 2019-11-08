## CRAN Task View: Anomaly Detection with R

                                                                     
--------------- --------------------------------------------------   
**Maintainer:** Priyanga Dilini Talagala, Rob J. Hyndman             
**Contact:**    Dilini.Talagala at monash.edu                        
**Version:**    2019-10-07                                           
**URL:**        <https://CRAN.R-project.org/view=AnomalyDetection>   

<div>

This CRAN task view contains a list of packages that can be used for
anomaly detection. Anomaly detection problems have many different facets
and the detection techniques can be highly influenced by the way we
define anomalies, the type of input data to the algorithm, the expected
output, etc. This leads to wide variations in problem formulations,
which need to be addressed through different analytical approaches.

Anomalies are often mentioned under several alternative names such as
outliers, novelty, odd values, extreme values, faults in different
application domains. These variants are also considered for this task
view.

The development of this task view is fairly new and still in its early
stages and therefore subject to changes. Please send suggestions for
additions and extensions for this task view to the task view maintainer.

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
  - The [DDoutlier](https://cran.r-project.org/package=DDoutlier) package provides a
    wide variety of distance- and density-based outlier detection
    functions mainly focusing local outliers in high-dimensional data.
  - The [OutlierDetection](https://cran.r-project.org/package=OutlierDetection)
    package provides different implementations for outlier detection
    namely model based, distance based, dispersion based, depth based
    and density based. This package provides labelling of observations
    as outliers and outlierliness of each outlier. For univariate,
    bivariate and trivariate data, visualization is also provided.
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
  - The [HDoutliers](https://cran.r-project.org/package=HDoutliers) package provides
    an implementation of an algorithm for univariate and multivariate
    outlier detection that can handle data with a mixed categorical and
    continuous variables and outlier masking problem.
  - The [mvoutlier](https://cran.r-project.org/package=mvoutlier) package provides
    multivariate outlier detection based on robust methods.
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
  - Function `dm.mahalanobis` in [DJL](https://cran.r-project.org/package=DJL)
    package implements Mahalanobis distance measure for outlier
    detection. In addition to the basic distance measure, boxplots are
    provided with potential outlier(s) to give an insight into the early
    stage of data cleansing task.
  - The [kmodR](https://cran.r-project.org/package=kmodR) package presents a unified
    approach for simultaneously clustering and discovering outliers in
    high dimensional data. Their approach is formalized as a
    generalization of the k-MEANS problem.
  - The [CrossClustering](https://cran.r-project.org/package=CrossClustering)
    package implements a partial clustering algorithm that combines the
    Ward's minimum variance and Complete Linkage algorithms, providing
    automatic estimation of a suitable number of clusters and
    identification of outlier elements.
  - The [DMwR2](https://cran.r-project.org/package=DMwR2) package uses hierarchical
    clustering to obtain a ranking of outlierness for a set of cases.
    The ranking is obtained on the basis of the path each case follows
    within the merging steps of a agglomerative hierarchical clustering
    method.
  - The [abodOutlier](https://cran.r-project.org/package=abodOutlier) package
    performs angle-based outlier detection on high dimensional data. A
    complete, a randomized and a knn based methods are available.
  - The [HighDimOut](https://cran.r-project.org/package=HighDimOut) package provides
    three high-dimensional outlier detection algorithms (angle-based,
    subspace based, feature bagging-based) and an outlier unification
    scheme.
  - A set of algorithms for detection of outliers based on frequent
    pattern mining is available in
    [fpmoutliers](https://cran.r-project.org/package=fpmoutliers) package. Such
    algorithms follow the paradigm: if an instance contains more
    frequent patterns,it means that this data instance is unlikely to be
    an anomaly.
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
  - Explainable outlier detection method through decision tree
    conditioning is facilitated by
    [outliertree](https://cran.r-project.org/package=outliertree) package .
  - The [mrfDepth](https://cran.r-project.org/package=mrfDepth) package provides
    tools to compute depth measures and implementations of related tasks
    such as outlier detection, data exploration and classification of
    multivariate, regression and functional data.
  - The [evtclass](https://cran.r-project.org/package=evtclass) package provides two
    classifiers for open set recognition and novelty detection based on
    extreme value theory.
  - The [dlookr](https://cran.r-project.org/package=dlookr) package provides a
    collection of tools that support data diagnosis, exploration, and
    transformation. Data diagnostics provides information and
    visualization of missing values and outliers and unique and negative
    values to understand the distribution and quality of data.
  - The [RaPKod](https://cran.r-project.org/package=RaPKod) package implements a
    kernel method that performs online outlier detection through random
    low dimensional projections in a kernel space on the basis of a
    reference set of non-outliers.
  - The [FastHCS](https://cran.r-project.org/package=FastHCS) package implements
    robust algorithm for principal component analysis and thereby
    provide robust PCA modelling and associated outlier detection and
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
    Point Anomaly (MVCAPA), and Proportion Adaptive Segment Selection
    (PASS) methods for the detection of *anomalies* in time series data.
  - The [anomalize](https://cran.r-project.org/package=anomalize) package enables a
    "tidy" workflow for detecting anomalies in data. The main functions
    are `time_decompose()`, `anomalize()`, and `time_recompose()`.
  - The [cbar](https://cran.r-project.org/package=cbar) package detect contextual
    anomalies in time-series data with Bayesian data analysis. It
    focuses on determining a normal range of target value, and provides
    simple-to-use functions to abstract the outcome.
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
  - The [SmartSifter](https://cran.r-project.org/package=SmartSifter) package
    provides online unsupervised outlier detection methods using finite
    mixtures with discounting learning algorithms.
  - Package [mmppr](https://cran.r-project.org/package=mmppr) (Markov modulated
    Poisson process) provides a framework for detecting anomalous events
    in time series of counts using an unsupervised learning approach.
  - A set of online fault (anomaly) detectors for time series using
    prediction-based and window-based techniques are available via
    [otsad](https://cran.r-project.org/package=otsad) package. It can handle both
    stationary and non-stationary environments.
  - The [jmotif](https://cran.r-project.org/package=jmotif) package provides tools
    based on Symbolic aggregate for finding discords (i.e. time series
    anomaly/ unusual time series subsequence).
  - The `detectAO` and `detectIO` functions in
    [TSA](https://cran.r-project.org/package=TSA) package support detecting additive
    outlier and innovative outlier in time series data.

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

  - Scan statistics are used to detect anomalous clusters in spatial or
    space-time data. The
    [scanstatistics](https://cran.r-project.org/package=scanstatistics) package
    provides functions for detection of anomalous space-time clusters
    using the scan statistics methodology. It focuses on prospective
    surveillance of data streams, scanning for clusters with ongoing
    anomalies.
  - The [solitude](https://cran.r-project.org/package=solitude) package provides an
    implementation of Isolation forest which detects anomalies purely
    based on the concept of isolation without employing any distance or
    density measure.
  - Functions for error detection and correction in point data quality
    datasets that are used in species distribution modelling are
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

  - The [preprocomb](https://cran.r-project.org/package=preprocomb) package provides
    an S4 framework for creating and evaluating preprocessing
    combinations for classification, clustering and outlier detection.
  - The [dobin](https://cran.r-project.org/package=dobin) package provides dimension
    reduction technique for outlier detection using neighbours,
    constructs a set of basis vectors for outlier detection. It brings
    outliers to the fore-front using fewer basis vectors.

**Specific Application Fields**

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
  - The [KRIS](https://cran.r-project.org/package=KRIS) package provides useful
    functions which are needed for bioinformatic analysis including
    detection of rough structures and outliers using unsupervised
    clustering.
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
  - The [OutlierDC](https://cran.r-project.org/package=OutlierDC) package implements
    algorithms to detect outliers based on quantile regression for
    censored survival data.
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

**Data Sets**

  - The [anomaly](https://cran.r-project.org/package=anomaly) package contains
    lightcurve time series data from the Kepler telescope.
  - Various high dimensional datasets are provided by
    [mvoutlier](https://cran.r-project.org/package=mvoutlier) package.
  - The [leri](https://cran.r-project.org/package=leri) package finds and downloads
    Landscape Evaporative Response Index (LERI) data, then reads the
    data into R. The LERI product measures anomalies in actual
    evapotranspiration, to support drought monitoring and early warning
    systems.
  - The [waterData](https://cran.r-project.org/package=waterData) package imports
    U.S. Geological Survey (USGS) daily hydrologic data from USGS web
    services and provides functions to calculate and plot anomalies.

**Miscellaneous**

  - The [analytics](https://cran.r-project.org/package=analytics) package provides
    support for (among other functions) outlier detection in a fitted
    linear model.
  - The [nlr](https://cran.r-project.org/package=nlr) package include tools to
    detecting outliers in nonlinear regression.
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
  - Outlier detection for compositional data using (robust) Mahalanobis
    distances in isometric logratio coordinates is implemented in
    `outCoDa()` function of
    [robCompositions](https://cran.r-project.org/package=robCompositions) package.
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
    equation
    models.

</div>

### CRAN packages:

  - [abodOutlier](https://cran.r-project.org/package=abodOutlier)
  - [adamethods](https://cran.r-project.org/package=adamethods)
  - [alphaOutlier](https://cran.r-project.org/package=alphaOutlier)
  - [amelie](https://cran.r-project.org/package=amelie)
  - [analytics](https://cran.r-project.org/package=analytics)
  - [ANN2](https://cran.r-project.org/package=ANN2)
  - [anomalize](https://cran.r-project.org/package=anomalize)
  - [anomaly](https://cran.r-project.org/package=anomaly)
  - [bigutilsr](https://cran.r-project.org/package=bigutilsr)
  - [biogeo](https://cran.r-project.org/package=biogeo)
  - [cbar](https://cran.r-project.org/package=cbar)
  - [cellWise](https://cran.r-project.org/package=cellWise)
  - [CerioliOutlierDetection](https://cran.r-project.org/package=CerioliOutlierDetection)
  - [CircOutlier](https://cran.r-project.org/package=CircOutlier)
  - [compositions](https://cran.r-project.org/package=compositions)
  - [CoordinateCleaner](https://cran.r-project.org/package=CoordinateCleaner)
  - [CrossClustering](https://cran.r-project.org/package=CrossClustering)
  - [dave](https://cran.r-project.org/package=dave)
  - [dbscan](https://cran.r-project.org/package=dbscan)
  - [ddalpha](https://cran.r-project.org/package=ddalpha)
  - [DDoutlier](https://cran.r-project.org/package=DDoutlier) (core)
  - [densratio](https://cran.r-project.org/package=densratio)
  - [depth.plot](https://cran.r-project.org/package=depth.plot)
  - [DescTools](https://cran.r-project.org/package=DescTools)
  - [dixonTest](https://cran.r-project.org/package=dixonTest)
  - [DJL](https://cran.r-project.org/package=DJL)
  - [dlookr](https://cran.r-project.org/package=dlookr)
  - [DMwR2](https://cran.r-project.org/package=DMwR2)
  - [dobin](https://cran.r-project.org/package=dobin)
  - [EFDR](https://cran.r-project.org/package=EFDR)
  - [enpls](https://cran.r-project.org/package=enpls)
  - [evtclass](https://cran.r-project.org/package=evtclass)
  - [extremevalues](https://cran.r-project.org/package=extremevalues)
  - [FactoInvestigate](https://cran.r-project.org/package=FactoInvestigate)
  - [faoutlier](https://cran.r-project.org/package=faoutlier)
  - [FastHCS](https://cran.r-project.org/package=FastHCS)
  - [FastPCS](https://cran.r-project.org/package=FastPCS)
  - [FastRCS](https://cran.r-project.org/package=FastRCS)
  - [fda.usc](https://cran.r-project.org/package=fda.usc)
  - [fdasrvf](https://cran.r-project.org/package=fdasrvf)
  - [fpmoutliers](https://cran.r-project.org/package=fpmoutliers)
  - [funModeling](https://cran.r-project.org/package=funModeling)
  - [GmAMisc](https://cran.r-project.org/package=GmAMisc)
  - [HDoutliers](https://cran.r-project.org/package=HDoutliers) (core)
  - [HighDimOut](https://cran.r-project.org/package=HighDimOut)
  - [hotspots](https://cran.r-project.org/package=hotspots)
  - [ICSOutlier](https://cran.r-project.org/package=ICSOutlier)
  - [jmotif](https://cran.r-project.org/package=jmotif)
  - [kernlab](https://cran.r-project.org/package=kernlab)
  - [kmodR](https://cran.r-project.org/package=kmodR)
  - [KRIS](https://cran.r-project.org/package=KRIS)
  - [kuiper.2samp](https://cran.r-project.org/package=kuiper.2samp)
  - [ldbod](https://cran.r-project.org/package=ldbod)
  - [leri](https://cran.r-project.org/package=leri)
  - [MALDIrppa](https://cran.r-project.org/package=MALDIrppa)
  - [MIPHENO](https://cran.r-project.org/package=MIPHENO)
  - [mmppr](https://cran.r-project.org/package=mmppr)
  - [modi](https://cran.r-project.org/package=modi)
  - [Morpho](https://cran.r-project.org/package=Morpho)
  - [mrfDepth](https://cran.r-project.org/package=mrfDepth)
  - [mvoutlier](https://cran.r-project.org/package=mvoutlier)
  - [nlr](https://cran.r-project.org/package=nlr)
  - [NMAoutlier](https://cran.r-project.org/package=NMAoutlier)
  - [npphen](https://cran.r-project.org/package=npphen)
  - [oclust](https://cran.r-project.org/package=oclust)
  - [otsad](https://cran.r-project.org/package=otsad)
  - [OutlierDC](https://cran.r-project.org/package=OutlierDC)
  - [OutlierDetection](https://cran.r-project.org/package=OutlierDetection)
  - [OutlierDM](https://cran.r-project.org/package=OutlierDM)
  - [outliers](https://cran.r-project.org/package=outliers)
  - [OutliersO3](https://cran.r-project.org/package=OutliersO3) (core)
  - [outliertree](https://cran.r-project.org/package=outliertree)
  - [pcadapt](https://cran.r-project.org/package=pcadapt)
  - [precintcon](https://cran.r-project.org/package=precintcon)
  - [preprocomb](https://cran.r-project.org/package=preprocomb)
  - [probout](https://cran.r-project.org/package=probout)
  - [qpcR](https://cran.r-project.org/package=qpcR)
  - [quokar](https://cran.r-project.org/package=quokar)
  - [rainbow](https://cran.r-project.org/package=rainbow)
  - [RaPKod](https://cran.r-project.org/package=RaPKod)
  - [referenceIntervals](https://cran.r-project.org/package=referenceIntervals)
  - [rgr](https://cran.r-project.org/package=rgr)
  - [Rlof](https://cran.r-project.org/package=Rlof)
  - [robCompositions](https://cran.r-project.org/package=robCompositions)
  - [Routliers](https://cran.r-project.org/package=Routliers)
  - [rrcovHD](https://cran.r-project.org/package=rrcovHD)
  - [RWBP](https://cran.r-project.org/package=RWBP)
  - [scanstatistics](https://cran.r-project.org/package=scanstatistics)
  - [seasonal](https://cran.r-project.org/package=seasonal)
  - [seismicRoll](https://cran.r-project.org/package=seismicRoll)
  - [SeleMix](https://cran.r-project.org/package=SeleMix)
  - [semdiag](https://cran.r-project.org/package=semdiag)
  - [sGMRFmix](https://cran.r-project.org/package=sGMRFmix)
  - [SmartSifter](https://cran.r-project.org/package=SmartSifter)
  - [SMLoutliers](https://cran.r-project.org/package=SMLoutliers)
  - [solitude](https://cran.r-project.org/package=solitude)
  - [spikes](https://cran.r-project.org/package=spikes)
  - [StatDA](https://cran.r-project.org/package=StatDA)
  - [survBootOutliers](https://cran.r-project.org/package=survBootOutliers)
  - [surveyoutliers](https://cran.r-project.org/package=surveyoutliers)
  - [trendsegmentR](https://cran.r-project.org/package=trendsegmentR)
  - [TSA](https://cran.r-project.org/package=TSA)
  - [tsoutliers](https://cran.r-project.org/package=tsoutliers)
  - [univOutl](https://cran.r-project.org/package=univOutl)
  - [washeR](https://cran.r-project.org/package=washeR)
  - [waterData](https://cran.r-project.org/package=waterData)
  - [wql](https://cran.r-project.org/package=wql)

### Related links:

  - CRAN Task View: [Cluster](Cluster.html)
  - CRAN Task View: [ExtremeValue](ExtremeValue.html)
  - [GitHub repository for this Task
    View](https://github.com/pridiltal/ctv-AnomalyDetection)
