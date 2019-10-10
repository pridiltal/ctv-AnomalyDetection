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
application domains. These variants are also considered for this view.

The development of this task view is fairly new and still in its early
stages and therefore subject to changes. Please send suggestions for
additions and extensions for this task view to the task view maintainer.

**High Dimensional Data**

  - The problems of anomaly detection in high-dimensional data are
    threefold, involving detection of: (a) global anomalies, (b) local
    anomalies and (c) micro clusters or clusters of anomalies. Global
    anomalies are very different from the dense area with respect to
    their attributes. In contrast, a local anomaly is only an anomaly
    when it is distinct from, and compared with, its local
    neighbourhood. Micro clusters or clusters of anomalies may cause
    masking problems.
  - Package [dixonTest](https://cran.r-project.org/package=dixonTest) provides
    Dixon's ratio test for outlier detection in small and normally
    distributed samples.
  - Package [univOutl](https://cran.r-project.org/package=univOutl) includes various
    methods for detecting univariate outliers, e.g. the
    Hidiroglou-Berthelot method. Methods to deal with skewed
    distribution are also included in this package.
  - Package [outliers](https://cran.r-project.org/package=outliers) provides a
    collection of some tests commonly used for identifying *outliers* .
    For most functions the input is a numeric vector. If argument is a
    data frame, then outlier is calculated for each column by sapply.
    The same behavior is applied by apply when the matrix is given.
  - Package [extremevalues](https://cran.r-project.org/package=extremevalues) offers
    outlier detection and plot functions for univariate data. In this
    work a value in the data is an outlier when it is unlikely to be
    drawn from the estimated distribution.
  - Package [kernlab](https://cran.r-project.org/package=kernlab) provides
    kernel-based machine learning methods including one-class Support
    Vector Machines for *novelty* detection.
  - Package [mvoutlier](https://cran.r-project.org/package=mvoutlier) provides
    multivariate outlier detection based on robust methods.
  - Package [amelie](https://cran.r-project.org/package=amelie) implements anomaly
    detection as binary classification for cross-sectional data
    (multivariate) using maximum likelihood estimates and normal
    probability functions.
  - Package [cellWise](https://cran.r-project.org/package=cellWise) provides tools
    for detecting cellwise outliers and robust methods to analyze data
    which may contain them. Cellwise outliers are entries in the data
    matrix which are substantially higher or lower than what could be
    expected based on the other cells in its column as well as the other
    cells in its row, taking the relations between the columns into
    account.
  - Package [kmodR](https://cran.r-project.org/package=kmodR) presents a unified
    approach for simultaneously clustering and discovering outliers in
    high dimensional data. Their approach is formalized as a
    generalization of the k-MEANS problem.
  - Package [OutlierDetection](https://cran.r-project.org/package=OutlierDetection)
    provides different implementations for outlier detection namely
    model based, distance based, dispersion based, depth based and
    density based. This package provides labelling of observations as
    outliers and outlierliness of each outlier. For univariate,
    bivariate and trivariate data, visualization is also provided.
  - Package [fpmoutliers](https://cran.r-project.org/package=fpmoutliers) implements
    a set of algorithms for detection of outliers based on frequent
    pattern mining. Such algorithms follow the paradigm: if an instance
    contains more frequent patterns,it means that this data instance is
    unlikely to be an anomaly.
  - Package [ICSOutlier](https://cran.r-project.org/package=ICSOutlier) performs
    multivariate outlier detection using invariant coordinates and
    offers different methods to choose the appropriate components. The
    current implementation targets data sets with only a small
    percentage of outliers but future extensions are under preparation.
  - Package [HighDimOut](https://cran.r-project.org/package=HighDimOut) provides
    three high-dimensional outlier detection algorithms (angle-based,
    subspace based, feature bagging-based) and an outlier unification
    scheme.
  - Package [SMLoutliers](https://cran.r-project.org/package=SMLoutliers) provides
    an implementation of the Local Correlation Integral method ( Lof:
    Identifying density-based local outliers) for outlier detection in
    multivariate data which consists of numeric values.
  - Functions `LOF()` and `GLOSH` in package
    [dbscan](https://cran.r-project.org/package=dbscan) provides density based
    anomaly detection methods using a kd-tree to speed up kNN search.
  - Package [Rlof](https://cran.r-project.org/package=Rlof) provides parallel
    implementation of Local Outlier Factor(LOF) which uses multiple CPUs
    to significantly speed up the LOF computation for large datasets.
  - Package [ldbod](https://cran.r-project.org/package=ldbod) provides flexible
    functions for computing local density-based outlier scores. It
    allows for subsampling of input data or a user specified reference
    data set to compute outlier scores against, so both unsupervised and
    semi-supervised outlier detection can be done.
  - Package [DDoutlier](https://cran.r-project.org/package=DDoutlier) provides a
    wide variety of distance- and density-based outlier detection
    functions mainly focusing local outliers in high-dimensional data.
  - Package [sGMRFmix](https://cran.r-project.org/package=sGMRFmix) provides an
    anomaly detection method for multivariate noisy sensor data using
    sparse Gaussian Markov random field mixtures. It can compute
    variable-wise anomaly scores.
  - Package [Routliers](https://cran.r-project.org/package=Routliers) provides
    robust methods to detect univariate (Median Absolute Deviation
    method) and multivariate outliers (Mahalanobis-Minimum Covariance
    Determinant method).
  - Package [alphaOutlier](https://cran.r-project.org/package=alphaOutlier) provides
    Alpha-Outlier regions (as proposed by Davies and Gather (1993)) for
    well-known probability distributions.
  - Package [ANN2](https://cran.r-project.org/package=ANN2) implements artificial
    neural networks for anomaly detection.
  - Package [HDoutliers](https://cran.r-project.org/package=HDoutliers) provides an
    implementation of an algorithm for univariate and multivariate
    outlier detection that can handle data with a mixed categorical and
    continuous variables and outlier masking problem.
  - Package [probout](https://cran.r-project.org/package=probout) estimates
    unsupervised outlier probabilities for multivariate numeric data
    with many observations from a nonparametric outlier statistic.
  - Package [abodOutlier](https://cran.r-project.org/package=abodOutlier) performs
    angle-based outlier detection on high dimensional data. A complete,
    a randomized and a knn based methods are available.
  - Package [outliertree](https://cran.r-project.org/package=outliertree) provides
    explainable outlier detection method through decision tree
    conditioning.
  - Package [RaPKod](https://cran.r-project.org/package=RaPKod) implements a kernel
    method that performs online outlier detection through random
    lowdimensional projections in a kernel space on the basis of a
    reference set of non-outliers.
  - Package [modi](https://cran.r-project.org/package=modi) implements Mahalanobis
    distance or depth-based algorithms for multivariate outlier
    detection in the presence of missing values (incomplete survey
    data).
  - Package
    [CerioliOutlierDetection](https://cran.r-project.org/package=CerioliOutlierDetection)
    implements the iterated RMCD method of Cerioli (2010) for
    multivariate outlier detection via robust Mahalanobis distances.
  - Package [mrfDepth](https://cran.r-project.org/package=mrfDepth) provides tools
    to compute depth measures and implementations of related tasks such
    as outlier detection, data exploration and classification of
    multivariate, regression and functional data.
  - Package [evtclass](https://cran.r-project.org/package=evtclass) provides two
    classifiers for open set recognition and novelty detection based on
    extreme value theory.

**Temporal Data**

  - Streaming data, Batch Data  
    Point anomaly, contextual anomaly, collective anomaly, anomalous
    series within a large collection of time series
  - Package [trendsegmentR](https://cran.r-project.org/package=trendsegmentR)
    performs the detection of point anomalies and linear trend changes
    for univariate time series by implementing the bottom-up unbalanced
    wavelet transformation.
  - Package [anomaly](https://cran.r-project.org/package=anomaly) implements
    Collective And Point Anomaly (CAPA) , Multi-Variate Collective And
    Point Anomaly (MVCAPA), and Proportion Adaptive Segment Selection
    (PASS) methods for the detection of *anomalies* in time series data.
  - The [anomalize](https://cran.r-project.org/package=anomalize) package enables a
    "tidy" workflow for detecting anomalies in data. The main functions
    are `time_decompose()`, `anomalize()`, and `time_recompose()`.
  - Package [cbar](https://cran.r-project.org/package=cbar) detect contextual
    anomalies in time-series data with Bayesian data analysis. It
    focuses on determining a normal range of target value, and provides
    simple-to-use functions to abstract the outcome.
  - Package [washeR](https://cran.r-project.org/package=washeR) performs time series
    outlier detection using non parametric test. An input can be a data
    frame (grouped time series: phenomenon+date+group+values) or a
    vector (single time series)
  - Package [tsoutliers](https://cran.r-project.org/package=tsoutliers) implements
    the Chen-Liu approach for detection of time series outliers such as
    innovational outliers, additive outliers, level shifts, temporary
    changes and seasonal level shifts.
  - Package [seasonal](https://cran.r-project.org/package=seasonal) provides
    easy-to-use interface to X-13-ARIMA-SEATS, the seasonal adjustment
    software by the US Census Bureau. It offers full access to almost
    all options and outputs of X-13, including outlier detection.
  - Package [npphen](https://cran.r-project.org/package=npphen) implements basic and
    high-level functions for detection of anomalies in vector data
    (numerical series/ time series) and raster data (satellite derived
    products). Processing of very large raster files is supported.
  - Package [SmartSifter](https://cran.r-project.org/package=SmartSifter) provides
    online unsupervised outlier detection methods using finite mixtures
    with discounting learning algorithms.
  - Package [mmppr](https://cran.r-project.org/package=mmppr) (markov modulated
    Poisson process) provides a framework for detecting anomalous events
    in time series of counts using an unsupervised learning approach.
  - Package [otsad](https://cran.r-project.org/package=otsad) implements a set of
    online fault (anomaly) detectors for time series using
    prediction-based and window-based techniques. It can handle both
    stationary and non-stationary environments.

**Spatial outliers**

  - Spatial objects whose non-spatial attribute values are markedly
    different from those of their spatial neighbors are known as Spatial
    outliers or abnormal spatial patterns (Kou Y., Lu CT., 2017).
  - Package [RWBP](https://cran.r-project.org/package=RWBP) detects spatial outliers
    using a Random Walk on Bipartite Graph.

**Spatio-Temporal data**

  - Scan statistics are used to detect anomalous clusters in spatial or
    space-time data. Package
    [scanstatistics](https://cran.r-project.org/package=scanstatistics) provides
    functions for detection of anomalous space-time clusters using the
    scan statistics methodology. Focuses on prospective surveillance of
    data streams, scanning for clusters with ongoing anomalies.
  - Package [solitude](https://cran.r-project.org/package=solitude) provides an
    implementation of Isolation forest which detects anomalies purely
    based on the concept of isolation without employing any distance or
    density measure.

**Functional Data**

  - Function `foutliers()` from package
    [rainbow](https://cran.r-project.org/package=rainbow) provides functional
    outlier detection methods. Bagplots and boxplots for functional data
    can also be used to identify outliers, which have either the lowest
    depth (distance from the centre) or the lowest density,
    respectively.
  - Package [adamethods](https://cran.r-project.org/package=adamethods) provides a
    collection of several algorithms to obtain archetypoids with small
    and large databases and with both classical multivariate data and
    functional data (univariate and multivariate). Some of these
    algorithms also allow to detect anomalies.

**Non numeric data**

**Visualization of anomalies**

  - Package [OutliersO3](https://cran.r-project.org/package=OutliersO3) provides
    tools to aid in the display and understanding of patterns of
    multivariate outliers. It uses the results of identifying outliers
    for every possible combination of dataset variables to provide
    insight into why particular cases are outliers.
  - Package [Morpho](https://cran.r-project.org/package=Morpho) provides a
    collection of tools for Geometric Morphometrics and mesh processing.
    Apart from the core functions it provides a graphical interface to
    find outliers and/or to switch mislabeld landmarks.

**Pre-processing methods for anomaly detection**

  - Package [preprocomb](https://cran.r-project.org/package=preprocomb) provides an
    S4 framework for creating and evaluating preprocessing combinations
    for classification, clustering and outlier detection.
  - Package [dobin](https://cran.r-project.org/package=dobin) provides dimension
    reduction technique for outlier detection using neighbours,
    constructs a set of basis vectors for outlier detection. It brings
    outliers to the fore-front using fewer basis vectors.

**Data sets**

  - Package [anomaly](https://cran.r-project.org/package=anomaly) contains
    lightcurve time series data from the Kepler telescope.
  - Various high dimensional datasets are provided by
    [mvoutlier](https://cran.r-project.org/package=mvoutlier).
  - Unlabeled : [cellWise](https://cran.r-project.org/package=cellWise)

**Specific application fields**

  - Package [precintcon](https://cran.r-project.org/package=precintcon) contains
    functions to analyze the precipitation intensity, concentration and
    anomaly.
  - Package [waterData](https://cran.r-project.org/package=waterData) imports U.S.
    Geological Survey (USGS) daily hydrologic data from USGS web
    services and provides functions to calculate and plot anomalies.
  - Package [survBootOutliers](https://cran.r-project.org/package=survBootOutliers)
    provides concordance based bootstrap methods for outlier detection
    in survival analysis.
  - Package [pcadapt](https://cran.r-project.org/package=pcadapt) provides methods
    to detect genetic markers involved in biological adaptation using
    statistical tools based on Principal Component Analysis.
  - Package [rgr](https://cran.r-project.org/package=rgr) supports exploratory data
    analysis with applied geochemical data, with special application to
    the estimation of background ranges and identification of anomalies
    to support mineral exploration and environmental studies.
  - Package [NMAoutlier](https://cran.r-project.org/package=NMAoutlier) implements
    the forward search algorithm for the detection of outlying studies
    (studies with extreme results) in network meta-analysis.

**Miscellaneous**

  - Package [analytics](https://cran.r-project.org/package=analytics) provides
    support for (among other functions) outlier detection in a fitted
    linear model.
  - Package [CircOutlier](https://cran.r-project.org/package=CircOutlier) enables
    detection of outliers in circular-circular regression models,
    modifying its and estimating of models parameters.
  - Package [oclust](https://cran.r-project.org/package=oclust) provides a function
    to detect and trim outliers in Gaussian mixture model based
    clustering using methods described in Clark and McNicholas (2019).
  - Package [semdiag](https://cran.r-project.org/package=semdiag) implements outlier
    and leverage diagnostics for Structural equation modeling.
  - Package [SeleMix](https://cran.r-project.org/package=SeleMix) provides functions
    for detection of outliers and influential errors using a latent
    variable model. A mixture model (Gaussian contamination model) based
    on response(s) y and a depended set of covariates is fit to the data
    to quantify the impact of errors to the
    estimates.

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
  - [cbar](https://cran.r-project.org/package=cbar)
  - [cellWise](https://cran.r-project.org/package=cellWise)
  - [CerioliOutlierDetection](https://cran.r-project.org/package=CerioliOutlierDetection)
  - [CircOutlier](https://cran.r-project.org/package=CircOutlier)
  - [dbscan](https://cran.r-project.org/package=dbscan)
  - [DDoutlier](https://cran.r-project.org/package=DDoutlier)
  - [dixonTest](https://cran.r-project.org/package=dixonTest)
  - [dobin](https://cran.r-project.org/package=dobin)
  - [evtclass](https://cran.r-project.org/package=evtclass)
  - [extremevalues](https://cran.r-project.org/package=extremevalues)
  - [fpmoutliers](https://cran.r-project.org/package=fpmoutliers)
  - [HDoutliers](https://cran.r-project.org/package=HDoutliers)
  - [HighDimOut](https://cran.r-project.org/package=HighDimOut)
  - [ICSOutlier](https://cran.r-project.org/package=ICSOutlier)
  - [kernlab](https://cran.r-project.org/package=kernlab)
  - [kmodR](https://cran.r-project.org/package=kmodR)
  - [ldbod](https://cran.r-project.org/package=ldbod)
  - [mmppr](https://cran.r-project.org/package=mmppr)
  - [modi](https://cran.r-project.org/package=modi)
  - [Morpho](https://cran.r-project.org/package=Morpho)
  - [mrfDepth](https://cran.r-project.org/package=mrfDepth)
  - [mvoutlier](https://cran.r-project.org/package=mvoutlier)
  - [NMAoutlier](https://cran.r-project.org/package=NMAoutlier)
  - [npphen](https://cran.r-project.org/package=npphen)
  - [oclust](https://cran.r-project.org/package=oclust)
  - [otsad](https://cran.r-project.org/package=otsad)
  - [OutlierDetection](https://cran.r-project.org/package=OutlierDetection)
  - [outliers](https://cran.r-project.org/package=outliers)
  - [OutliersO3](https://cran.r-project.org/package=OutliersO3)
  - [outliertree](https://cran.r-project.org/package=outliertree)
  - [pcadapt](https://cran.r-project.org/package=pcadapt)
  - [precintcon](https://cran.r-project.org/package=precintcon)
  - [preprocomb](https://cran.r-project.org/package=preprocomb)
  - [probout](https://cran.r-project.org/package=probout)
  - [rainbow](https://cran.r-project.org/package=rainbow)
  - [RaPKod](https://cran.r-project.org/package=RaPKod)
  - [rgr](https://cran.r-project.org/package=rgr)
  - [Rlof](https://cran.r-project.org/package=Rlof)
  - [Routliers](https://cran.r-project.org/package=Routliers)
  - [RWBP](https://cran.r-project.org/package=RWBP)
  - [scanstatistics](https://cran.r-project.org/package=scanstatistics)
  - [seasonal](https://cran.r-project.org/package=seasonal)
  - [SeleMix](https://cran.r-project.org/package=SeleMix)
  - [semdiag](https://cran.r-project.org/package=semdiag)
  - [sGMRFmix](https://cran.r-project.org/package=sGMRFmix)
  - [SmartSifter](https://cran.r-project.org/package=SmartSifter)
  - [SMLoutliers](https://cran.r-project.org/package=SMLoutliers)
  - [solitude](https://cran.r-project.org/package=solitude)
  - [survBootOutliers](https://cran.r-project.org/package=survBootOutliers)
  - [trendsegmentR](https://cran.r-project.org/package=trendsegmentR)
  - [tsoutliers](https://cran.r-project.org/package=tsoutliers)
  - [univOutl](https://cran.r-project.org/package=univOutl)
  - [washeR](https://cran.r-project.org/package=washeR)
  - [waterData](https://cran.r-project.org/package=waterData)

### Related links:

  - CRAN Task View: [Cluster](Cluster.html)
  - CRAN Task View: [ExtremeValue](ExtremeValue.html)
  - [GitHub repository for this Task
    View](https://github.com/pridiltal/ctv-AnomalyDetection)
