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
outliers, novelty, odd values, extreme values in different application
domains. These variants are also considered for this view.

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
  - Functions `LOF()` and `GLOSH` in package
    [dbscan](https://cran.r-project.org/package=dbscan) provides density based
    anomaly detection methods using a kd-tree to speed up kNN search.
  - Package [ldbod](https://cran.r-project.org/package=ldbod) provides flexible
    functions for computing local density-based outlier scores. It
    allows for subsampling of input data or a user specified reference
    data set to compute outlier scores against, so both unsupervised and
    semi-supervised outlier detection can be done.
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

**Spatio-Temporal data**

  - Scan statistics are used to detect anomalous clusters in spatial or
    space-time data. Package
    [scanstatistics](https://cran.r-project.org/package=scanstatistics) provides
    functions for detection of anomalous space-time clusters using the
    scan statistics methodology. Focuses on prospective surveillance of
    data streams, scanning for clusters with ongoing anomalies.

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

  - [OutliersO3](https://cran.r-project.org/package=OutliersO3)

**Pre-processing methods for anomaly detection**

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

</div>

### CRAN packages:

  - [adamethods](https://cran.r-project.org/package=adamethods)
  - [alphaOutlier](https://cran.r-project.org/package=alphaOutlier)
  - [amelie](https://cran.r-project.org/package=amelie)
  - [anomalize](https://cran.r-project.org/package=anomalize)
  - [anomaly](https://cran.r-project.org/package=anomaly)
  - [cbar](https://cran.r-project.org/package=cbar)
  - [cellWise](https://cran.r-project.org/package=cellWise)
  - [dbscan](https://cran.r-project.org/package=dbscan)
  - [dobin](https://cran.r-project.org/package=dobin)
  - [extremevalues](https://cran.r-project.org/package=extremevalues)
  - [fpmoutliers](https://cran.r-project.org/package=fpmoutliers)
  - [HighDimOut](https://cran.r-project.org/package=HighDimOut)
  - [ICSOutlier](https://cran.r-project.org/package=ICSOutlier)
  - [kernlab](https://cran.r-project.org/package=kernlab)
  - [kmodR](https://cran.r-project.org/package=kmodR)
  - [ldbod](https://cran.r-project.org/package=ldbod)
  - [mvoutlier](https://cran.r-project.org/package=mvoutlier)
  - [OutlierDetection](https://cran.r-project.org/package=OutlierDetection)
  - [outliers](https://cran.r-project.org/package=outliers)
  - [OutliersO3](https://cran.r-project.org/package=OutliersO3)
  - [precintcon](https://cran.r-project.org/package=precintcon)
  - [rainbow](https://cran.r-project.org/package=rainbow)
  - [Routliers](https://cran.r-project.org/package=Routliers)
  - [scanstatistics](https://cran.r-project.org/package=scanstatistics)
  - [sGMRFmix](https://cran.r-project.org/package=sGMRFmix)
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
