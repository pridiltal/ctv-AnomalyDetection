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
outliers, novelty, odd values in different application domains. These
variants are also considered for this view.

The deveolpment of this task view is fairly new and still in its early
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
  - [outliers](https://cran.r-project.org/package=outliers) provides a collection of
    some tests commonly used for identifying outliers. For most
    functions the input is a numeric vector. If argument is a dataframe,
    then outlier is calculated for each column by sapply. The same
    behavior is applied by apply when the matrix is given.
  - *Global anomaly* :
  - *Micro cluster* :

**Temporal Data**

  - Streaming data, Batch Data  
    Point anomaly, contexual anomaly, collective anomaly, anomalous
    series within a large collection of time series

**Functional Data**

**Spacio-Temporal data**

**Non numeric data**

**Visualization of anomalies**

  - [OutliersO3](https://cran.r-project.org/package=OutliersO3)

**Data sets**

  - *Labeled* :
  - *Unlabeled* :

**Specific application fields**

</div>

### CRAN packages:

  - [outliers](https://cran.r-project.org/package=outliers)
  - [OutliersO3](https://cran.r-project.org/package=OutliersO3)

### Related links:

  - CRAN Task View: [Cluster](Cluster.html)
  - CRAN Task View: [ExtremeValue](ExtremeValue.html)
  - [GitHub repository for this Task
    View](https://github.com/pridiltal/ctv-AnomalyDetection)
