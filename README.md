## CRAN Task View: Anomaly Detection with R

                                                                     
--------------- --------------------------------------------------   
**Maintainer:** Priyanga Dilini Talagala, Rob J. Hyndman             
**Contact:**    Dilini.Talagala at monash.edu                        
**Version:**    2019-10-07                                           
**URL:**        <https://CRAN.R-project.org/view=AnomalyDetection>   

<div>

This CRAN task view contains a list of packages that can be used for
anomaly detection. Anomaly detection problems have many different
facets, and the detection techniques can be highly influenced by the way
we define anomalies, the type of input data to the algorithm, the
expected output, etc. This leads to wide variations in problem
formulations, which need to be addressed through different analytical
approaches.

Anomalies are often mentioned under several alternative names such as
outliers, novelty, odd values, faults, deviants, discordant
observations, extreme values/ cases, change points, events, intrusions,
misuses, exceptions, aberrations, surprises, peculiarities, or
contaminants in different application domains. These variants are also
considered for this view. The packages in this view can be roughly
structured into the following topics.

The deveolpment of this task view is fairly new and still in its early
stages and therefore subject to changes. Please send suggestions for
additions and extensions for this task view to the task view maintainer.

**Specific types of input data**

  - *High Dimensional Data* : Local anomaly, Global anomaly, Micro
    cluster
  - *Temporal Data* : Streaming data, Batch Data  
    Point anomaly, contexual anomaly, collective anomaly, anomalous
    series within a large collection of time series
  - *Functional Data* :
  - *Spacio-Temporal data* :
  - *Non numeric data*

**Modeling Approches**

  - *Distance based* : Nearest neighbour based techniques
  - *Density based* :
  - *Angle based* :
  - *Depth based* :
  - *Classification based* :
  - *Clustering based* :
  - *Prjection Pursuit based* :
  - *Distribution based* :

**Nature of the Methods**

  - *Supervised* :
  - *Unsupervised* :
  - *Semi-supervised* :

**Anomalous threshold calculation methos**

  - *User defined threshold* :
  - *Extreme values theory based apparoach* :

**Output of an algorithm**

  - *Scores* :
  - *Lables* :

**Visualization of anomalies**

  - [OutliersO3](https://cran.r-project.org/package=OutliersO3)

**Anomaly detection data**

  - *Labeled* :
  - *Unlabeled* :

**Specific application
    fields**

</div>

### CRAN packages:

  - [abodOutlier](https://cran.r-project.org/package=abodOutlier)
  - [adamethods](https://cran.r-project.org/package=adamethods)
  - [alphaOutlier](https://cran.r-project.org/package=alphaOutlier)
  - [altmeta](https://cran.r-project.org/package=altmeta)
  - [amelie](https://cran.r-project.org/package=amelie)
  - [analysisPipelines](https://cran.r-project.org/package=analysisPipelines)
  - [analytics](https://cran.r-project.org/package=analytics)
  - [ANN2](https://cran.r-project.org/package=ANN2)
  - [anomaly](https://cran.r-project.org/package=anomaly)
  - [anomalyDetection](https://cran.r-project.org/package=anomalyDetection)
  - [anomazlize](https://cran.r-project.org/package=anomazlize)
  - [BAS](https://cran.r-project.org/package=BAS)
  - [Benchmarking](https://cran.r-project.org/package=Benchmarking)
  - [biogeo](https://cran.r-project.org/package=biogeo)
  - [carx](https://cran.r-project.org/package=carx)
  - [cbar](https://cran.r-project.org/package=cbar)
  - [cellWise](https://cran.r-project.org/package=cellWise)
  - [CerioliOutlierDetection](https://cran.r-project.org/package=CerioliOutlierDetection)
  - [CircOutlier](https://cran.r-project.org/package=CircOutlier)
  - [cmsaf](https://cran.r-project.org/package=cmsaf)
  - [compositions](https://cran.r-project.org/package=compositions)
  - [CoordinateCleaner](https://cran.r-project.org/package=CoordinateCleaner)
  - [CORElearn](https://cran.r-project.org/package=CORElearn)
  - [ctmm](https://cran.r-project.org/package=ctmm)
  - [CVXR](https://cran.r-project.org/package=CVXR)
  - [daewr](https://cran.r-project.org/package=daewr)
  - [dave](https://cran.r-project.org/package=dave)
  - [dbscan](https://cran.r-project.org/package=dbscan)
  - [ddalpha](https://cran.r-project.org/package=ddalpha)
  - [DDoutlier](https://cran.r-project.org/package=DDoutlier)
  - [depth.plot](https://cran.r-project.org/package=depth.plot)
  - [DescTools](https://cran.r-project.org/package=DescTools)
  - [DJL](https://cran.r-project.org/package=DJL)
  - [dlookr](https://cran.r-project.org/package=dlookr)
  - [DMwR](https://cran.r-project.org/package=DMwR)
  - [DMwR2](https://cran.r-project.org/package=DMwR2)
  - [doex](https://cran.r-project.org/package=doex)
  - [drsmooth](https://cran.r-project.org/package=drsmooth)
  - [dsa](https://cran.r-project.org/package=dsa)
  - [enpls](https://cran.r-project.org/package=enpls)
  - [EnvStats](https://cran.r-project.org/package=EnvStats)
  - [evtclass](https://cran.r-project.org/package=evtclass)
  - [extremevalues](https://cran.r-project.org/package=extremevalues)
  - [FactoInvestigate](https://cran.r-project.org/package=FactoInvestigate)
  - [faoutlier](https://cran.r-project.org/package=faoutlier)
  - [fda.usc](https://cran.r-project.org/package=fda.usc)
  - [fdasrvf](https://cran.r-project.org/package=fdasrvf)
  - [forecast:tsoutlier](https://cran.r-project.org/package=forecast:tsoutlier)
  - [fpmoutliers](https://cran.r-project.org/package=fpmoutliers)
  - [fsdaR](https://cran.r-project.org/package=fsdaR)
  - [funModeling](https://cran.r-project.org/package=funModeling)
  - [fwdmsa](https://cran.r-project.org/package=fwdmsa)
  - [gcookbook](https://cran.r-project.org/package=gcookbook)
  - [ggformula](https://cran.r-project.org/package=ggformula)
  - [ggplot2](https://cran.r-project.org/package=ggplot2)
  - [ggpol](https://cran.r-project.org/package=ggpol)
  - [ggstane](https://cran.r-project.org/package=ggstane)
  - [GmAMisc](https://cran.r-project.org/package=GmAMisc)
  - [h2o](https://cran.r-project.org/package=h2o)
  - [HDoutliers](https://cran.r-project.org/package=HDoutliers)
  - [HighDimOut](https://cran.r-project.org/package=HighDimOut)
  - [hotspots](https://cran.r-project.org/package=hotspots)
  - [ICSOutlier](https://cran.r-project.org/package=ICSOutlier)
  - [jmotif](https://cran.r-project.org/package=jmotif)
  - [kmodR](https://cran.r-project.org/package=kmodR)
  - [ldbod](https://cran.r-project.org/package=ldbod)
  - [MALDIrppa](https://cran.r-project.org/package=MALDIrppa)
  - [mbgraphic](https://cran.r-project.org/package=mbgraphic)
  - [MCMC.OTU](https://cran.r-project.org/package=MCMC.OTU)
  - [MCMC.qpcr](https://cran.r-project.org/package=MCMC.qpcr)
  - [metafor](https://cran.r-project.org/package=metafor)
  - [metaplus](https://cran.r-project.org/package=metaplus)
  - [metR](https://cran.r-project.org/package=metR)
  - [MIPHENO](https://cran.r-project.org/package=MIPHENO)
  - [modi](https://cran.r-project.org/package=modi)
  - [MoEClust](https://cran.r-project.org/package=MoEClust)
  - [mrfDepth](https://cran.r-project.org/package=mrfDepth)
  - [muma](https://cran.r-project.org/package=muma)
  - [MVN](https://cran.r-project.org/package=MVN)
  - [mvoutlier](https://cran.r-project.org/package=mvoutlier)
  - [Ncmisc](https://cran.r-project.org/package=Ncmisc)
  - [nlr](https://cran.r-project.org/package=nlr)
  - [NMAoutlier](https://cran.r-project.org/package=NMAoutlier)
  - [npphen](https://cran.r-project.org/package=npphen)
  - [oce](https://cran.r-project.org/package=oce)
  - [OutlierDC](https://cran.r-project.org/package=OutlierDC)
  - [OutlierDetection](https://cran.r-project.org/package=OutlierDetection)
  - [OutlierDM](https://cran.r-project.org/package=OutlierDM)
  - [outliers](https://cran.r-project.org/package=outliers)
  - [OutliersO3](https://cran.r-project.org/package=OutliersO3)
  - [pcadapt](https://cran.r-project.org/package=pcadapt)
  - [PMCMRplus](https://cran.r-project.org/package=PMCMRplus)
  - [precintcon](https://cran.r-project.org/package=precintcon)
  - [probout](https://cran.r-project.org/package=probout)
  - [psych](https://cran.r-project.org/package=psych)
  - [qpcR](https://cran.r-project.org/package=qpcR)
  - [quokar](https://cran.r-project.org/package=quokar)
  - [RaceID](https://cran.r-project.org/package=RaceID)
  - [Rainbow](https://cran.r-project.org/package=Rainbow)
  - [RaPKod](https://cran.r-project.org/package=RaPKod)
  - [rapportools](https://cran.r-project.org/package=rapportools)
  - [rAverage](https://cran.r-project.org/package=rAverage)
  - [referenceIntervals](https://cran.r-project.org/package=referenceIntervals)
  - [REPPlab](https://cran.r-project.org/package=REPPlab)
  - [Rfast](https://cran.r-project.org/package=Rfast)
  - [rgam](https://cran.r-project.org/package=rgam)
  - [Rlof](https://cran.r-project.org/package=Rlof)
  - [robCompositions](https://cran.r-project.org/package=robCompositions)
  - [robfilter](https://cran.r-project.org/package=robfilter)
  - [robmixglm](https://cran.r-project.org/package=robmixglm)
  - [robustarima](https://cran.r-project.org/package=robustarima)
  - [robustbase](https://cran.r-project.org/package=robustbase)
  - [rrcov](https://cran.r-project.org/package=rrcov)
  - [rrcovHD](https://cran.r-project.org/package=rrcovHD)
  - [rucrdtw](https://cran.r-project.org/package=rucrdtw)
  - [RWBP](https://cran.r-project.org/package=RWBP)
  - [s2dverification](https://cran.r-project.org/package=s2dverification)
  - [scanstatistics](https://cran.r-project.org/package=scanstatistics)
  - [SCORPIUS](https://cran.r-project.org/package=SCORPIUS)
  - [seasonal](https://cran.r-project.org/package=seasonal)
  - [seawaveQ](https://cran.r-project.org/package=seawaveQ)
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
  - [TSA](https://cran.r-project.org/package=TSA)
  - [tsoutliers](https://cran.r-project.org/package=tsoutliers)
  - [univOutl](https://cran.r-project.org/package=univOutl)
  - [washeR](https://cran.r-project.org/package=washeR)
  - [waterData](https://cran.r-project.org/package=waterData)
  - [wql](https://cran.r-project.org/package=wql)
  - [xray](https://cran.r-project.org/package=xray)

### Related links:

  - CRAN Task View: [Cluster](Cluster.html)
  - CRAN Task View: [ExtremeValue](ExtremeValue.html)
  - [GitHub repository for this Task
    View](https://github.com/pridiltal/ctv-AnomalyDetection)
