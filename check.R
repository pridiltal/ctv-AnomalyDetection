
## Modified Rob J. Hyndman's code by Priyanga Dilini Talagala for use with AnomalyDetection.ctv

## Check ctv is correct

library(ctv)

ctv2html("MyTopic.md")
browseURL("MyTopic.html")
htmlfile <- here::here(paste0(ctv, ".html"))

# Set CRAN mirror
r <- getOption("repos")
r["CRAN"] <- "https://cloud.r-project.org"
options(repos = r)

# Run the check
check_ctv_packages(ctvfile)

# Create html file from ctv file
ctv2html(read.ctv(ctvfile), htmlfile)

# View html file
browseURL(htmlfile)

cat("Done.\n")
