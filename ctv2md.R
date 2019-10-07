#!/usr/bin/r
## if you do not have /usr/bin/r from littler, just use Rscript
## 
## Copyright 2014 - 2017  Dirk Eddelbuettel 
## Released under GPL-2 or later
## Modified by Priyanga Dilini Talagala for use with AnomalyDetection.ctv

ctv <- "AnomalyDetection"

ctvfile  <- paste0(ctv, ".ctv")
htmlfile <- paste0(ctv, ".html")
#mdfile   <- paste0(ctv, ".md")
mdfile   <- "README.md"

## load packages
suppressMessages(library(XML))          # called by ctv
suppressMessages(library(ctv))

r <- getOption("repos")                 # set CRAN mirror
r["CRAN"] <- "https://cloud.r-project.org"
options(repos=r)

check_ctv_packages(ctvfile)             # run the check

## create html file from ctv file
ctv2html(read.ctv(ctvfile), htmlfile)

### these look atrocious, but are pretty straight forward. read them one by one
###  - start from the htmlfile
cmd <- paste0("cat ", htmlfile,
###  - in lines of the form  ^<a href="Word">Word.html</a>
###  - capture the 'Word' and insert it into a larger URL containing an absolute reference to task view 'Word'
              " | sed -e 's|^<a href=\"\\([a-zA-Z]*\\)\\.html|<a href=\"https://cran.r-project.org/web/views/\\1.html\"|' | ",
###  - call pandoc, specifying html as input and github-flavoured markdown as output
###    (use 'gfm' for pandoc 2.*, and 'markdown_github' pandoc 1.*)
              "pandoc -s -r html -w gfm | ",
###  - deal with the header by removing extra ||, replacing |** with ** and **| with **:
              "sed -e's/||//g' -e's/|\\*\\*/\\*\\*/g' -e's/\\*\\*|/\\*\\* /g' -e's/|$/  /g' ",
###  - remove the table: remove the '| ' vertical bar, and remove the frame line
              "-e's/| //g' -e'/^|-----/d' ",
###  - make the implicit URL to packages explicit
              "-e's|../packages/\\([^/]*\\)/index.html|https://cran.r-project.org/package=\\1|g' ",
              "-e's|../packages/\\([^/]*\\)|https://cran.r-project.org/package=\\1|g' ",
              "-e's/( \\[/(\\[/g' ",
###  - write out mdfile
              "> ", mdfile)

system(cmd)                             # run the conversion

unlink(htmlfile)                        # remove temporary html file

cat("Done.\n")
