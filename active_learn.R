## scratch code
set.seed(8675309)

## load libraries
require("tm")
require("lda")

## load data
data("acq")
data("crude")

## form corpus C^*
corpus <- c(acq, crude)

## clean data
removeStopWords <- function(x) removeWords(x, stopwords("en"))
funs = list(removePunctuation, removeStopWords, removeNumbers)

corpus <- tm_map(corpus, FUN=tm_reduce, tmFuns=funs)

topics = list()
for (i in 1:length(corpus)) {
	topics[i] <- list(strsplit(meta(corpus[[i]], tag="Topics"), " "))
}

## format for LDA
corp_lda <- lexicalize(corpus, lower=TRUE)

## parameters for LDA
K <- 10 ## number of topics
num.iterations <- 25
alpha <- 0.1
eta <- 0.1

## run LDA Gibbs
result_lda <- lda.collapsed.gibbs.sampler(corp_lda$documents, K, corp_lda$vocab, num.iterations, alpha, eta, compute.log.likelihood=TRUE)

## Get the top words in the cluster
top.words <- top.topic.words(result_lda$topics, 5, by.score=TRUE)
top.words