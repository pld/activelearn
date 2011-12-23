set.seed(8675309)

require("lda")

## Use the political blogs data set.
data(poliblog.documents)

data(poliblog.vocab)

data(poliblog.ratings)

num.topics <- 25

docs_n <- length(poliblog.documents)

## calculate topic document score
topic_doc_score <- function(cur_t, d) {
	## counts for words for topic
	topic_word_counts <- t(test.model$topics)[,cur_t]
	## normalize
	if (max(topic_word_counts) > 0)
		topic_word_counts <- topic_word_counts/max(topic_word_counts)
	doc_word_counts <- rep(0, length(topic_word_counts))
	for (i in 1:length(test_docs[[d]][1,])) {
		doc_word_counts[test_docs[[d]][,i][1]]<-test.model$assignments[[d]][i]
	}
	## normalize
	if (max(doc_word_counts) > 0)
		doc_word_counts <- doc_word_counts/max(doc_word_counts)
	return(topic_word_counts%*%doc_word_counts)
}

inc <- 1
trials <- seq(inc,(docs_n-inc),by=25)
## sLDA full in 1, clusters in 2 for 0-1 loss
errors <- matrix(0, length(trials), 4)

index <- 1
for (t in trials) {
	train_n <- t
	test_n <- docs_n - train_n

	train_docs <- poliblog.documents[1:train_n]
	train_ratings <- poliblog.ratings[1:train_n]
	test_docs <- poliblog.documents[(train_n+1):docs_n]
	test_ratings <- poliblog.ratings[(train_n+1):docs_n]/100

	## Initialize the params
	params <- sample(c(-1, 1), num.topics, replace=TRUE)
	
	## we get NaN if for n < 10 num.m.iterations > 1
	num.m.iters <- ifelse(t < 10, 1, 4)
	result <- slda.em(documents=train_docs, K=num.topics, vocab=poliblog.vocab, num.e.iterations=10, num.m.iterations=num.m.iters, alpha=1.0, eta=0.1, train_ratings / 100, params, variance=0.25, lambda=1.0, logistic=FALSE, method="sLDA")

	## predicts with all test
	predictions <- slda.predict(test_docs, result$topics, result$model)
	for (i in 1:length(predictions)) {
		predictions[i] <- ifelse(predictions[i] > 0, 1, -1)
	}

	## 0-1 loss, normalizing will favor large test
	errors[index,2] <- length(which(predictions!=test_ratings))
	## SSE
	errors[index,4] <- sum((predictions-test_ratings)^2)/test_n
	
	## Run LDA on test docs
	test.model <- lda.collapsed.gibbs.sampler(test_docs, num.topics, poliblog.vocab, 50, 0.1, 0.1, compute.log.likelihood=TRUE)

	## And which topic is most expressed by the cited documents.
	max.topics <- apply(test.model$document_sums, 2, which.max)

	## store doc representing topic here
	max.doc_for_topic <- matrix(0,num.topics,2)
	doc_for_topics <- list(rep(0,num.topics))

	## for each doc calculate max topic and score for it
	for (d in 1:test_n) {
		cur_t <- max.topics[d]
		score <- topic_doc_score(cur_t, d)
		if (score > max.doc_for_topic[cur_t,2]) {
			max.doc_for_topic[cur_t,] <- c(d, score)
			doc_for_topics[[cur_t]] <- test_docs[[d]]
		}
	}

	## check if we have any unrepresented topics
	for (i in 1:num.topics) {
		if (class(doc_for_topics[[i]]) == "numeric") {
			## no doc assigned
			cur_t <- i
			for (d in 1:test_n) {
				score <- topic_doc_score(cur_t, d)
				if (score > max.doc_for_topic[cur_t,2]) {
					max.doc_for_topic[cur_t,] <- c(d, score)
					doc_for_topics[[cur_t]] <- test_docs[[d]]
				}
			}
		}
	}

	## predictions with just cluster centers
	predictions <- slda.predict(doc_for_topics, result$topics, result$model)

	for (i in 1:length(predictions)) {
		predictions[i] <- ifelse(predictions[i] > 0, 1, -1)
	}

	full_predictions <- matrix(1,test_n)

	for (d in 1:test_n) {
		cur_t <- max.topics[d]
		full_predictions[d] <- predictions[cur_t]
	}

	## SSE
	errors[index,3] <- sum((full_predictions-test_ratings)^2)/test_n
	## 0-1 loss
	errors[index,1] <- length(which(full_predictions!=test_ratings))

	print(errors[index,])
	index <- index + 1
}


start <- 13
end <- length(trials)
plot(errors[start:end,3], type="o", col="blue", xlab="Number Training Points", ylab="Error Rate", xaxt="n", main="Training Size versus Error")
lines(errors[start:end,4], type="o", col="red")
axis(1, at=seq(1, 19), labels=trials[start:end])
legend("topright", inset=.05, c("Predicted","Clustered"), fill=c("red","blue"))

print("mean clust and pred")
sum(errors[start:end,3])/(end-start)
sum(errors[start:end,4])/(end-start)

print("min clust and pred")
min(errors[start:end,3])
min(errors[start:end,4])

print("max clust and pred")
max(errors[start:end,3])
max(errors[start:end,4])

print("var clust and pred")
var(errors[start:end,3])
var(errors[start:end,4])

## Make a pretty picture.
require("ggplot2")

Topics <- apply(top.topic.words(result$topics, 5, by.score=TRUE), 2, paste, collapse=" ")

coefs <- data.frame(coef(summary(result$model)))

theme_set(theme_bw())

coefs <- cbind(coefs, Topics=factor(Topics, Topics[order(coefs$Estimate)]))

coefs <- coefs[order(coefs$Estimate),]

qplot(Topics, Estimate, colour=Estimate, size=abs(t.value), data=coefs) + geom_errorbar(width=0.5, aes(ymin=Estimate-Std..Error, ymax=Estimate+Std..Error)) + coord_flip()