# NB Code
#####################################################################


# generate a data frame from the list of tweets
require(twitteR)
twtsdf<- twListToDF(abortion_tweets)

twtsdf$hash<- hash

# Drop un-needed variables from the data frame
 keeps <- c("text", "id", "retweetCount", "isRetweet", "screenName", "hash")
twtsdf<- twtsdf[,keeps]

list.vector.words<- list()
allwords<- NULL

names<- NULL
	for (i in 1:dim(twtsdf)[1]){
	  each.vector<- strsplit(twtsdf$text[i], split="")
	  names<- c(names, twtsdf$screenName[i])
	  allwords<- c(allwords, each.vector)
	  list.vector.words[[i]] <- each.vector
}


outcome<- twtsdf$hash

require(tm)
 dat.tm <- Corpus(VectorSource(list.vector.words))	# make a corpus
 dat.tm <- tm_map(dat.tm, tolower)		# convert all words to lowercase
 dat.tm <- tm_map(dat.tm, removePunctuation)			# remove punctuation
 dat.tm <- tm_map(dat.tm, removeWords, words=c("prochoice"))	# remove the hashtags
 dat.tm <- tm_map(dat.tm, removeWords, words=c("prolife")) 	# remove the hashtags
 dat.tm <- tm_map(dat.tm, stripWhitespace)	# remove extra white space
 dat.tm <- tm_map(dat.tm, stemDocument)		# stem all words

# create a bigram tokenizer using the RWeka package
 require(RWeka)
BigramTokenizer<- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
# create the document-term matrix
datmat<- DocumentTermMatrix(dat.tm, control = list(tokenize = BigramTokenizer))
dat<- as.matrix(datmat)
# Add user names as rownames to matrix
rownames(dat) <- names

word.usage<- colSums(dat)
table(word.usage)

# first, set all values in the matrix that are greater than 1 to 1
dat[dat>1] <- 1
threshold <- 9	# set a threshold
tokeep <- which(word.usage>threshold)	# find which column sums are above the threshold
# keep all rows, and only columns with sums greater than the threshold
dat.out<- dat[,tokeep]

# Drop users with few words....
# find how many zeroes are in each row
num.zero <- rowSums(dat.out==0)

# explore data by making a table; can inform choice of cutoff
table(num.zero)
# the number of columns of the document bigram matrix
num_cols <- dim(dat.out)[2]
# users must have used this many bigrams to scale
cutoff <- 2
# create a list of authors to keep
authors_tokeep <- which(num.zero <(num_cols-cutoff))
# keep only users with 2 bigrams
dat.drop <- dat.out[authors_tokeep,]
# similarly, drop those users from the vector of hashtags
outcome <- outcome[authors_tokeep]


require(e1071)
# append the outcome to the first column of dat.drop
myDat <- cbind(outcome, dat.drop)
# turn the doc-term matrix into a data frame
myDat <- as.data.frame(myDat)
# turn the outcome variable (first column) into a factor
myDat[,1] <- as.factor(myDat[,1])

# run the model; save the results to an object
NBmod<- naiveBayes(myDat[,-1], myDat[,1])

# generate a vector of predictions
# arguments: esimated model, predictors, outcome to predict
NBpredictions <- predict(NBmod, myDat[,-1])
# pull out the actual outcomes
actual<- myDat[,1]
# make the confusion matrix
table(NBpredictions, actual, dnn=list("predicted", "actual"))

# run the model on the 1000 labelled instances
NBmod<- naiveBayes(myDat[1:1000,-1], myDat[1:1000,1])
# predict outcomes for the last 100 unlabeled instances
NBpredictions<- predict(NBmod, myDat[1001:1100,-1])
# make a table of the predictions
table(NBpredictions)

