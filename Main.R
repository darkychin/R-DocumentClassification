## Task 1
## File Classification Training

## library 
# require("readtext") [call both install.packages("x") then library(x)]
install.packages("readtext") 
library(readtext)

## -----------------------------------------------------------


## Setting file directory [rerun it every time in new instance]
datdir =setwd("C:/Users/Junbo/Desktop/Darky_stuffs/R_Darky/TextFileClassification")

## -----------------------------------------------------------


## read files [correct]
files=readtext(paste0(datdir,"/Petronas Data/POC_Form/*/*.txt"))
## read files ver2 to extract file names  [undoable - because of different filename length]
# files=readtext(paste0(datdir,"/Petronas Data/POC_Form/*/*.txt"),
#                docvarsfrom = "filenames", 
#                docvarnames = c("no", "name", "year"),
#                dvsep = "_")
## read files ver3 to extract file in a cleaner way [wrong, text_field is wrongly used]
# files=readtext(paste0(datdir,"/Petronas Data/POC_Form/*/*",
#                       text_field=".txt"))

## check presence of pdf [0 is correct] 
grep("?.pdf$",files$doc_id)

## save files image as rds
saveRDS(files,"readFiles.rds")
## load file image from rds
files = readRDS("readFiles.rds")

## -----------------------------------------------------------


## Function :: Trim unwanted "path/" in doc_id
customTrim <- function(string,re){
    name=unlist(strsplit(string,re))
    return(name[length(name)])
    ## for list
    # name=strsplit(string,re)
    # return(name[[1]][length(name[[1]])]) 
}

## split filenames and remove annoying file row name
filesMatrix= unname(sapply(files$doc_id,customTrim,"/"))

## all to lower case
filesMatrix=unname(sapply(filesMatrix,tolower))
filesText=unname(sapply(files$text,tolower))

## -----------------------------------------------------------


## Get file category by splitting filenames
## Function :: return category name
category = function(string) {
    cat = unlist(strsplit(string,"_"))[2]
    return(cat)
}

filesCategory= unname(sapply(filesMatrix,category))

# Discovering data [discover file category]
filesCategoryTest=unique(filesCategory)
filesCategoryTest[order(filesCategoryTest,decreasing = FALSE)]

## -----------------------------------------------------------


## Cleaning empty category name and empty text data
## Combing filesCategory, filesText
filesTmp = as.data.frame(cbind(category=filesCategory, text=filesText),stringsAsFactors = FALSE)
# as.data.frame(cbind(doc_id=filesCategory, text=filesText))

##           remove "" in category              ""  in text
filesTmp = filesTmp[(filesTmp$category!="" & filesTmp$text!=""),]

#summary(filesUpdated)
## now filesTmp will be the complete file table [without ""]
saveRDS(filesTmp, "readFiles_complete.rds")
filesTmp= readRDS("readFiles_complete.rds")

## -----------------------------------------------------------


## Building Corpus
## 1. using tm
library(tm)
corpus_tm = VCorpus(VectorSource(filesTmp$text), readerControl = list(language="en")) #English
inspect(corpus_tm)
# head(corpus_tm)
# summary(corpus_tm)
# corpus_tm

## 2. using quanteda
install.packages("quanteda")
library(quanteda)
corpus_qtd = corpus(filesText)
summary(corpus_qtd)

## -----------------------------------------------------------

## Tranformation/ Preprocessing then Generate DTM/DFM

## custom functions:
removeURL <- function(x) gsub("http[[:alnum:][:punct:]]*", "", x) 
removeNonAlphNum <- function(x) gsub("[^[:alnum:]]","",x)
removeSpecialChars <- function(x) gsub("•","",x)

## using tm library::    notes: cleaning starts from specific -> general
##  1. remove all URLs 
corpus_tm = tm_map(corpus_tm, content_transformer(removeURL))
##  2. remove all non alphabet and numeric chars    
#corpus_tm2 = tm_map(corpus_tm, content_transformer(removeNonAlphNum))[X causing space and newLine to be missing]
#corpus_tm2 = gsub("[^[:alnum:]]"," ", corpus_tm)
#corpus_tm = tm_map(corpus_tm, content_transformer(function(x) gsub("[^[:alnum:]]"," ",x)))
##  3. remove stopwords
corpus_tm = tm_map(corpus_tm, removeWords, stopwords("english"))
##  4. remove numbers
corpus_tm = tm_map(corpus_tm, removeNumbers)
##  5. remove whitespace 
corpus_tm = tm_map(corpus_tm,stripWhitespace)
##  6. remove punctuation
corpus_tm = tm_map(corpus_tm,removePunctuation)
##  7. remove special characters
corpus_tm = tm_map(corpus_tm,content_transformer(removeSpecialChars))
##  8. change all words to stem words
corpus_tm = tm_map(corpus_tm, stemDocument, language="english")
##   note: content_transformer is required for non-tm functions
##  9. to lower case [already done, see top]
corpus_tm = tm_map(corpus_tm, content_transformer(tolower))

## 10. creating Document Term Matrix
dtm_tm = DocumentTermMatrix(corpus_tm)
saveRDS(dtm_tm,"dtm(tm).rds")
dtm_tm = readRDS("dtm(tm).rds")
## 11. sparsing the table to reduce noises
dtm_tm2 = removeSparseTerms(dtm_tm,0.99)


## using quanteda library::
##  1. cleaning and tokenize // not including to lower case
tokens_qtd = tokens(corpus_qtd,what="word",
                           remove_numbers=TRUE,remove_punct=TRUE,
                           remove_symbols=TRUE,remove_separators=TRUE,
                           remove_twitter=TRUE,remove_hyphens=TRUE,
                           remove_url=TRUE)

## 2. cleaning and create Document Feature Matrix
dfm_qtd = dfm(corpus_qtd_tokens,tolower=TRUE,stem=TRUE,remove=stopwords("english"))
summary(corpus_qtd_dfm)
## 3. sparsing Document Frequency Matrix
dfm_qtd2=dfm_trim(corpus_qtd_dfm,min_count=0.99)

## -----------------------------------------------------------


## the chosen library is tm and the following script is the following of it

## Using tm::

## 1. converting dtm to dataframe
dtm_df_tm <- as.data.frame(as.matrix(dtm_tm2))
## 2. cleanup column names to remove special characters!
colnames(dtm_df_tm)=make.names(colnames(dtm_df_tm))
## 3. make a processed complete dataframe
finaldtmdf_tm = as.data.frame(cbind(category=filesTmp$category,dtm_df_tm))
## 4. save and read RDS
saveRDS(finaldtmdf_tm,"finaldtmdf_tm.rds")
finaldtmdf_tm = readRDS("finaldtmdf_tm.rds")

## train test split [option 1]
## full train full test [option 2] <- is chosen

## Models Selections::

## notes: require(caret) for confusionMatrix

## 1)  kNN model [kernel estimation]
## pros:
## - high accuracy 
## - 
## con:
## - need to determine a meaningful "k" value
## - require high computational power
## side note: kmean(the unlying str of kNN model)

## import library
require("class")

## train test split [option 1: example]
## 1. reproducible random
set.seed(1) 
## 2. make row index
ind = 1:nrow(filesTmp)
## 3. generate split the data index
trainIndex= createDataPartition(ind,p=0.8,list=FALSE)
## 4. prepare train, test data index
train= dtm_df_tm[trainIndex,]
test = dtm_df_tm[-trainIndex,]
## 5. assign train, test category data 
trainCat = filesTmp$category[trainIndex]
testCat = filesTmp$category[-trainIndex]
## 6. create prediction model and get prediction
knn_pred_tm = knn(train, test, trainCat, k=70) ## it return prediction directly
## 7. set common levels
levels(knn_pred_tm)= levels(finaldtmdf_tm$category)
levels(testCat) = levels(finaldtmdf_tm$category)
## side note: use "levels" when factor level is different
## 8. accuracy test [confusion Matrix]
knn_cm_tm = confusionMatrix(knn_pred_tm,testCat) ##undoable, because difference in column
## 8. accuracy test [customize accuracy check with mean]
mean(knn_cm_tm==testCat)*100

## full train full test [option 2: example]
## 1. create prediction model and get prediction
knn_pred_tm = knn(dtm_df_tm,dtm_df_tm,filesTmp$category,k=70)
## 2. accuracy test [confusion Matrix]
knn_cm_tm=confusionMatrix(knn_pred_tm,(filesTmp$category))

## After Test Result [with full train full test]
## - [70% accuracy]
## - (model) not producible and can only generate final prediction
## - slow -> [document classification, 5981 rows,111 categories, 1 hour]
## Conclusion: not suitable
## not good in this scenario, too slow when creating model, and not reproducible


## 2) c5.0 model [decision tree]
## pros:
## - 
## con:
## - 

## import library
install.packages("C50")
library(C50)

## full train full test [option 2: example]
## 1. create prediction model
c50model_tm= C5.0(category~.,finaldtmdf_tm)
## 2. create prediction
pred_c50_tm= predict(c50model_tm,finaldtmdf_tm)
## 3. accuracy test [confusion Matrix]
cm_c50_tm = confusionMatrix(pred_c50_tm,finaldtmdf_tm) # unable to do

## After Test Result [with full train full test]
## - return msg [unable to produce tree(exit with value 1)]
## - confusion matrix is not available because of the tree
## - fast -> [document classification, 5981 rows,111 categories, <15 minutes]
## Conclusion: not suitable
## unable to create a satisfy testing


## 3) randomForest model [decision tree]
## pros:
## - Accurate
## con:
## - Super slow

## import library
library(randomForest)

## full train full test [option 2: example]
## 1. create prediction model [run in 20 minutes and stop]
rfmodel_tm = randomForest(category~. ,finaldtmdf_tm)
## 2. create prediction
rfpred_tm = predict(rfmodel_tm,finaldtmdf_tm)
## 3. accuracy test [confusion Matrix]
rfcm_tm = confusionMatrix(rfpred_tm,finaldtmdf_tm$category)

## After Test Result [with full train full test]
## - 97% accuracy
## - slow -> [document classification, 5981 rows,111 categories, 1 hour]
## Hypothesis:
## - will be super slow, so i didn't try
## Conclusion: not suitable
## - the result is overfit, so it is not suitable


## 4) ctree [decision tree]
## pros:
## - 
## con:
## - 

## import library
library(party)

## full train full test [option 2: example]
## 1. create prediction model
ctmodel_tm = ctree(category~. ,finaldtmdf_tm)
## 2. create prediction
pred_ct_tm = predict(ctmodel_tm,finaldtmdf_tm)
## 3. accuracy test [confusion Matrix]
cm_ct_tm = confusionMatrix(pred_ct_tm,finaldtmdf_tm$category)

## After Test Result [with full train full test]
## -only have 20.8% accuracy
## - moderate -> [document classification, 5981 rows,111 categories, <30 minutes]
## Conclusion: not suitable
## - accuracy is too low in this scenario


## 5) k-mean [clustering, unsupervised learning]
## pros:
## - 
## con:
## - 

## import library
library(stats)

## full train full test [option 2: example]
## 1. create prediction model
kmmodel_tm = kmeans(x=subset(finaldtmdf_tm,select=-category),center=111)
## 2. create prediction
table(kmmodel_tm$cluster,finaldtmdf_tm$category) # prediction is in cluster

## After Test Result [with full train full test]
## -
## Conclusion: not suitable
## - clustering is created without names, so umcomparable


## 6) svm model []
## pros:
## - highest accuracy in text classification
## con:
## - 

## import library
library(e1071)

## train test split [option 1: example]
## 1. reproducible random
set.seed(1)
## 2. make row index
ind = 1:nrow(finaldtmdf_tm)
## 3. generate split the data index
trainIndex= createDataPartition(ind, p=0.85,list=FALSE)
## 4.assign train, test category data 
train = finaldtmdf_tm[trainIndex,]
test = finaldtmdf_tm[-trainIndex,]
## 5. create prediction model
svmmodeltts_tm = svm(category~.,train)
## 6. create prediction
pred_svmtts_tm = predict(svmmodeltts_tm,test)
#pred_svmtts_tm = predict(svmmodeltts_tm,test[1,])
## 7. accuracy test [confusion Matrix]
cm_svmtts_tm = confusionMatrix(pred_svmtts_tm,test$category)

## After Test Result [with 0.85 train 0.15 test]
## - 80% accuracy
## - fast -> [document classification, 5981 rows,111 categories, <15 minutes]
## Conclusion: Chosen
## - Best speed and accuracy, is chosen!

## full train full test [option 2: example]
## 1. create prediction model
svmmodel_tm = svm(category~., finaldtmdf_tm)
## 1.1 save and read model
saveRDS(svmmodel_tm,"svmModel.rds")
svmmodel_tm = readRDS("svmModel.rds")
## 2. create prediction
pred_svm_tm = predict(svmmodel_tm, finaldtmdf_tm)
## 2.1 save and read prediction
saveRDS(pred_svm_tm, "svmPrediction.rds")
pred_svm_tm = readRDS("svmPrediction.rds")
## 3. accuracy test [confusion Matrix]
cm_svm_tm = confusionMatrix(pred_svm_tm, finaldtmdf_tm$category)

## After Test Result [with full train full test]
## - 86% accuracy
## - fast -> [document classification, 5981 rows,111 categories, 15 minutes]
## Conclusion: Chosen
## - Best speed and accuracy, is chosen!

## -----------------------------------------


## Task 2: 
## improve accuracy with train on train
## a) create a lookup table( for user reference only)
## sum all of the factors which are wrongly predicted

##------------------------- lookup table function DRAFTS ---------
x = data.frame(stringsAsFactors = FALSE)
xnames=c("success","fail","sugg1","occur1","freq1","sugg2","occur2","freq2","sugg3","occur3","freq3","sugg4","occur4","freq4")
for(i in  1:nrow(testing2.1Agg)){
    y = data.frame(stringsAsFactors = FALSE)
   
    subset1 = subset(testing2.1,testing2.1$prediction==testing2.1Agg$prediction[i])
    success=subset1$Freq[1]/testing2.1Agg$total[i] *100
    fail = 100 - success
    
    y= c(y,success,fail)
    
    for (j in 2:5){
        ##suggestion = levels(testing2.1$sample)[(subset1$sample[j])]
        suggestion = as.character(subset1$sample)[j]
        occurance = subset1$Freq[j]/testing2.1Agg$total[i] *100
        frequency = subset1$Freq[j]
        y=c(y,suggestion,occurance,frequency)
    }
    
    names(y)= xnames
    #rowNames(y)= testing2.1Agg$prediction[i] 
    x = rbind.data.frame(x,y,stringsAsFactors = FALSE)
   
}
z=cbind(prediction=testing2.1Agg$prediction,x)
# return(z)

## found way to create the table
testing=as.data.frame(table(prediction=pred_svm_tm,sample=finaldtmdf_tm$category))
testing2=subset(testing,testing$Freq!=0)
testing2.1=testing2[order(testing2$prediction,-testing2$Freq),]
testing2.1Agg = setNames(aggregate(testing2.1$Freq , by = list(testing2.1$prediction), FUN=sum),c("prediction","total"))
## testing3 not required for now
testing3=subset(testing,testing$Freq!=0 & testing$prediction!=testing$sample)
testing3.1 = testing3[order(testing3$prediction,-testing3$Freq),]
## Testing
z.1 = createLookUpTB(pred_svm_tm,finaldtmdf_tm$category) ## Successful
##------------------------- DRAFTS END---------

predictionTable = as.data.frame(table(pred_svm_tm,finaldtmdf_tm$category))

createLookUpTB <- function(pred,samp){
    tb=as.data.frame(table(prediction=pred,sample=samp))
    tb=subset(tb,tb$Freq!=0)
    tb=tb[order(tb$prediction,-tb$Freq),]
    aggtb = setNames(aggregate(tb$Freq, by = list(tb$prediction), FUN =sum),c("prediction","total"))
    lookuptb = data.frame()
    lookupColNames = c("success","fail","sugg1","occur1","freq1","sugg2","occur2","freq2","sugg3","occur3","freq3","sugg4","occur4","freq4")
    
    for(i in  1:nrow(aggtb)){
        row = data.frame()
        subset1 = subset(tb,tb$prediction==aggtb$prediction[i])
        success=subset1$Freq[1]/aggtb$total[i] *100
        fail = 100 - success
        row = c(row,success,fail)
        for(j in 2:5){
            suggestion = as.character(subset1$sample)[j]
            occurance = subset1$Freq[j]/aggtb$total[i] *100
            frequency = subset1$Freq[j]
            row = c(row,suggestion,occurance,frequency)
        }
        names(row) = lookupColNames
        lookuptb = rbind.data.frame(lookuptb,row,stringsAsFactors = FALSE)
    }
    
    lookuptb = cbind.data.frame(prediction=aggtb$prediction,lookuptb)
    return(lookuptb)
}
## generate Look up table
lookupTable = createLookUpTB(pred_svm_tm,finaldtmdf_tm$category)

## b) construct a suggestion table
##----------------------- DRAFTS --------------------------------------------------
testing4.1 = setNames(aggregate(lookupTable$freq1, by =list(lookupTable$sugg1),FUN=sum),c("sugg","freq"))
testing4.2 = setNames(aggregate(lookupTable$freq2, by =list(lookupTable$sugg2),FUN=sum),c("sugg","freq"))
testing4.3 = setNames(aggregate(lookupTable$freq3, by =list(lookupTable$sugg3),FUN=sum),c("sugg","freq"))
testing4.4 = setNames(aggregate(lookupTable$freq4, by =list(lookupTable$sugg4),FUN=sum),c("sugg","freq"))
testing4.45 = rbind(testing4.1,testing4.2,testing4.3,testing4.4)
testing.final = setNames(aggregate(testing4.45$freq, by=list(testing4.45$sugg),FUN=sum),c("sugg","freq"))
testing.final2 = testing.final[(order(-testing.final$freq)),]

y=createSuggTB(lookupTable)

test= setNames(as.data.frame(pred_svm_tm),"prediction")
test$id = 1:nrow(test)

test=merge(test,suggestionTable,by="prediction",all=TRUE)
test2 = test[order(test$id),]

test3 = test2[,c(1,3:6)]

##----------------------- DRAFTS END --------------------------------------------------

createSuggTB <- function (lookuptb){
    aggtb1 = setNames(aggregate(lookuptb$freq1, by =list(lookuptb$sugg1),FUN=sum),c("sugg","freq"))
    aggtb2 = setNames(aggregate(lookuptb$freq2, by =list(lookuptb$sugg2),FUN=sum),c("sugg","freq"))
    aggtb3 = setNames(aggregate(lookuptb$freq3, by =list(lookuptb$sugg3),FUN=sum),c("sugg","freq"))
    aggtb4 = setNames(aggregate(lookuptb$freq4, by =list(lookuptb$sugg4),FUN=sum),c("sugg","freq"))
    aggtbFinal = rbind(aggtb1,aggtb2,aggtb3,aggtb4)
    aggtbFinal = setNames(aggregate(aggtbFinal$freq, by=list(aggtbFinal$sugg),FUN=sum),c("suggestion","frequency"))
    aggtbFinal = aggtbFinal[(order(-aggtbFinal$frequency)),]
    suggtb  = lookuptb[,c("prediction","sugg1","sugg2","sugg3","sugg4")]
    
    for( i in 1:nrow(suggtb)){
        counter = 1
        
        if(is.na(suggtb$sugg1[i])){
            suggtb$sugg1[i] = aggtbFinal$suggestion[counter]
            counter = counter +1
        }else if(suggtb$sugg1[i] == aggtbFinal$suggestion[counter]){
            counter = counter +1
        }
        
        if(is.na(suggtb$sugg2[i])){
            if(suggtb$sugg1[i] == aggtbFinal$suggestion[counter]){
                counter = counter +1
            }
            suggtb$sugg2[i] = aggtbFinal$suggestion[counter]
            counter = counter +1
        }else if(suggtb$sugg2[i] == aggtbFinal$suggestion[counter]){
            counter = counter +1
        }
        
        if(is.na(suggtb$sugg3[i])){
            if(suggtb$sugg1[i] == aggtbFinal$suggestion[counter]){
                counter = counter +1
            }
            if(suggtb$sugg2[i] == aggtbFinal$suggestion[counter]){
                counter = counter +1
            }
            suggtb$sugg3[i] = aggtbFinal$suggestion[counter]
            counter = counter +1
        }else if(suggtb$sugg3[i] == aggtbFinal$suggestion[counter]){
            counter = counter +1
        }
        
        if(is.na(suggtb$sugg4[i])){
            if(suggtb$sugg1[i] == aggtbFinal$suggestion[counter]){
                counter = counter +1
            }
            if(suggtb$sugg2[i] == aggtbFinal$suggestion[counter]){
                counter = counter +1
            }
            if(suggtb$sugg3[i] == aggtbFinal$suggestion[counter]){
                counter = counter +1
            }
            suggtb$sugg4[i] = aggtbFinal$suggestion[counter]
        }
    }
    suggtb = setNames(suggtb,c("prediction","suggestion1","suggestion2","suggestion3","suggestion4"))
    return(suggtb)
}

suggestionTable = createSuggTB(lookupTable)

## check table row for duplication value in the column
checkDuplication <- function (tb){
    for ( i in 1:nrow(tb)){
        if(length(unique(tb[i,]))!=length(tb[i,])){
            return(FALSE)
            break
        }
    }
    return(TRUE)
}

checkDuplication(cbind.data.frame(suggestionTable$suggestion1,suggestionTable$suggestion2,
                                  suggestionTable$suggestion3,suggestionTable$suggestion4,stringsAsFactors=FALSE))
## the table is correct with no column duplication

## Save Suggestion Table
saveRDS(suggestionTable,"sgTable.rds")
suggestionTable = readRDS("sgTable.rds")

