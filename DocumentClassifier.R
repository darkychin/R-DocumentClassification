## final clean script for generating the results

docClassifier = function(df) {
    ## Establish Path
    datdir = setwd("C:/Users/Junbo/Desktop/Darky_stuffs/R_Darky/TextFileClassification")
    
    ## Import library
    library(stats)
    
    ## Read Model
    model = readRDS("svmModel.rds")
    
    ## Read Suggestion Table
    suggestionTable = readRDS("sgTable.rds")
    
    
    ## Do Prediction
    ##prediction = predict(model,df)
    
    ## Prediction Data Frame
    ##predictiondf= setNames(as.data.frame(prediction),"prediction")
    predictiondf= setNames(as.data.frame(df),"prediction")
    predictiondf$id = 1:nrow(predictiondf)
    
    ## Merge with Suggestion Table
    tmp = merge(suggestionTable,predictiondf,by="prediction",all=TRUE)
    
    ## Sort Table According to Prediction
    tmp = tmp[order(tmp$id),]
    
    ## Subset Final Table
    final  = tmp[,c(1:5)]
    
    write.csv(final,"PredictionTable.csv")
}

## fetch from path 
docClassifier2 = function(path) {
    
    ## Import library
    require("readtext") #install.packages("readtext"), #library(readtext)
    library(stats)
    require("tm")
    require("e1071")
    
    ## Custom Functions
    ## -Function :: Trim unwanted "path/" in doc_id
    customTrim <- function(string,re){
        name=unlist(strsplit(string,re))
        return(name[length(name)])
    }
    ## -Function :: return category name
    category = function(string) {
        cat = unlist(strsplit(string,"_"))[2]
        return(cat)
    }
    removeURL <- function(x) gsub("http[[:alnum:][:punct:]]*", "", x) 
    removeNonAlphNum <- function(x) gsub("[^[:alnum:]]","",x)
    removeSpecialChars <- function(x) gsub(".","",x)
    
    ## Establish Path
    datdir = setwd(path)
    
    ## Read Model
    model = readRDS("svmModel.rds")
    
    ## Read Suggestion Table
    suggestionTable = readRDS("sgTable.rds")
    
    ## Read from Path
    files=readtext(paste0(datdir,"/Petronas Data/POC_Form/*/*.txt"))
    
    ## Trim Filenames
    filesMatrix= unname(sapply(files$doc_id,customTrim,"/"))
    
    ## Convert to Lowercase
    filesMatrix=unname(sapply(filesMatrix,tolower))
    filesText=unname(sapply(files$text,tolower))
    
    ## Get Category
    filesCategory= unname(sapply(filesMatrix,category))
    
    ## Combing filesCategory, filesText
    filesTmp = as.data.frame(cbind(category=filesCategory, text=filesText),stringsAsFactors = FALSE)
    
    ## Clean Empty filesTmp
    filesTmp = filesTmp[(filesTmp$category!="" & filesTmp$text!=""),]
    
    ## Build Corpus
    corpus_tm = VCorpus(VectorSource(filesTmp$text), readerControl = list(language="en")) #English
    
    ## Tranformation
    ##  1. remove all URLs 
    corpus_tm = tm_map(corpus_tm, content_transformer(removeURL))
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
    ##  9. to lower case [already done, see top]
    corpus_tm = tm_map(corpus_tm, content_transformer(tolower))
    
    ## Create Document Term Matrix
    dtm_tm = DocumentTermMatrix(corpus_tm)
    dtm_tm2 = removeSparseTerms(dtm_tm,0.99)
    
    ## Converting tmdata to dataframe
    dtm_df_tm <- as.data.frame(as.matrix(dtm_tm2))
    
    ## Cleanup Special Characters
    colnames(dtm_df_tm)=make.names(colnames(dtm_df_tm))
    
    ## Make a  Complete Processed Dataframe
    finaldf= as.data.frame(cbind(category=filesTmp$category,dtm_df_tm))
    
    ## Do Prediction
    prediction = predict(model,finaldf)
    
    ## Prediction Data Frame
    predictiondf= setNames(as.data.frame(prediction),"prediction")
    predictiondf$id = 1:nrow(predictiondf)
    
    ## Merge with Suggestion Table
    tmp = merge(suggestionTable,predictiondf,by="prediction",all=TRUE)
    
    ## Sort Table According to Prediction
    tmp = tmp[order(tmp$id),]
    
    ## Subset Final Table
    final = tmp[,c(1:5)]
    
    write.csv(final,"PredictionTable.csv")
}
