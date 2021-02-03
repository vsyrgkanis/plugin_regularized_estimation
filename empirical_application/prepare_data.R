rm(list=ls())

library(sas7bdat)
setwd("/net/holyparkesec/data/tata/EJ/")
ctadmrec<-read.sas7bdat("./data/ctadmrec.sas7bdat")
colnames(ctadmrec)<-tolower(colnames(ctadmrec))
## focus on observations whose # of children is not missing (kidcount), following Kline and Tartari (2016)
ctadmrec<-ctadmrec[!is.na(ctadmrec$kidcount),]
## welfare measures after random assignment
outcome_names<-grep("fstq|ernq|adcq",names(ctadmrec),value=TRUE)



baseline_varnames<-c("white","black","hisp","marnvr","marapt", "agelt20",  "age2024",  "age2534" , "age3544",  "agege45",  "agelt24",  "agege24" ,
                     "agelt30",  "agege30",
                     "yrern","yremp","yradc",  "yrfst","yrernsq","yrvad","yrkvad","yrvfs",
                     "yremp", 
                     "ernpq8","ernpq6", "ernpq5","ernpq4","ernpq3","ernpq2","ernpq1",
                     "adcpq7","adcpq6","adcpq5","adcpq4","adcpq3","adcpq2","adcpq1",
                     "fstpq7","fstpq6","fstpq5","fstpq4","fstpq3","fstpq2","fstpq1",
                     "anyernpq8","anyernpq6","anyernpq5","anyernpq4","anyernpq3","anyernpq2","anyernpq1",
                     "anyadcpq7","anyadcpq6","anyadcpq5","anyadcpq4","anyadcpq3","anyadcpq2","anyadcpq1",
                     "anyfstpq7","anyfstpq6","anyfstpq5","anyfstpq4","anyfstpq3","anyfstpq2","anyfstpq1",                      
                     c("nohsged","hsged","applcant"),c("anyernpq","anyadcpq","anyfstpq")
)
### create discrete representation of continuous variable 
## anyx = (x>0) for a numeric x
for (name in c("ernpq8","ernpq6", "ernpq5","ernpq4","ernpq3","ernpq2","ernpq1",
               "adcpq7","adcpq6","adcpq5","adcpq4","adcpq3","adcpq2","adcpq1",
               "fstpq7","fstpq6","fstpq5","fstpq4","fstpq3","fstpq2","fstpq1")) {
  x<-(ctadmrec[,name]>0)
  anyname<-paste0("any",name)
  ctadmrec$x<-x
  colnames(ctadmrec)[colnames(ctadmrec)=="x"]<-anyname
}
ctadmrec$anyernpq<-apply(ctadmrec[,grep("anyernpq",colnames(ctadmrec),TRUE)],1,sum)>0
ctadmrec$anyadcpq<-apply(ctadmrec[,grep("anyadcpq",colnames(ctadmrec),TRUE)],1,sum)>0
ctadmrec$anyfstpq<-apply(ctadmrec[,grep("anyfstpq",colnames(ctadmrec),TRUE)],1,sum)>0
mydata<-sapply(ctadmrec[,c("treatmnt",baseline_varnames,outcome_names)], as.numeric)
mydata[is.na(mydata)]<-0
write.csv(mydata,"./data/mydata.csv")
