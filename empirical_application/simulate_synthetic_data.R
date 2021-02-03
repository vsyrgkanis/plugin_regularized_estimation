rm(list=ls())
library(hdm)
library(ranger)
library(glmnet)
setwd("/net/holyparkesec/data/tata/EJ/")
mydata<-read.csv("./data/mydata.csv")
## on welfare 1 year after RA
mydata$on_welfare<-(mydata$adcq4+mydata$fstq4)>0
outcome_name<-"on_welfare"
baseline_varnames<-c("white","black","hisp","marnvr","marapt", "agelt20",  "age2024",  "age2534" , "age3544",  "agege45",  "agelt24",  "agege24" ,
                     "agelt30",  "agege30",
                     "yrern","yremp","yradc",  "yrfst","yrernsq","yrvad","yrkvad","yrvfs",
                     "ernpq8","ernpq6", "ernpq5","ernpq4","ernpq3","ernpq2","ernpq1",
                     "adcpq7","adcpq6","adcpq5","adcpq4","adcpq3","adcpq2","adcpq1",
                     "fstpq7","fstpq6","fstpq5","fstpq4","fstpq3","fstpq2","fstpq1",
                     "anyernpq8","anyernpq6","anyernpq5","anyernpq4","anyernpq3","anyernpq2","anyernpq1",
                     "anyadcpq7","anyadcpq6","anyadcpq5","anyadcpq4","anyadcpq3","anyadcpq2","anyadcpq1",
                     "anyfstpq7","anyfstpq6","anyfstpq5","anyfstpq4","anyfstpq3","anyfstpq2","anyfstpq1",                      
                     c("nohsged","hsged","applcant"),c("anyernpq","anyadcpq","anyfstpq")
)
baseline_discrete_covariates<-c("white","black","hisp","marnvr","marapt", "agelt20",  "age2024",  "age2534" , "age3544",  "agege45",  "agelt24",  "agege24" ,
                                "agelt30",  "agege30", "anyernpq8","anyernpq6","anyernpq5","anyernpq4","anyernpq3","anyernpq2","anyernpq1",
                                "anyadcpq7","anyadcpq6","anyadcpq5","anyadcpq4","anyadcpq3","anyadcpq2","anyadcpq1",
                                "anyfstpq7","anyfstpq6","anyfstpq5","anyfstpq4","anyfstpq3","anyfstpq2","anyfstpq1",                      
                                c("nohsged","hsged","applcant"),c("anyernpq","anyadcpq","anyfstpq"))
baseline_continuous_covariates<-setdiff(baseline_varnames,baseline_discrete_covariates)



outcome_name<-"on_welfare"
outcome_names<-grep("fstq|earnq|adcq",names(mydata),value=TRUE)

heterogeneous_covariates<-c("white","black","hisp","marnvr","marapt",  "yrern","yrernsq","yradc",  "yrfst","yremp","yrvad","yrvfs",
                            "yrkvad",c("anyernpq","anyadcpq","anyfstpq"), c("nohsged","applcant"))


simulated_continuous_data<-mydata[,baseline_continuous_covariates]
simulated_discrete_data<-mydata[,baseline_discrete_covariates]


add_discrete_noise<-function(seed) {
  set.seed(seed)
  name=baseline_discrete_covariates[seed]
  cov<- mydata[,name]
  n<-length(cov)
  
  z<-(cov-0.5)*(rbinom(p=0.5,n=n,size=1)-0.5)*2+0.5

  return(z)
}
add_continuous_noise<-function(seed) {
  set.seed(seed)
  name=baseline_continuous_covariates[seed]
  cov<- mydata[,name]
  n<-length(cov)
  sigma<-sd(cov)
  return(cov+sigma/2*rnorm(n))
}
for (seed in 1:length(baseline_continuous_covariates)) {
  
  simulated_continuous_data[,seed]<-add_continuous_noise(seed)
}
for (seed in 1:length(baseline_discrete_covariates)) {
  simulated_discrete_data[,seed]<-add_discrete_noise(seed)
}

simulated_data<-cbind(simulated_discrete_data,simulated_continuous_data)

pscore_form<-as.formula(paste0("treatmnt~(",paste0(baseline_varnames,collapse="+"),")"))
q_form<-as.formula(paste0(outcome_name,"~."))
q_form2<-as.formula(paste0("log_odds~."))

pscore.fit<-glm(pscore_form,mydata,family=binomial)
pscore.pred<-predict(pscore.fit,mydata,type="response")

mydf=mydata[,c(baseline_varnames,"treatmnt",outcome_name)]
mydf$on_welfare<-as.numeric(mydf$on_welfare)
set.seed(1)
myForest<-ranger(q_form, data=mydf,seed=100,
                 probability=TRUE,num.trees=100,importance="permutation")
pred.D.X<-predict(myForest,mydata,type="response")
prob.D.X<-pred.D.X$predictions[,1]
prob.D.X<-sapply(sapply(prob.D.X,min,0.9999),max,0.0001)
set.seed(1)
simulated_data$treatmnt<-sapply(pscore.pred,rbinom,size=1,n=1)
simulated_data$on_welfare<-sapply(prob.D.X,rbinom,size=1,n=1)

write.csv(simulated_data,"./data/simulated_data.csv")
#simulated_data$on_welfare_prob<-plogis()