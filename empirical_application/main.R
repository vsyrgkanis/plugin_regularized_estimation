### direct approach inspired by Belloni, Chernozhukov, Wei (2016)
rm(list=ls())
#install.packages(c("hdm","ranger","glmnet","xtable"))
library(hdm)
### requires most recent version of ranger ranger_0.12.1 
library(ranger)
library(glmnet)
library(xtable)

#setwd("/net/holyparkesec/data/tata/EJ/replication/")
### load synthetic data
synthetic<-FALSE
if (synthetic) {
  mydata<-read.csv("./data/simulated_data.csv")
  data_option<-"synthetic"
  
} else {
  ### this option is not available in public use files
  ### please check Readme on how to obtain MDRC data and preprocess them
  ## change filepath to the location of your file
  mydata<-read.csv("./data/mydata.csv")
  ## on welfare 1 year after RA
  mydata$on_welfare<-(mydata$adcq4+mydata$fstq4)>0
  data_option<-""
}

outcome_name<-"on_welfare"
outcome_names<-grep("fstq|earnq|adcq",names(mydata),value=TRUE)
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

heterogeneous_covariates<-c("white","black","hisp","marnvr","marapt",  "yrern","yrernsq","yradc",  "yrfst","yremp","yrvad","yrvfs",
                            "yrkvad",c("anyernpq","anyadcpq","anyfstpq"), c("nohsged","applcant"))
final_table<-matrix(0,length(heterogeneous_covariates)+1,5)
mydata$yrkvad<-mydata$yrkvad/100



form_matrix<-paste0("~treatmnt+treatmnt*(",paste0(heterogeneous_covariates,collapse="+"),")+(", paste0(baseline_varnames,collapse="+"),")*(",paste0(baseline_continuous_covariates,collapse="+"),")")
#### data 
mydata1<-mydata
mydata1$treatmnt<-1
mydata0<-mydata
mydata0$treatmnt<-0


covariates<-model.matrix(as.formula(form_matrix),mydata)
covariates1<-model.matrix(as.formula(form_matrix),mydata1)
covariates0<-model.matrix(as.formula(form_matrix),mydata0)
form_outcome<-paste0(outcome_name,form_matrix)

################## LINEAR LOGISTIC APPROACH ##########################
# DIRECT APPROACH (COLUMN 1) #
n<-dim(mydata)[1]
lambda1<-(1.1/2/sqrt(n))*qnorm(1-0.05/(dim(covariates)[2]*log(n)))
glm.naive.fit<-glmnet(x=covariates,
                      y=mydata[,outcome_name],family="binomial",standardize=TRUE, intercept=TRUE,
                      lambda=lambda1,penalty.factor=c(0,0,rep(1,dim(covariates)[2]-1 )))
theta.tilde<-as.numeric(coef(glm.naive.fit))
final_table[,1]<-c(theta.tilde[3],rep(0,length(heterogeneous_covariates)))

################## PARTIALLY LINEAR LOGISTIC APPROACH based on ORTHOGONAL MOMENT ##########################
################# LASSO (COLUMN 2)  #############
## propensity score 
pscore_form<-as.formula(paste0("treatmnt~(",paste0(baseline_varnames,collapse="+"),")"))
q_form<-as.formula(paste0(outcome_name,"~."))
q_form2<-as.formula(paste0("log_odds~."))

pscore.fit<-glm(pscore_form,mydata,family=binomial)
pscore.pred<-predict(pscore.fit,mydata,type="response")
pscore_resid<-mydata$treatmnt-pscore.pred


mydf=mydata[,c(baseline_varnames,"treatmnt",outcome_name)]
mydf$on_welfare<-as.numeric(mydf$on_welfare)
set.seed(1)
myForest<-ranger(q_form, data=mydf,seed=100,
                 probability=TRUE,num.trees=100,importance="permutation")
pred1<-predict(myForest,mydata1,type="response")
pred0<-predict(myForest,mydata0,type="response")

pred.D.X<-predict(myForest,mydata,type="response")
prob.D.X<-pred.D.X$predictions[,1]
prob.D.X<-sapply(sapply(prob.D.X,min,0.9999),max,0.0001)
mydata$log_odds<-log(prob.D.X/(1-prob.D.X))


myForest2<-ranger(q_form2,data=mydata[,c(baseline_varnames,"log_odds")],seed=1000,num.trees=100,importance="permutation")
pred<-predict(myForest2,mydata)
q0x<-pred$predictions


#### V_0(d,x) = G(d theta_0 + f_0(x))*(1-G(d theta_0 + f_0(x)))
min_weight<-0.01
rho<-0.2
g.hat.fit<-predict(myForest,mydata,type="response")
g.hat<-g.hat.fit$predictions[,1]
weights<-as.numeric(g.hat*(1-g.hat))
weights<-sapply(weights,max,min_weight)

# treatment interactions #
treat_interactions<-model.matrix(as.formula(paste0("~treatmnt+treatmnt:(",paste0(heterogeneous_covariates,collapse="+"),")-1")),
                                 mydata)

treat_interactions1<-model.matrix(as.formula(paste0("~treatmnt+treatmnt:(",paste0(heterogeneous_covariates,collapse="+"),")-1")),
                                  mydata1)
treat_interactions0<-model.matrix(as.formula(paste0("~treatmnt+treatmnt:(",paste0(heterogeneous_covariates,collapse="+"),")-1")),
                                  mydata0)

treat_interactions_matrix<-matrix(rep(pscore_resid,dim(treat_interactions)[2]),ncol = dim(treat_interactions)[2])*treat_interactions
treat_interactions_matrix1<-matrix(rep(pscore_resid,dim(treat_interactions1)[2]),ncol = dim(treat_interactions)[2])*treat_interactions1
treat_interactions_matrix0<-matrix(rep(pscore_resid,dim(treat_interactions0)[2]),ncol = dim(treat_interactions)[2])*treat_interactions0

colnames(treat_interactions_matrix)<-gsub(":",".",colnames(treat_interactions_matrix))
colnames(treat_interactions_matrix1)<-gsub(":",".",colnames(treat_interactions_matrix1))
colnames(treat_interactions_matrix0)<-gsub(":",".",colnames(treat_interactions_matrix0))


p<-dim(treat_interactions_matrix)[2]

glm.ortho.fit<-glmnet(x=treat_interactions_matrix,y=as.factor(mydata[,outcome_name]),
                      standardize=TRUE,
                      family="binomial",weights=weights,
                      lambda=(1.01/2/sqrt(n))*qnorm(1-0.05/n), offset=q0x,
                      penalty.factor=c(0,rep(1,dim(treat_interactions_matrix)[2]-1)))
final_table[,2]<-as.numeric(coef(glm.ortho.fit))[-1]

################# POST-LASSO (COLUMN 3)  #############
glm.post.ortho.fit<-glm(as.formula(paste0(outcome_name,"~.")),data=data.frame(cbind(treat_interactions_matrix[,c("treatmnt",
                                                                                                                 "treatmnt.yrkvad"  )],
                                                                                    on_welfare=as.numeric(mydata$on_welfare))), 
                        family="binomial",weights=weights,offset=q0x)

final_table[,3]<-c(coef(glm.post.ortho.fit)[2],rep(0,12), coef(glm.post.ortho.fit)[3],  rep(0,5))


################# UNPENALIZED LASSO (COLUMN 4)  #############
glm.fit.all<-glm(as.formula(paste0(outcome_name,"~.")),
                 data=data.frame(cbind(treat_interactions_matrix,on_welfare=as.numeric(mydata$on_welfare))),
                 family="binomial",weights=weights,offset=q0x)
colnames(treat_interactions_matrix1)<-names(coef(glm.fit.all))[-1]
colnames(treat_interactions_matrix0)<-names(coef(glm.fit.all))[-1]
final_table[,4]<-coef(glm.fit.all)[-1]
final_table[,5]<-summary(glm.fit.all)$coefficients [-1,2]


##### PRINT the RESULTS ####
final_table<-apply(final_table,2,round,4)
rownames(final_table)<-c("intercept",heterogeneous_covariates)

write.table(xtable(final_table,digits = 3),paste0("./results/finaltable",data_option,".txt"))
