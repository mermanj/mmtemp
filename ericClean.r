#Behavioral plots
rm(list=ls())

library(plyr)
library(stringr)


setwd("/home/hanshalbe/Desktop/mastermind-master")

d<-read.csv("cleanedDataMmind18_emap.csv")



d$idn<-as.numeric(substr(d$id, 6,7))
d<-subset(d, !is.na(d$idn))

for (i in 1:nrow(d)){
  s1<-gsub("\\[|\\]", "", paste(d$guesses[i]))
  m1<-matrix(as.numeric(strsplit(s1, ";")[[1]]), ncol=3, byrow = TRUE)
  s2<-gsub("\\[|\\]", "", paste(d$truecode[i]))
  m2<-as.numeric(strsplit(s2, ";")[[1]])
  m2<-matrix(rep(m2, nrow(m1)), ncol=3, byrow=TRUE)
  s3<-gsub("\\[|\\]", "", paste(d$codejar[i]))
  m3<-as.numeric(strsplit(s3, ";")[[1]])
  m3<-matrix(rep(m3, each=nrow(m2)), ncol=6)
  dummy<-as.data.frame(cbind(m1, m2, m3))      
  names(dummy)<-c(paste0("guess", 1:3), paste0("truth", 1:3), paste0("code", 1:6))
  dummy$game<-i
  dummy$id<-d$idn[i]
  if (i ==1){dat<-dummy}
  if (i>1){dat<-rbind(dat, dummy)}
}
dat<-dat[order(dat$id),]

k<-1
dat$n<-k
for (i in 2:nrow(dat)){
  if (dat$id[i]==dat$id[i-1]){
    dat$n[i]<-k
  }
  if (dat$id[i]!=dat$id[i-1]){
    k<-k+1
    dat$n[i]<-k
  }
}
head(dat, 200)
dat$id<-dat$n
dat$n<-NULL
head(dat)
dat$id
dat$game<-dat$game-ave(dat$game, dat$id, FUN=min)+1
head(dat, 200)

write.csv(dat, "mastermindclean.csv")