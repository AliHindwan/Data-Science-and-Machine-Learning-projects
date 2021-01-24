 #----------Structring the Code---------------
Yemen <- read.csv("Yemen.csv", header = TRUE)


#--------------Preliminaries-----------------
library("ggplot2")
library("magrittr")
library("dplyr")
library("ggthemes")
library("beepr")

#---------------Loading my COde-------------------
Yemen <- read.csv("Yemen.csv", header = TRUE)


#---------------Inspecting the data---------------
str(Yemen)
summary(Yemen)
View(Yemen)
head(Yemen)
#---------------Visualizing my Goals and Results---------

#----------- 1-Frequency of Bombing From 26-3-2015 to 25-3-2018---------
Frequency<-table(Yemen$Date)  
 View(Frequency)
 Frequen<-as.data.frame(Frequency)
 my_date<- Frequen[order(as.Date(Frequen$Var1, format="%d/%m/%Y")),]
 my_date$Var1 = as.Date(my_date$Var1, format="%d/%m/%Y")
 typeof(my_date$Var1)
 str(my_date)
 View(my_date)
 T<-ggplot(data=my_date, aes(x=Var1,y=Freq))
 T+geom_line() +ggtitle("Frequency of Bombing From 26-3-2015 to 25-3-2018") +
   xlab("Dates") +
   ylab("Frequency") +
   scale_y_continuous(breaks=seq(0,60,by=10),labels=c(0,10,20,30,40,50,60))+
   theme(plot.title=element_text(color="blue", size=12, hjust= 0.5),
        axis.title=element_text(size=12,face="bold"),
        axis.title.x=element_text(color="black"),
        axis.text=element_text(size=10),
        axis.text.y=element_text(color="black"),
        axis.text.x=element_text(color="red", size=10))

#----------2- The Frequency between Main Category & Time Interval---------------
 MCTI <- ggplot(data=Yemen, aes(x=Main.category, fill=Time.Interval))
 MCTI + geom_bar(position="dodge")+
ggtitle("The Frequency between Main Category & Time Interval")+
   xlab("Main Category") +
   ylab("Frequency") +
   scale_y_continuous(breaks=seq(0,1800,by=200),labels=c(0,200,400,600,800,
    1000,1200,1400,1600,1800))+
   theme(plot.title=element_text(color="black", size=12, hjust= 0.5),
         axis.title=element_text(size=12,face="bold"),
         axis.title.x=element_text(color="blue"),
         axis.text=element_text(size=7),
         axis.text.y=element_text(color="black"),
         axis.text.x=element_text(color="black", size=7))
 
#-------3- The Frequency of bombing in relation to Governorates-----------
Frequency2 <-table(Yemen$Governorate)
View(Frequency2)
Frequen2<-as.data.frame(Frequency2)
View(Frequen2)
G<-ggplot(data=Frequen2, aes(x=Freq,y=reorder(factor(Var1),Freq)))
G+geom_point(color="black",size=3,alpha=1/2)+theme_economist()+ 
ggtitle("The Frequency of bombing in relation to Governorates")+
  xlab("Frequency of Aerial BOmbs") +
  ylab("Governorates") +
  scale_x_continuous(breaks=seq(0,3500,by=500),labels=c(0,500,1000,1500,2000,
        2500,3000,3500))+
  theme(plot.title=element_text(color="red", size=12, hjust= 0.5),
        axis.title=element_text(size=12,face="bold"),
        axis.title.x=element_text(color="black"),
        axis.text=element_text(size=8),
        axis.text.y=element_text(color="black"),
        axis.text.x=element_text(color="black", size=10))

#--------4-The Frequency of bombing in relation to Time Interval on Governorates----------------
TIG<-ggplot(data = Yemen,aes(x=Yemen$Time.Interval,y=Yemen$Governorate))
TIG+geom_jitter()+ 
  ggtitle("The Frequency of bombing in relation to Time Interval on Governorates")+
  xlab("Time Interval") +
  ylab("Governorates") +
  theme(plot.title=element_text(color="red", size=10, hjust= 0.5),
        axis.title=element_text(size=12,face="bold"),
        axis.title.x=element_text(color="blue"),
        axis.text=element_text(size=10),
        axis.text.y=element_text(color="black"),
        axis.text.x=element_text(color="orange", size=8))

#---------- 5-The Frequency of bombing in relation to Time Interval on 3 Governorates-------
Yemen1 <- Yemen %>% filter(Governorate == c("Hadramawt","Amran","Saada"))

TI3G<-ggplot(data=Yemen1, aes(x=Yemen1$Time.Interval,y=Yemen1$Governorate))
View(TI3G)
TI3G+geom_jitter()+theme_economist()+ 
  ggtitle("The Frequency of bombing in relation to Time Interval on 3 Governorates")+
  xlab("Time Interval") +
  ylab("Governorates") +
  theme(plot.title=element_text(color="darkblue", size=8, hjust= 0.5),
        axis.title=element_text(size=12,face="bold"),
        axis.title.x=element_text(color="Red"),
        axis.text=element_text(size=12),
        axis.text.y=element_text(color="black"),
        axis.text.x=element_text(color="darkred", size=7))

#-----------6-  The Frequency of bombing in Main Categories------------
MCT<-table(Yemen$Main.category)
View(MCT)
MCTT<-as.data.frame(MCT)
View(MCTT)
MCFT<-ggplot(data=MCTT,aes(x=Freq,y=Var1))
MCFT+geom_point(color="blue")+ 
  ggtitle("The Frequency of bombing in Main Categories")+
  xlab("Frequency of Aerial Bombs") +
  ylab("Main Categories") +
  scale_x_continuous(breaks=seq(0,7500,by=1500),labels=c(0,1500,3000,
         4500,6000,7500))+
  theme(plot.title=element_text(color="black", size=12, hjust= 0.5),
        axis.title=element_text(size=12,face="bold"),
        axis.title.x=element_text(color="purple"),
        axis.text=element_text(size=7),
        axis.text.y=element_text(color="black"),
        axis.text.x=element_text(color="black", size=7))
#--------------7-The Frequency of bombing in Sub Categories---------
SCT<-table(Yemen$Sub.category)
View(SCT)
SCTT<-as.data.frame(SCT)
View(SCTT)
SCFT <- ggplot(data=SCTT,aes(x=Freq,y=Var1))
SCFT + geom_jitter()+theme_excel()+
  scale_color_excel() + 
  ggtitle("The Frequency of bombing in Sub Categories")+
  xlab("Frequency of Aerial Bombs") +
  ylab("Sub Categories") +
  scale_x_continuous(breaks=seq(0,6000,by=1000),labels=c(0,1000,2000,
        3000,4000,5000,6000))+
  theme(plot.title=element_text(color="darkred", size=12, hjust= 0.5),
        axis.title=element_text(size=12,face="bold"),
        axis.title.x=element_text(color="gold"),
        axis.text=element_text(size=7),
        axis.text.x=element_text(color="black"),
        axis.text.y=element_text(color="black", size=7))


#-----------------------8-The Frequency of bombing regard to Time Interval-------
TIT<-table(Yemen$Time.Interval)
View(TIT)
TITT<-as.data.frame(TIT)
View(TITT)
TIFT<-ggplot(data = TITT,aes(x=Freq,y=Var1))
TIFT+geom_jitter()+ 
  ggtitle("The Frequency of bombing regard to Time Interval")+
  xlab("Frequency of Aerial Bombs") +
  ylab("Time Interval") +
  scale_x_continuous(breaks=seq(0,5000,by=1000),labels=c(0,1000,2000,
                                                         3000,4000,5000))+
  theme(plot.title=element_text(color="gold", size=12, hjust= 0.5),
        axis.title=element_text(size=12,face="bold"),
        axis.title.x=element_text(color="blue"),
        axis.text=element_text(size=10),
        axis.text.x=element_text(color="black"),
        axis.text.y=element_text(color="black", size=10))

#-------------The END----------------


-----------------------------
   

