# Regression Analysis on The Median Value of Owner-Occupied Homes
# YICHAO, IVAN, DAI
# ID 1098325, Wenzhou-Kean University 
# Major in FIN

# ----------------------------------------

# Package needed:
library(readxl)
library(tidyr)
library(dplyr)
library(ggplot2)
library(caret)
library(corrplot)
library(gridExtra)
library(statsr)
library(ggExtra)
library(rpart)
library(rpart.plot)
library(glmnet)
library(car)
library(DescTools)

# ----------------------------------------
# Data input and Description
# read the xlsx file:
data = as.data.frame(read_xlsx('Suburbs data.xlsx'))
head(data)
## Data sample: 
##      crim zn indus chas   nox    rm  age    dis rad tax ptratio  black lstat medv
## 1 0.00632 18  2.31    0 0.538 6.575 65.2 4.0900   1 296    15.3 396.90  4.98 24.0
## 2 0.02731  0  7.07    0 0.469 6.421 78.9 4.9671   2 242    17.8 396.90  9.14 21.6
## 3 0.02729  0  7.07    0 0.469 7.185 61.1 4.9671   2 242    17.8 392.83  4.03 34.7
## 4 0.03237  0  2.18    0 0.458 6.998 45.8 6.0622   3 222    18.7 394.63  2.94 33.4
## 5 0.06905  0  2.18    0 0.458 7.147 54.2 6.0622   3 222    18.7 396.90  5.33 36.2
## 6 0.02985  0  2.18    0 0.458 6.430 58.7 6.0622   3 222    18.7 394.12  5.21 28.7
str(data)
## Data Type:
## 'data.frame':	506 obs. of  14 variables:
## $ crim   : num  0.00632 0.02731 0.02729 0.03237 0.06905 ...
## $ zn     : num  18 0 0 0 0 0 12.5 12.5 12.5 12.5 ...
## $ indus  : num  2.31 7.07 7.07 2.18 2.18 2.18 7.87 7.87 7.87 7.87 ...
## $ chas   : num  0 0 0 0 0 0 0 0 0 0 ...
## $ nox    : num  0.538 0.469 0.469 0.458 0.458 0.458 0.524 0.524 0.524 0.524 ...
## $ rm     : num  6.58 6.42 7.18 7 7.15 ...
## $ age    : num  65.2 78.9 61.1 45.8 54.2 58.7 66.6 96.1 100 85.9 ...
## $ dis    : num  4.09 4.97 4.97 6.06 6.06 ...
## $ rad    : num  1 2 2 3 3 3 5 5 5 5 ...
## $ tax    : num  296 242 242 222 222 222 311 311 311 311 ...
## $ ptratio: num  15.3 17.8 17.8 18.7 18.7 18.7 15.2 15.2 15.2 15.2 ...
## $ black  : num  397 397 393 395 397 ...
## $ lstat  : num  4.98 9.14 4.03 2.94 5.33 ...
## $ medv   : num  24 21.6 34.7 33.4 36.2 28.7 22.9 27.1 16.5 18.9 ...
dim(data)
## The data have 506 observation and 14 variables
## [1] 506  14
summary(data) ## Show the descriptive statistics of each variables 
colSums(is.na(data))
##     crim       zn    indus     chas      nox       rm      age      dis 
##        0        0        0        0        0        0        0        0 
##      rad      tax  ptratio    black    lstat     medv
##        0        0        0        0        0        0 
## There is no missing value. 


# ---------------------------------------------

# Split the data set:
set.seed(123) # to ensure repeated ressult 
index = createDataPartition(y = data$medv ,p = 0.8, list = FALSE) 
# Partition Indedx, 80% goes to trianing set, 
# another 20% will goes to test for model prediction
train = data[index, ] # Training set
test = data[-index, ] # Test set
dim(train)
## [1] 407  14
dim(test)
## [1] 99 14

# ---------------------------------------------

# Explanatory Data Analysis
## 1. histogram of the medv:
bl = qplot(train$medv,geom = 'blank', main = 'Histogram of Medv',
      xlab = 'Medv', ylab = 'Frequence') +
        geom_histogram(color = 'black', fill = 'white', 
                       closed = 'right', bins = 25) +
        geom_vline(mapping=aes(xintercept=c(mean(train$medv), ## add three important line
                                            median(train$medv), 
                                            quantile(train$medv)[4]+1.5* IQR(train$medv)),
                               linetype=factor(c("mean","median",'threshold')),
                               color = factor(c("mean","median",'threshold'))), show.legend = TRUE)+
        scale_linetype_manual(values=c(4,5,6)) + 
        scale_color_manual(values=c(4,3,8)) +
        labs(linetype="Legend", color = "Legend")+
        geom_segment(aes(x = 45, y = 30, xend = 38, yend = 20),
                     arrow = arrow(length = unit(0.3, "cm")))+
        annotate("text", x=45, y=35, label= "Upper bound to\n detect the outlier") + 
        theme_bw()
bl
knitr::kable(train %>% ## show the skewness of the distribution 
                     summarise(skewness = Skew(medv), logSkew = log(Skew(medv))))

## | skewness|   logSkew|
## |--------:|---------:|
## | 1.106148| 0.1008842|
## original result shows high skewness, it should be transformed to be more normal.

al = qplot(log(train$medv),geom = 'blank', main = 'Histogram of Medv.log',
           xlab = 'medv.log', ylab = 'Frequence')+
       geom_histogram(color = 'black', fill = 'white', 
                      closed = 'right', bins = 25) +
       scale_linetype_manual(values=c(4,5,6)) + 
       scale_color_manual(values=c(4,3,8)) +
       labs(linetype="Legend", color = "Legend")+
       geom_vline(mapping=aes(xintercept=c(mean(log(train$medv)),
                                           median(log(train$medv)), 
                                           quantile(log(train$medv))[4]+1.5* IQR(log(train$medv))),
                              linetype=factor(c("mean","median",'threshold')),
                              color = factor(c("mean","median",'threshold'))), show.legend = TRUE)+
       theme_bw()
al
ab = grid.arrange(bl, al)  ## Put two plot in one plot
ggsave(plot = ab, '1.png') ## save two plot in one plot


## 2. Scatter plot between medv and lstat, scaled by crime
cor.test(train$medv, train$lstat) ## show the correlation between the medv and lstat
ggplot(data = train, aes(x = lstat, y = medv, size = crim, color = crim))+
        geom_point(alpha = 0.9)+
        scale_color_continuous(low = '#FDD819', high = '#E04C4C', breaks = seq(0,80,20))+
        theme_bw()+
        labs(color = 'Crim', size = 'Crim', title = 'Scatter Plot Between medv and lstat')+
        geom_smooth(method = 'loess', se = FALSE, color = 'black', lty = 2, lwd =0.5)
ggsave('2.png') # save the plot
## to test whether there is significant difference
t.test(train$medv[train$crim < 50], train$medv[train$crim > 50]) 
## simple regression between medv and lstat
s1 = lm(medv~lstat, data = train)
summary(s1)

## 3. Scatter Plot Between medv and dis, scaled by rad
cor.test(train$medv, train$dis)  ## correlation between the medv and dis
## 0.2595908 
cor.test(train$rad, train$dis)   ## correlation between the rad and dis
## -0.5133001 
ggplot(data = train, aes(x = dis, y = medv, size = factor(rad), color = rad))+
        geom_point(alpha = 0.7)+
        scale_color_continuous(low = 'lightblue', high = 'darkblue', breaks = seq(0,20,5))+ ## manual color
        theme_bw()+
        labs(color = 'rad', size = 'rad', 
             title = 'Scatter Plot Between medv and dis')
ggsave('3.png') ## save the plot

e = ggplot(data = train, aes(x = factor(rad), y = dis, fill = factor(rad)))+
  geom_boxplot(alpha = 0.6, outlier.shape = 4, outlier.size = 3, show.legend = FALSE)+
  theme_bw()+
  labs(x ='rad', y = 'dis', fill = 'rad')  ## Boxplot of dis against rad
ggsave(plot = e, '3.1.png')

## simple regression between medv and lstat
s2 = lm(medv~dis, data = train)
summary(s2)

## 4. Boxplot rad and medv
c = ggplot(data = train, aes(x = factor(rad), y = medv, fill = factor(rad)))+
  geom_boxplot(alpha = 0.6, outlier.shape = 4, outlier.size = 3)+
  theme_bw()+
  labs(x ='rad', y = 'medv', fill = 'rad')+
  coord_flip() ## Boxplot of medv against rad

d = ggplot(data = train, aes(x = medv, fill = factor(rad)))+
  geom_density(alpha = 0.6)+
  theme_bw()+
  labs(fill =  'rad') ## density distribution of medv against rad

cd = grid.arrange(c, d, top = 'Boxplot of medv against the rad') ## Put two different plots in one plot
ggsave(plot = cd, '4.png') ## save the plot

## ANOVA test to check whether there is a significant different between at least one pair. 
anova = aov(data = train, medv~factor(rad))
summary(anova)
##              Df Sum Sq Mean Sq F value Pr(>F)    
## factor(rad)   8   8031  1003.9   15.31 <2e-16 ***
##   Residuals   398  26094    65.6                   
## ---
##   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

## 5. Boxplot of medv against chas
a = ggplot(data = train, aes(x = factor(chas), y = medv, fill = factor(chas)))+
        geom_boxplot(alpha = 0.6, outlier.shape = 4, outlier.size = 3)+
        theme_bw()+
        geom_jitter(aes(x = factor(chas), y =medv, col =factor(chas)), ## add the jitter point
                    width = 0.2, ## width variance equal to 20%
                    alpha = 0.5)+
        scale_color_manual(values = c(2,6))+
        labs(x ='', y = '', fill = 'chas', color = 'chas') ## Boxplot of medv against chas

b = ggplot(data = train, aes(x = factor(chas), y = medv, fill = factor(chas)))+
        geom_violin(alpha = 0.6)+
        theme_bw()+
        labs(x ='', y = '', fill = 'chas', color = 'chas') ## violin ploy of medv against chas
ab = grid.arrange(a, b, 
                  top = 'Boxplot of medv against the chas',
                  bottom = 'chas') ## put two plot in one plot 
ggsave(plot = ab, '5.png') ## save the plot

## t-test to check whehter there is a significant different if there is a river limitation 
inference(data = train, y = medv, x = factor(chas), 
          type = 'ht', statistic = 'mean',null = 0, 
          alternative = 'twosided', method = 'theoretical')
## Response variable: numerical
## Explanatory variable: categorical (2 levels) 
## n_0 = 377, y_bar_0 = 22.0899, s_0 = 8.8819
## n_1 = 30, y_bar_1 = 27.7967, s_1 = 11.077
## H0: mu_0 =  mu_1
## HA: mu_0 != mu_1
## t = -2.7523, df = 29
## p_value = 0.0101


## 6. Scatter plot between rm ad medv
p <- ggplot(data = train, aes(x=rm, y=medv, size = age,color = age)) +
        geom_point(alpha = 0.7) +
        theme_bw()+
        theme(legend.position="none")
pp = ggMarginal(p, type="histogram", fill = "red", alpha = 0.6) ## add the marginal histogram plot
pp
ggsave(plot = pp, '6.png')
## Check the correlation by setting different rm group 
cor(train$medv[train$rm<=4.5], train$rm[train$rm<=4.5])
## [1] -0.978819
cor(train$medv[train$rm>4.5], train$rm[train$rm>4.5])
## [1] 0.7154657

## Plot the scatter plot and regression line according different age group (<50 and >50)
tt = train %>% 
  mutate(age_f = ifelse(age < 50, 'Low age (< 50)', 'High age (>50)')) %>% ## add one new factor varaible
  ggplot(aes(y = medv, x = rm,size = age,color = age)) + 
  geom_point(alpha = 0.6)+
  facet_grid(.~age_f)+ ## make the plot separately
  theme_bw() +
  geom_smooth(method = 'lm', ## add the regression line 
              formula = y ~ x, se = FALSE, lty = 2, 
              col = 'orange', lwd = 1)
hh = grid.arrange(pp, tt) ## put the ggMarginal plot and facet plot together
ggsave(plot = hh, '6.1.png') ## save the plot 

## Check the correlation between medv and rm according to different age group 
cor(train$medv[train$age < 50], train$rm[train$age < 50])
## [1] 0.9233481
cor(train$medv[train$age > 50], train$rm[train$age > 50])
##[1] 0.6185904

## Simple Regression Beween medv and rm
s3 = lm(medv~rm, data = train)
summary(s3)

## 7. Correlation Matrix
corrplot(cor(train[,-14]),method = 'square',type = 'lower')
mod = lm(medv~., data = train) ## first build the full model 
vif(mod) ## Check the variance inflation factor 
# ---------------------------------------------

# Data transformation:
# We have seen that the response variables is highly right skewed. In the model,
# we would like to log the response variable to get more normal distribution.
train$medv = log(train$medv)
colnames(train)[14] = 'medv.log'

# ---------------------------------------------

# Modwl Building / Analysis
## Performane function
Performane <- function(true, predicted, df, k) {  ## k is number of the regressors
        SSE <- sum((exp(predicted) - exp(true))^2)  ## because the predited result is log()
        SST <- sum((exp(true) - mean(exp(true)))^2)
        R_square <- 1 - SSE / SST   
        adj_R_square <- 1 - ((1 - R_square) * (nrow(df) - 1) / (nrow(df) - k - 1)) 
        RMSE = sqrt(SSE/nrow(df))
        # print the result 
        data.frame(
                RMSE = RMSE,
                Rsquare = R_square,
                AdjRsquare = adj_R_square
        )
        
}

# -----------

## OLS FULL MODEL
mod = lm(medv.log~., data = train) ## OLS full model 
summary(mod)
## Model Performance:
## Call:
## lm(formula = medv.log ~ ., data = train)
## 
## Residuals:
##   Min       1Q   Median       3Q      Max 
## -0.75651 -0.09795 -0.01562  0.09365  0.87472 
## 
## Coefficients:
##                  Estimate Std. Error t value Pr(>|t|)    
##   (Intercept)  4.0742251  0.2194337  18.567  < 2e-16 ***
##   crim        -0.0086881  0.0015289  -5.683 2.59e-08 ***
##   zn           0.0010046  0.0006242   1.609 0.108363    
##   indus        0.0016972  0.0027174   0.625 0.532606    
##   chas         0.0922525  0.0367271   2.512 0.012410 *  
##   nox         -0.7689606  0.1695466  -4.535 7.65e-06 ***
##   rm           0.0822323  0.0176269   4.665 4.23e-06 ***
##   age          0.0007802  0.0006056   1.288 0.198424    
##   dis         -0.0443855  0.0090044  -4.929 1.22e-06 ***
##   rad          0.0141637  0.0030118   4.703 3.56e-06 ***
##   tax         -0.0005809  0.0001695  -3.427 0.000676 ***
##   ptratio     -0.0381756  0.0058688  -6.505 2.37e-10 ***
##   black        0.0004942  0.0001146   4.313 2.04e-05 ***
##   lstat       -0.0307852  0.0023107 -13.323  < 2e-16 ***
##   ---
##   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
## 
## Residual standard error: 0.1874 on 393 degrees of freedom
## Multiple R-squared:  0.7918,	Adjusted R-squared:  0.7849 
## F-statistic:   115 on 13 and 393 DF,  p-value: < 2.2e-16

png('ols.1.png') ## save the plot
plot(mod, which = 1) ## to check the qqplot whether the residual are around 0
dev.off()
png('ols.2.png', width = 960 ) ## save the plot
par(mfrow = c(1,2)) ## to separate the plot surface into two parts 
plot(mod, which = c(5,4)) ## to check whether there is influential point (extreme outlier)
dev.off()
par(mfrow = c(1,1)) ## default the plot surface 

## FULL model performance on the train set:
x = train[,-14]
y = train$medv.log
predictions_train = predict(mod, newdata = x)
Performane(y, predictions_train, train, 13)
##      RMSE   Rsquare AdjRsquare
## 1 4.331008 0.7762864  0.7688862

# -----------

## Backward AIC Selection Method
start = lm(medv.log ~ .,data=train) # full model / start model, becauseit is backward 
empty = lm(medv.log ~ 1,data=train) # empty model 
backwardStepwise = step(start,
                        scope=list(upper=start,lower=empty),
                        direction='forward')  ## backward process to get the final model 
summary(backwardStepwise)
## AIC model and model performance 
## Call:
##   lm(formula = medv.log ~ crim + zn + chas + nox + rm + dis + rad + 
##        tax + ptratio + black + lstat, data = train)
## 
## Residuals:
##   Min       1Q   Median       3Q      Max 
## -0.75622 -0.09902 -0.01684  0.09560  0.87624 
## 
## Coefficients:
##                 Estimate Std. Error t value Pr(>|t|)    
##   (Intercept)  4.0417400  0.2181850  18.524  < 2e-16 ***
##   crim        -0.0087764  0.0015275  -5.745 1.83e-08 ***
##   zn           0.0008860  0.0006186   1.432  0.15287    
##   chas         0.0963215  0.0364773   2.641  0.00860 ** 
##   nox         -0.6834618  0.1568958  -4.356 1.69e-05 ***
##   rm           0.0859062  0.0171156   5.019 7.86e-07 ***
##   dis         -0.0491355  0.0083509  -5.884 8.56e-09 ***
##   rad          0.0134374  0.0029194   4.603 5.63e-06 ***
##   tax         -0.0005342  0.0001569  -3.405  0.00073 ***
##   ptratio     -0.0369238  0.0057589  -6.412 4.11e-10 ***
##   black        0.0005033  0.0001142   4.408 1.35e-05 ***
##   lstat       -0.0295791  0.0021369 -13.842  < 2e-16 ***
##   ---
##   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
## 
## Residual standard error: 0.1875 on 395 degrees of freedom
## Multiple R-squared:  0.7907,	Adjusted R-squared:  0.7849 
## F-statistic: 135.7 on 11 and 395 DF,  p-value: < 2.2e-16


x = train[,-14]
y = train$medv.log
predictions_train = predict(jj, newdata = x)
Performane(y, predictions_train, train, 11) ## AIC Model performance 
##      RMSE  Rsquare AdjRsquare
## 1 4.3393 0.775429  0.7691752

# -----------

## LASSO Method
x = model.matrix(medv.log~.-1,data=train)
y = train$medv.log
lassoModel = glmnet(x,y, alpha=1) # alpha = 1 means lasso model / 0 means Ridge model 

par(mfrow = c(1,2))
plot(lassoModel,xvar='lambda',label=T) ## relation between lambda and coeficients of each variables 
plot(lassoModel,xvar='dev',label=T)  ## relation between R square and coeficients of each variables 
par(mfrow = c(1,1))
set.seed(111) ## make sure the repeated result 
cv.lasso = cv.glmnet(x,y,alpha=1) # Default as 10-fold cross-validation
plot(cv.lasso) ## LASSO model: lambda selection 

lambda_best <-cv.lasso$lambda.min  # Best lambda corresponding with best model 
lambda_best
## [1] 0.000336631
lasso_model <- glmnet(x, y, alpha = 1, lambda = lambda_best) ## use the best lambda from previous 
lasso_model$beta
## Lasso Model Coeficient:
## 13 x 1 sparse Matrix of class "dgCMatrix"
## s0
## crim    -0.0085736823
## zn       0.0009285056
## indus    0.0012299462
## chas     0.0929906556
## nox     -0.7470500709
## rm       0.0831255962
## age      0.0007155919
## dis     -0.0438638987
## rad      0.0133598297
## tax     -0.0005398714
## ptratio -0.0378202880
## black    0.0004911908
## lstat   -0.0306514035

## LASSO model performance based on training set 
predictions_train <- predict(lasso_model, newx = x)
Performane(y, predictions_train, train, 13)
##       RMSE   Rsquare AdjRsquare
## 1 4.335383 0.7758343  0.7684191

# -----------

## Decision Tree Method
## cp stands for complexity parameter, when cp = 0, the tree will be the most complex one 
## however, cp = 0 may not be the best one 
regTree = rpart(medv.log ~ ., train, cp = 0) ## Maximum tree
rpart.plot(regTree) ## visualize the maximum tree 

## Tuning tree to find the cp corresponding to best-performance tree 
trControl = trainControl(method='cv',number = 5) ## cross validation equal to 5 
tuneGrid = expand.grid(.cp = seq(from = 0,to = 0.01,by = 0.0001)) ## set the gap for each test
set.seed(666)
cvModel = train(medv.log~.,
                data=train,
                method="rpart",
                trControl = trControl,
                tuneGrid = tuneGrid)
## First 10 rows with differe
cvModel$results[1:10,]
##       cp      RMSE  Rsquared       MAE     RMSESD RsquaredSD       MAESD
## 1  0e+00 0.2024554 0.7536719 0.1452233 0.01657282 0.08082679 0.01202937
## 2  1e-04 0.2024554 0.7536719 0.1452233 0.01657282 0.08082679 0.01202937
## 3  2e-04 0.2024554 0.7536719 0.1452233 0.01657282 0.08082679 0.01202937
## 4  3e-04 0.2024554 0.7536719 0.1452233 0.01657282 0.08082679 0.01202937
## 5  4e-04 0.2024554 0.7536719 0.1452233 0.01657282 0.08082679 0.01202937
## 6  5e-04 0.2022443 0.7542351 0.1448821 0.01611488 0.07960634 0.01133690
## 7  6e-04 0.2025216 0.7531873 0.1447722 0.01586935 0.07926145 0.01157248
## 8  7e-04 0.2022043 0.7542438 0.1444160 0.01517880 0.07698290 0.01087097
## 9  8e-04 0.2020450 0.7548817 0.1444828 0.01531609 0.07709387 0.01127228
## 10 9e-04 0.2022152 0.7545465 0.1449131 0.01543135 0.07705662 0.01144616

## visualize the selection process  
ggplot(data=cvModel$results, aes(x=cp, y=RMSE))+
        geom_line(size=0.6,alpha=0.3)+
        geom_point(color='orange')+ 
        theme_bw()+
        labs(title = 'Model Parameter Selection Process') + 
        geom_vline(xintercept = cvModel$bestTune$cp, lwd = 0.5, lty = 2, color = 'grey')

## The best cp:
cvModel$bestTune$cp
## 8e-04
## Build the model and plot the tree
regTree_best = rpart(medv.log ~ ., train, cp = cvModel$bestTune$cp)
rpart.plot(regTree_best)

## Tree performance based on the train data set 
x = train[,-14]
y = train$medv.log
predictions_train = predict(regTree_best, newdata = x)
Performane(y, predictions_train, train, 13)
##       RMSE   Rsquare AdjRsquare
## 1 3.119253 0.8839579  0.8801194

# ---------------------------------------------

# Test the test set by using the best model -- Tree Model 
par(mfrow = c(1,3))
x = test[,-14]
y = log(test$medv)
predictions_test = predict(regTree_best, newdata = x)
Performane(y, predictions_test, test, 13)
##       RMSE   Rsquare AdjRsquare
## 1 4.164666 0.8000974  0.7695241
plot(exp(y), exp(predictions_test), col = rgb(0.4, 0.5,0.5), pch = 19,
     main  = 'Tree Models', xlab = 'Observed', ylab = 'Predicted')
abline(c(0,1))


# Test the test set by using the LASSO model 
x = model.matrix(log(medv)~.-1,data=test)
y = log(test$medv)
predictions_test <- predict(lasso_model, newx = x)
Performane(y, predictions_test, test, 13)
##       RMSE   Rsquare AdjRsquare
## 1 4.221027 0.7946502  0.7632437
plot(exp(y), exp(predictions_test), pch = 19, col = rgb(0.4, 0.5,0.5),
     main  = 'LASSO Models', xlab = 'Observed', ylab = 'Predicted')
abline(c(0,1))


# Test the test set by using AIC model 
x = test[,-14]
y = log(test$medv)
predictions_test = predict(backwardStepwise, newdata = x)
Performane(y, predictions_test, test, 13)
##       RMSE   Rsquare AdjRsquare
## 1 4.208077 0.7959083  0.7646942
plot(exp(y), exp(predictions_test), col = rgb(0.4, 0.5,0.5),pch = 19, 
     main  = 'AIC Models', xlab = 'Observed', ylab = 'Predicted')
abline(c(0,1))

# ----------------------------------------

# Conclusion:
## Through the previous model building and model analysis, tree model perform the 
## best compared to other two models. We do not use the full model because it is 
## overfitting and have multicollinearity problems. 
## Tree model have a R square equal to 80% and RMSE equal to 4.164666, which has the
## highest R square (explain more variability of medv) and the lowest RMSE (the model 
## are more accurate). Overall, in this report, we will choose the Tree model as the 
## best performance model. 

# ----------------------------------------

# Further thinking
## When the observed medv is equal to 50, all the variables show a relative large 
## bias.
## if we remove these values:
test = test[test$medv != 50, ]

# Test the test set by using the best model -- Tree Model 
par(mfrow = c(1,3))
x = test[,-14]
y = log(test$medv)
predictions_test = predict(regTree_best, newdata = x)
Performane(y, predictions_test, test, 13)
##    RMSE   Rsquare AdjRsquare
##1 3.88535 0.7376172  0.6955063
plot(exp(y), exp(predictions_test), col = rgb(0.4, 0.5,0.5), pch = 19,
     main  = 'Tree Models', xlab = 'Observed', ylab = 'Predicted')
abline(c(0,1))


# Test the test set by using the LASSO model 
x = model.matrix(log(medv)~.-1,data=test)
y = log(test$medv)
predictions_test <- predict(lasso_model, newx = x)
Performane(y, predictions_test, test, 13)
##        RMSE   Rsquare AdjRsquare
##  1 3.523647 0.7841958  0.7495605
plot(exp(y), exp(predictions_test), pch = 19, col = rgb(0.4, 0.5,0.5),
     main  = 'LASSO Models', xlab = 'Observed', ylab = 'Predicted')
abline(c(0,1))


# Test the test set by using AIC model 
x = test[,-14]
y = log(test$medv)
predictions_test = predict(backwardStepwise, newdata = x)
Performane(y, predictions_test, test, 13)
##       RMSE   Rsquare AdjRsquare
## 1 3.526118 0.7838929  0.7492091
plot(exp(y), exp(predictions_test), col = rgb(0.4, 0.5,0.5),pch = 19, 
     main  = 'AIC Models', xlab = 'Observed', ylab = 'Predicted')
abline(c(0,1))

## Then the AIC model becomes best-performance model in the traning set. 
## Further researcg can focus on the observation with medv equal to 50, and 
## explore why they are different and hard to predict.







