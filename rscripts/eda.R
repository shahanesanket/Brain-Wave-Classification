dataset.test1 = read.csv('../data/training_data_complete.csv')
plot(density(dataset.test1[dataset.test1$X7.00E.00==7,1]))
lines(density(dataset.test1[dataset.test1$X7.00E.00==2,1]))
lines(density(dataset.test1[dataset.test1$X7.00E.00==3,1]))
#lines(density(dataset.test1[,20]))

plot(density(dataset.test1[dataset.test1$X7.00E.00==7,50]))
lines(density(dataset.test1[dataset.test1$X7.00E.00==2,50]))
lines(density(dataset.test1[dataset.test1$X7.00E.00==3,50]))

plot(density(dataset.test1[dataset.test1$X7.00E.00==7,72]))
lines(density(dataset.test1[dataset.test1$X7.00E.00==2,72]))
lines(density(dataset.test1[dataset.test1$X7.00E.00==3,72]))

correlationMatrix = cor(dataset.test1[,-97])
correlationDataframe = data.frame(correlationMatrix)

for(i in seq(1,96,1)){
  for(j in seq(i,96,1)){
    if(abs(correlationMatrix[i,j])>0.70 && i!=j){
      print(paste(i,j,correlationMatrix[i,j]))
    }
  }
}

var(dataset.test1)
apply(dataset.test1,2,var)
