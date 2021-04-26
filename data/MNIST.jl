using MLDatasets, DataFrames, FileIO

datadir = "data"
if !isfile("t10k-labels-idx1-ubyte.gz") || !isfile("train-labels-idx1-ubyte.gz")
    MNIST.download(datadir)
end

Xtrain, ytrain = MNIST.traindata(dir=datadir)
Xtest, ytest = MNIST.testdata(dir=datadir)
Xdata = [reshape(Xtrain, 60000, 784); reshape(Xtest, 10000, 784)]
ydata = [ytrain; ytest]

DataFrame([Xdata ydata], :auto) |> FileIO.save("$(datadir)/MNIST.csv")
