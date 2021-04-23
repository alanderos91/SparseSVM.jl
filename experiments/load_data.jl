function load_data(fname; seed::Int=1234)
    if fname == "iris"
        fname = "data/iris.data"
        has_header = false
        cols = [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth, :Class]
        colidx = collect(1:4)
        shuffle_samples = true
    elseif fname == "letter-recognition"
        fname = "data/letter-recognition.data"
        has_header = false
        cols = [:Class, :xbox, :ybox, :width, :height, :pixels, :xbar, :ybar, :x2bar, :y2bar, :xybar, :x2ybr, :xy2br, :xedge, :xegvy, :yedge, :yegvx]
        colidx = collect(2:17)
        shuffle_samples = false
    else
        error("Unknown data set $(fname).")
    end

    data = CSV.read(fname, DataFrame, header=has_header)
    rename!(data, cols)
    classes = unique(data.Class)

    if shuffle_samples
        perm = Random.randperm(MersenneTwister(seed), size(data, 1))
        for col in eachcol(data)
            permute!(col, perm)
        end
    end

    X = Matrix{Float64}(data[:, colidx])

    return data, X, classes
end