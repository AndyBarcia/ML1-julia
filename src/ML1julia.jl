module ML1julia
    export buildClassANN, trainClassANN, crossvalidation, holdOut, labelEncoding, 
            confusionMatrix, printConfusionMatrix, accuracy, oneHotEncoding, normalize, 
            compute_μσ, crossvalidation, dataset_to_matrix, value_counts, plot_value_counts,
            count_nulls, plot_null_counts, plot_heatmap, createScikitLearn, modelCrossValidation

    include("utils/holdOut.jl")
    include("utils/metrics.jl")
    include("utils/normalization.jl")
    include("utils/oneHot.jl")
    include("utils/crossvalidation.jl")
    include("utils/dataset.jl")
    include("ann/build.jl")
    include("ann/train.jl")
    include("scikit/train.jl")
    #include("scikit/ensemble.jl")
end