module ML1julia
    # This is needed because of this weird error. Otherwise, it causes a segmentation fault.
    # https://github.com/cstjean/ScikitLearn.jl/issues/50. Seriously, wtf Julia.
    __precompile__(false)

    export buildClassANN, trainClassANN, crossvalidation, holdOut, labelEncoding, 
            confusionMatrix, printConfusionMatrix, accuracy, oneHotEncoding, normalize, 
            compute_μσ, dataset_to_matrix, value_counts, plot_value_counts, count_nulls, 
            plot_null_counts, plot_heatmap, createScikitLearnModel, modelCrossValidation,
            trainClassEnsemble

    include("utils/holdOut.jl")
    include("utils/metrics.jl")
    include("utils/normalization.jl")
    include("utils/oneHot.jl")
    include("utils/crossvalidation.jl")
    include("utils/dataset.jl")
    include("ann/build.jl")
    include("ann/train.jl")
    include("scikit/build.jl")
    include("scikit/train.jl")
    include("scikit/ensemble.jl")
end