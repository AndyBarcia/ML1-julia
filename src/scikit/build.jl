using ScikitLearn;

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier
@sk_import neural_network: MLPClassifier

"""
Create a scikit-learn machine learning model based on the specified model type and hyperparameters.

# Arguments
- `modelType::Symbol`: The type of machine learning model to create. 
  Supported types include:
  - `:MLP`: Multi-layer Perceptron (Neural Network Classifier)
  - `:SVM`: Support Vector Machine Classifier
  - `:DecisionTree`: Decision Tree Classifier
  - `:kNN`: K-Nearest Neighbors Classifier

- `modelHyperparameters::Dict`: A dictionary of hyperparameters specific to the chosen model type

# Returns
- `PyCall.PyObject`: A scikit-learn compatible machine learning model instance
"""
function createScikitLearnModel(
    modelType::Symbol,
    modelHyperparameters::Dict,
)
    if modelType == :MLP || modelType == :ANN
        params = Dict{Symbol, Any}()
        params[:early_stopping] = true
        if haskey(modelHyperparameters, :topology)
            params[:hidden_layer_sizes] = modelHyperparameters[:topology]
        end
        if haskey(modelHyperparameters, :maxEpochs)
            params[:max_iter] = modelHyperparameters[:maxEpochs]
        end
        if haskey(modelHyperparameters, :learningRate)
            params[:learning_rate_init] = modelHyperparameters[:learningRate]
        end
        if haskey(modelHyperparameters, :maxEpochsVal)
            params[:n_iter_no_change] = modelHyperparameters[:maxEpochsVal]
        end
        if haskey(modelHyperparameters, :validationRatio)
            params[:validation_fraction] = modelHyperparameters[:validationRatio]
        end
        return MLPClassifier(; params...)
    elseif modelType == :SVM
        return  SVC(; modelHyperparameters...)
    elseif modelType == :DecisionTree
        return  DecisionTreeClassifier(; modelHyperparameters...)
    elseif modelType == :kNN
        return  KNeighborsClassifier(; modelHyperparameters...)
    else
        error("Unsupported model type: $modelType")
    end
end