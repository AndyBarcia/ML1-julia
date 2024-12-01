using Test;
using ScikitLearn;

@sk_import ensemble: VotingClassifier
@sk_import ensemble: StackingClassifier

function trainClassEnsemble(
    estimators::AbstractArray{Symbol,1},
    modelsHyperParameters::AbstractArray{<:Dict,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
    crossValidationIndices::Array{Int64,1};
    verbose::Bool=false,
    metric::Symbol=:f1Score,
    numJobs::Int64=1
)

    (train_inputs, train_targets) = trainingDataset

    # Metric results (train loss, validation loss, and test loss) for each of the k folds.
    fold_results = Vector{Dict{Symbol,Dict}}()
    
    # Overall best model across all folds.
    best_overall_model = nothing
    best_overall_metrics = nothing
    best_overall_score = -Inf

    # Dictionary to count how many models of each type are added
    model_count = Dict{Symbol, Int}()
    
    # Define the base models with hyperparameters
    base_models = []
    for (model_type, hyperparams) in zip(estimators, modelsHyperParameters)
        # Initialize the counter for this model type if it doesn't exist
        if !haskey(model_count, model_type)
            model_count[model_type] = 0
        end
        model_count[model_type] += 1        
        model_name = "$(string(model_type))_$(model_count[model_type])"
        
        # Create scikitlearn model based on the specifcied hyperparameters.
        model = createScikitLearnModel(model_type, hyperparams)
        push!(base_models, (model_name, model))
    end

    k_folds = maximum(crossValidationIndices)
    for k in 1:k_folds
        # Get the inputs and targets specific for this k-fold.
        k_test_mask = crossValidationIndices .== k
        k_test_inputs = train_inputs[k_test_mask, :]
        k_test_targets = train_targets[k_test_mask, :]

        # Prepare train-set for this k-fold.
        k_train_inputs = train_inputs[.!k_test_mask, :]
        k_train_targets = train_targets[.!k_test_mask, :]

        # Initialize VotingClassifier with base models
        ensemble_model = VotingClassifier(
            estimators=deepcopy(base_models),
            n_jobs=numJobs,
            voting="hard"
        )

        # Train the ensemble model. ScikitLearn models want label encoding only 
        # for some reason. Just label encode them.
        k_train_targets_labels = labelEncoding(k_train_targets)
        fit!(ensemble_model, k_train_inputs, k_train_targets_labels)

        # Test the model on both sets for keeping of metrics.
        metrics = Dict{Symbol, Dict{Symbol, Any}}()
        metrics[:training] = confusionMatrix(ensemble_model, k_train_inputs, k_train_targets)
        metrics[:test] = confusionMatrix(ensemble_model, k_test_inputs, k_test_targets)

        # Register metrics for this fold.
        push!(fold_results, metrics)

        # Report metrics if needed
        if verbose
            @printf("Fold %d - Train - Loss: %.4f, Metric: %.4f\n", 
                k, rep,
                metrics[:training][:loss], 
                metrics[:training][metric])
            
            if !isempty(k_test_inputs)
                @printf("Fold %d - Test - Loss: %.4f, Metric: %.4f\n", 
                    k, rep,
                    metrics[:test][:loss], 
                    metrics[:test][metric])
            end
        end

        # Update overall best model on the test set.
        if metrics[:test][metric] > best_overall_score
            best_overall_model = ensemble_model
            best_overall_metrics = metrics
            best_overall_score = metrics[:test][metric]
        end
    end

    # Initialize resume_metrics of all folds
    resume_metrics = Dict{Symbol,Dict{Symbol,Dict{Symbol,Float32}}}()
    # Go through each subset (training, validation, test)
    for subset in [:training, :validation, :test]
        # Skip if no fold results for this subset
        subset_results = [fold[subset] for fold in fold_results if haskey(fold, subset)]
        if !isempty(subset_results)
            # Initialize metrics for this subset
            resume_metrics[subset] = Dict{Symbol,Dict{Symbol,Float32}}()
            
            # Compute mean, max, min for each metric
            for metric in keys(subset_results[1])
                # Collect all values for this metric across folds
                metric_values = [result[metric] for result in subset_results]
                # Compute mean, maximum, etc only if the metrics are numbers.
                if all(x -> isa(x, Real), metric_values)
                    resume_metrics[subset][metric] = Dict(
                        :mean => mean(metric_values),
                        :max => maximum(metric_values),
                        :min => minimum(metric_values),
                        :std => std(metric_values)
                    )
                end
            end
        end
    end

    return best_overall_model, best_overall_metrics, resume_metrics
end

function trainClassEnsemble(
    estimators::AbstractArray{Symbol,1},
    modelsHyperParameters::AbstractArray{<:Dict,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
    crossValidationIndices::Array{Int64,1};
    verbose::Bool=false,
    metric::Symbol=:f1Score
)

    (train_inputs, train_targets) = trainingDataset

    train_targets = hcat(
        train_targets,   # First column for true values
        .!train_targets  # Second column for false values
    )

    return trainClassEnsemble(
        estimators,
        modelsHyperParameters,
        (train_inputs, train_targets),
        crossValidationIndices;
        verbose,
        metric
    )
end

function trainClassEnsemble(
    estimator::Symbol,
    modelsHyperParameters::Dict,
    numEstimators::Int,
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
    crossValidationIndices::Array{Int64,1};
    verbose::Bool=false,
    metric::Symbol=:f1Score
)
    estimators = fill(estimator, numEstimators)
    modelsHyperParametersList = [copy(modelsHyperParameters) for _ in 1:numEstimators]
    return trainClassEnsemble(
        estimators,
        modelsHyperParametersList,
        trainingDataset,
        crossValidationIndices;
        verbose,
        metric
    )
end

function trainClassEnsemble(
    estimator::Symbol,
    modelsHyperParameters::Dict,
    numEstimators::Int,
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
    crossValidationIndices::Array{Int64,1};
    verbose::Bool=false,
    metric::Symbol=:f1Score
)
    (train_inputs, train_targets) = trainingDataset

    train_targets = hcat(
        train_targets,   # First column for true values
        .!train_targets  # Second column for false values
    )

    return trainClassEnsemble(
        estimator,
        modelsHyperParameters,
        numEstimators,
        (train_inputs, train_targets),
        crossValidationIndices;
        verbose,
        metric
    )
end

@testset "trainClassEnsemble" begin
    # Set seed for repetibility
    Random.seed!(42)
    
    n_samples = 200  # Total number of samples
    n_features = 4    # Number of input features

    # Generate dummy dataset. The target is based on 
    # whether the first column is greater than 0.5
    train_input  = rand(Float32, n_samples, n_features + 1)    
    train_output = vec(train_input[:, 1] .> 0.5)

    train_output = oneHotEncoding(train_output)

    # Create indices for kfolds
    crossvalidation_indices = crossvalidation(train_output, 5)

    # Small test configuration
    knn_hyperparameters = Dict(:n_neighbors => 5)

    # Train 4 instances of the ANN configuration.
    bestEnsemble, bestEnsembleMetrics, metrics = trainClassEnsemble(
        [:kNN],
        [knn_hyperparameters],
        (train_input, train_output),
        crossvalidation_indices,
        numJobs=1
    )

    @test metrics[:test][:accuracy][:mean] > 0.6
    @test metrics[:test][:errorRate][:mean] < 0.4
    @test metrics[:test][:recall][:mean] > 0.6
    @test metrics[:test][:specificity][:mean] > 0.6
    @test metrics[:test][:posPredValue][:mean] > 0.6
    @test metrics[:test][:negPredValue][:mean] > 0.6
    @test metrics[:test][:f1Score][:mean] > 0.6
    
end