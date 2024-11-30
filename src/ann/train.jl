using Printf;
using Flux;
using Statistics;
using Test;

"""
Compute metrics for a Flux model using confusion matrix and optional loss.

# Arguments
- `model::Flux.Chain`: Trained Flux model
- `inputs::AbstractArray{<:Real,2}`: Input features
- `targets::AbstractArray{Bool,2}`: One-hot encoded target labels
- `loss_fn::Union{Function, Nothing}=nothing`: Optional loss function to compute model loss

# Returns
- `Dict{Symbol, Any}`: Dictionary containing confusion matrix metrics and optional loss
    - Includes metrics such as `:accuracy`, `:errorRate`, `:recall`, etc.
    - Optionally includes `:loss` if loss function is provided
"""
function confusionMatrix(
    model::Flux.Chain, 
    inputs::AbstractArray{<:Real,2}, 
    targets::AbstractArray{Bool,2}, 
    loss_fn::Union{Function, Nothing}=nothing
)
    # Compute model predictions
    predictions = model(inputs')
    
    # Compute confusion matrix
    conf_matrix = confusionMatrix(
        predictions', 
        targets, 
        weighted=true
    )
    
    # Conditionally add loss if loss function is provided
    if loss_fn !== nothing
        model_loss = loss_fn(model, inputs', targets')
        conf_matrix[:loss] = model_loss
    end
    
    return conf_matrix
end

"""
Train a Multi-class Classification Artificial Neural Network with metrics tracking and early 
stopping criteria.

# Arguments
- `topology::AbstractArray{<:Int,1}`: Network layer topology defining neurons in each layer
- `trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}`: Tuple of 
    training inputs and targets
- `validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=<empty>`: Optional 
    validation dataset
- `testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=<empty>`: Optional test 
    dataset
- `transferFunctions::Union{AbstractArray{<:Function,1}, Function} = σ`: Transfer functions for 
    each layer, or a single transfer function for all layers.
- `maxEpochs::Int=1000`: Maximum number of training epochs
- `minLoss::Real=0.0`: Minimum loss threshold to stop training
- `learningRate::Real=0.01`: Learning rate for the optimizer
- `maxEpochsVal::Int=20`: Maximum epochs without improvement before early stopping
- `verbose::Bool=false`: Flag to print training progress
- `printFrequency::Int=10`: The number of iterations between prints.

# Returns
- `Tuple{Flux.Chain, Dict{Symbol, Vector{Dict{Symbol, Any}}}}`: 
    - Best trained neural network model
    - Dictionary of metrics for training, validation, and test datasets. Each entry contains a 
        vector of metrics for each epoch
"""
function trainClassANN(
    topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=
    (Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0, 0)),
    testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=
    (Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0, 0)),
    transferFunctions::Union{AbstractArray{<:Function,1}, Function} = σ,
    maxEpochs::Int=1000, 
    minLoss::Real=0.0, 
    learningRate::Real=0.01,
    maxEpochsVal::Int=20, 
    verbose::Bool=false,
    printFrequency::Int=10
)
    (train_inputs, train_targets) = trainingDataset
    (val_inputs, val_targets) = validationDataset
    (test_inputs, test_targets) = testDataset

    # Ensure input sets and their corresponding targets have the same number of entries.
    @assert(size(train_inputs, 1) == size(train_targets, 1))
    @assert(size(val_inputs, 1) == size(val_targets, 1))
    @assert(size(test_inputs, 1) == size(test_targets, 1))

    # Ensure all targets have the same number of columns.
    @assert(size(train_targets, 2) == size(test_targets, 2))
    if (!isempty(val_inputs))
        @assert(size(train_targets, 2) == size(val_targets, 2))
    end

    # Define the loss function
    loss(model, x, y) = (size(y, 1) == 1) ? 
        Flux.Losses.binarycrossentropy(model(x), y) : 
        Flux.Losses.crossentropy(model(x), y)

    # We define the ANN
    ann = buildClassANN(
        size(train_inputs, 2), 
        topology, 
        size(train_targets, 2),
        transferFunctions=transferFunctions
    )
    
    # Metrics storage using a dictionary 
    metrics = Dict(
        :training => Vector{Dict{Symbol, Any}}(),
        :validation => Vector{Dict{Symbol, Any}}(),
        :test => Vector{Dict{Symbol, Any}}()
    )

    # Initialize the best ANN
    minValidationLoss = Inf
    bestAnn = deepcopy(ann)

    # Initialize counters
    numEpoch = 0
    numEpochsValidation = 0

    # Initial metrics computation
    push!(metrics[:training], confusionMatrix(ann, train_inputs, train_targets, loss))
    if !isempty(val_inputs)
        push!(metrics[:validation], confusionMatrix(ann, val_inputs, val_targets, loss))
    end
    if !isempty(test_inputs)
        push!(metrics[:test], confusionMatrix(ann, test_inputs, test_targets, loss))
    end

    # Optional initial output
    if verbose
        println("Epoch 0: Initial metrics computed")
    end

    # Define the optimazer for the network
    opt_state = Flux.setup(Adam(learningRate), ann)

    # Start the training until it reaches one of the stop criteria
    while (numEpoch < maxEpochs) && 
        (metrics[:training][end][:loss] > minLoss) && 
        (isempty(val_inputs) || (numEpochsValidation < maxEpochsVal))

        # Train a single iteration
        Flux.train!(loss, ann, [(train_inputs', train_targets')], opt_state)

        numEpoch += 1
        numEpochsValidation += 1

        # Compute and store training metrics
        push!(metrics[:training], confusionMatrix(ann, train_inputs, train_targets, loss))

        # Validation metrics and early stopping criteria with accuracy
        if (!isempty(val_inputs))
            val_metrics = confusionMatrix(ann, val_inputs, val_targets, loss)
            push!(metrics[:validation], val_metrics)

            # Early stopping and best model selection
            if val_metrics[:accuracy] > metrics[:validation][end-1][:accuracy]
                bestAnn = deepcopy(ann)
                minValidationLoss = val_metrics[:loss]
                numEpochsValidation = 0
            end
        end

        # Test metrics
        if (!isempty(test_inputs))
            push!(metrics[:test], confusionMatrix(ann, test_inputs, test_targets, loss))
        end

        # Optional verbose output
        if verbose && (numEpoch % printFrequency == 0 || numEpoch == 1)
            @printf("Epoch: %d\n", numEpoch)
            @printf("Training - Loss: %.4f, Accuracy: %.4f\n", 
                metrics[:training][end][:loss], 
                metrics[:training][end][:accuracy])
            
            if !isempty(val_inputs)
                @printf("Validation - Loss: %.4f, Accuracy: %.4f\n", 
                    metrics[:validation][end][:loss], 
                    metrics[:validation][end][:accuracy])
            end
            
            if !isempty(test_inputs)
                @printf("Test - Loss: %.4f, Accuracy: %.4f\n", 
                    metrics[:test][end][:loss], 
                    metrics[:test][end][:accuracy])
            end
        end
    end

    # Return results
    if !isempty(val_inputs)
        if verbose
            println("Returning best ANN from epoch ", numEpoch - maxEpochsVal)
        end
        return (bestAnn, metrics)
    else
        if verbose
            println("Returning last ANN")
        end
        return (ann, metrics)
    end
end;

"""
Train a Binary Classification Artificial Neural Network with metrics tracking and early 
stopping criteria.

# Arguments
- `topology::AbstractArray{<:Int,1}`: Network layer topology defining neurons in each layer
- `trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}`: Tuple of 
    training inputs and targets
- `validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=<empty>`: Optional 
    validation dataset
- `testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=<empty>`: Optional test 
    dataset
- `transferFunctions::Union{AbstractArray{<:Function,1}, Function} = σ`: Transfer functions for 
    each layer, or a single transfer function for all layers.
- `maxEpochs::Int=1000`: Maximum number of training epochs
- `minLoss::Real=0.0`: Minimum loss threshold to stop training
- `learningRate::Real=0.01`: Learning rate for the optimizer
- `maxEpochsVal::Int=20`: Maximum epochs without improvement before early stopping
- `verbose::Bool=false`: Flag to print training progress
- `printFrequency::Int=10`: The number of iterations between prints.

# Returns
- `Tuple{Flux.Chain, Dict{Symbol, Vector{Dict{Symbol, Any}}}}`: 
    - Best trained neural network model
    - Dictionary of metrics for training, validation, and test datasets. Each entry contains a 
        vector of metrics for each epoch
"""
function trainClassANN(
    topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}=
    (Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0)),
    testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}=
    (Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0)),
    transferFunctions::Union{AbstractArray{<:Function,1}, Function} = σ,
    maxEpochs::Int=1000, 
    minLoss::Real=0.0, 
    learningRate::Real=0.01,
    maxEpochsVal::Int=20, 
    verbose::Bool=false,
    printFrequency::Int=10
)

    (train_inputs, train_targets) = trainingDataset
    (val_inputs, val_targets) = validationDataset
    (test_inputs, test_targets) = testDataset

    train_targets = hcat(
        train_targets,   # First column for true values
        .!train_targets  # Second column for false values
    )
    val_targets = hcat(
        val_targets,   # First column for true values
        .!val_targets  # Second column for false values
    )
    test_targets = hcat(
        test_targets,   # First column for true values
        .!test_targets  # Second column for false values
    )

    return trainClassANN(
        topology,
        (train_inputs, train_targets);
        validationDataset=(val_inputs, val_targets),
        testDataset=(test_inputs, test_targets),
        transferFunctions=transferFunctions,
        maxEpochs=maxEpochs,
        minLoss=minLoss,
        learningRate=learningRate,
        maxEpochsVal=maxEpochsVal,
        verbose=verbose,
        printFrequency=printFrequency
    )
end;

"""
Train a Multi-class Classification Artificial Neural Network with k-fold cross-validation.

# Arguments
- `topology::AbstractArray{<:Int,1}`: Network layer topology defining neurons in each layer
- `trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}`: Tuple of 
    training inputs and targets, where targets are already in two-column format
- `kFoldIndices::Array{Int64,1}`: Indices defining the k-fold cross-validation splits
- `transferFunctions::Union{AbstractArray{<:Function,1}, Function} = σ`: Transfer functions for 
    each layer, defaults to sigmoid activation
- `maxEpochs::Int=1000`: Maximum number of training epochs
- `minLoss::Real=0.0`: Minimum loss threshold to stop training
- `learningRate::Real=0.01`: Learning rate for the optimizer
- `repetitionsTraining::Int=1`: Number of times to repeat training for each fold
- `validationRatio::Real=0.0`: Ratio of training data to use for validation
- `maxEpochsVal::Int=20`: Maximum epochs without improvement before early stopping
- `verbose::Bool=false`: Flag to print training progress
- `metric::Symbol=:f1Score`: Metric used for model selection and reporting

# Returns
- `Tuple{Union{Nothing, Any}, Union{Nothing, Dict}, Dict}`: 
    - Best trained neural network model across all folds
    - Best model's metrics 
    - Summary metrics across all folds (mean, max, min, std)
"""
function trainClassANN(
    topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}},
    kFoldIndices::Array{Int64,1};
    transferFunctions::Union{AbstractArray{<:Function,1}, Function} = σ,
    maxEpochs::Int=1000, 
    minLoss::Real=0.0, 
    learningRate::Real=0.01,
    repetitionsTraining::Int=1,
    validationRatio::Real=0.0,
    maxEpochsVal::Int=20, 
    verbose::Bool=false,
    metric::Symbol=:f1Score
)

    (train_inputs, train_targets) = trainingDataset

    # Metric results (train loss, validation loss, and test loss) for each of the k folds.
    fold_results = Vector{Dict{Symbol,Dict}}()
    
    # Overall best model across all folds.
    best_overall_model = nothing
    best_overall_metrics = nothing
    best_overall_score = -Inf

    k_folds = maximum(kFoldIndices)
    for k in 1:k_folds
        # Get the inputs and targets specific for this k-fold.
        k_test_mask = kFoldIndices .== k
        k_test_inputs = train_inputs[k_test_mask, :]
        k_test_targets = train_targets[k_test_mask, :]

        # Prepare train-set for this k-fold.
        k_train_inputs = train_inputs[.!k_test_mask, :]
        k_train_targets = train_targets[.!k_test_mask, :]

        # Separate train-set into validation set if needed.
        # TODO Adjust validationRation based on total size of dataset.  
        if validationRatio > 0.0
            # Use holdOut function to split train data into train and validation
            train_indices, val_indices = holdOut(sum(.!k_test_mask), validationRatio)
            k_val_inputs = k_train_inputs[val_indices, :]
            k_val_targets = k_train_targets[val_indices, :]
            k_train_inputs = k_train_inputs[train_indices, :]
            k_train_targets = k_train_targets[train_indices, :]
        else
            k_val_inputs = Array{Float32}(undef, 0, 0)
            k_val_targets = falses(0, 0)
        end

        # Track best model and metrics for this fold across all repetitions.
        fold_best_model = nothing
        fold_best_metrics = nothing
        fold_best_score = -Inf

        for rep in 1:repetitionsTraining
            ann, metrics = trainClassANN(
                topology,
                (k_train_inputs, k_train_targets);
                validationDataset=(k_val_inputs, k_val_targets),
                testDataset=(k_test_inputs, k_test_targets),
                transferFunctions=transferFunctions,
                maxEpochs=maxEpochs,
                minLoss=minLoss,
                learningRate=learningRate,
                maxEpochsVal=maxEpochsVal
            )

            # Report training, validation and test metrics.
            if verbose
                @printf("Fold %d, Repetition %d - Train - Loss: %.4f, Metric: %.4f\n", 
                    k, rep,
                    metrics[:training][end][:loss], 
                    metrics[:training][end][metric])
                
                if !isempty(k_val_inputs)
                    @printf("Fold %d, Repetition %d - Validation - Loss: %.4f, Metric: %.4f\n", 
                        k, rep,
                        metrics[:validation][end][:loss], 
                        metrics[:validation][end][metric])
                end
                
                if !isempty(k_test_inputs)
                    @printf("Fold %d, Repetition %d - Test - Loss: %.4f, Metric: %.4f\n", 
                        k, rep,
                        metrics[:test][end][:loss], 
                        metrics[:test][end][metric])
                end
            end

            # If using validation sets, pick the best model across the folds based on
            # the given metric over the validation set. Otherwise, use the train set.
            scoring_set = validationRatio > 0.0 ? :validation : :training
            # Update best model for this fold 
            if metrics[scoring_set][end][metric] > fold_best_score
                fold_best_model = ann
                fold_best_metrics = metrics
                fold_best_score = metrics[scoring_set][end][metric]
            end

        end

        # Copy all metrics for all subsets at the final iteration
        fold_result = Dict()
        for subset in [:training, :validation, :test]
            # Only add subset if it's not empty
            if haskey(fold_best_metrics, subset) && !isempty(fold_best_metrics[subset])
                # Take the metrics from the last iteration
                fold_result[subset] = fold_best_metrics[subset][end]
            end
        end
        push!(fold_results, fold_result)

        # Update overall best model
        if fold_best_metrics[:test][end][metric] > best_overall_score
            best_overall_model = fold_best_model
            best_overall_metrics = fold_best_metrics
            best_overall_score = fold_best_metrics[:test][end][metric]
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
end;

"""
Train a Binary Classification Artificial Neural Network with k-fold cross-validation for 
a dataset with boolean targets represented as a 1D array.

# Arguments
- `topology::AbstractArray{<:Int,1}`: Network layer topology defining neurons in each layer
- `trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}`: Tuple of 
    training inputs and targets
- `kFoldIndices::Array{Int64,1}`: Indices defining the k-fold cross-validation splits
- `transferFunctions::Union{AbstractArray{<:Function,1}, Function} = σ`: Transfer functions for 
    each layer, defaults to sigmoid activation
- `maxEpochs::Int=1000`: Maximum number of training epochs
- `minLoss::Real=0.0`: Minimum loss threshold to stop training
- `learningRate::Real=0.01`: Learning rate for the optimizer
- `repetitionsTraining::Int=1`: Number of times to repeat training for each fold
- `validationRatio::Real=0.0`: Ratio of training data to use for validation
- `maxEpochsVal::Int=20`: Maximum epochs without improvement before early stopping
- `verbose::Bool=false`: Flag to print training progress
- `metric::Symbol=:f1Score`: Metric used for model selection and reporting

# Returns
- `Tuple{Union{Nothing, Any}, Union{Nothing, Dict}, Dict}`: 
    - Best trained neural network model across all folds
    - Best model's metrics 
    - Summary metrics across all folds (mean, max, min, std)
"""
function trainClassANN(
    topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}},
    kFoldIndices::Array{Int64,1};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, 
    minLoss::Real=0.0, 
    learningRate::Real=0.01, 
    repetitionsTraining::Int=1,
    validationRatio::Real=0.0, 
    maxEpochsVal::Int=20,
    verbose::Bool=false,
    metric::Symbol=:f1Score
)
    
    (train_inputs, train_targets) = trainingDataset

    train_targets = hcat(
        train_targets,   # First column for true values
        .!train_targets  # Second column for false values
    )

    return trainClassANN(
        topology,
        (train_inputs, train_targets),
        kFoldIndices;
        transferFunctions,
        maxEpochs,
        minLoss,
        learningRate,
        repetitionsTraining,
        validationRatio,
        maxEpochsVal,
        verbose,
        metric
    )
end;

@testset "trainClassANN" begin
    # Set seed for repetibility
    Random.seed!(42)
    
    n_samples = 200  # Total number of samples
    n_features = 4    # Number of input features

    # Generate dummy dataset. The target is based on 
    # whether the first column is greater than 0.5
    input_data  = rand(Float32, n_samples, n_features + 1)    
    output_data = vec(input_data[:, 1] .> 0.5)

    # Split the inputs and targets
    train_idx, test_idx = holdOut(size(input_data, 1), 0.1)

    train_input  = input_data[train_idx, :]
    train_output = output_data[train_idx]

    test_input  = input_data[test_idx, :]
    test_output = output_data[test_idx]

    # Calculate normalization parameters (mean and std) from training inputs
    μ, σ = compute_μσ(train_input)
    # Normalize the data
    train_input = normalize(train_input, μ, σ)
    test_input  = normalize(test_input, μ, σ)

    # Small test topology
    topology = [3]

    # Train the small topology. The results should be
    # at least moderately good, or at least sligtly
    # better than random chance.
    bestAnn, metrics = trainClassANN(
        topology,
        (train_input, train_output);
        testDataset = (test_input, test_output),
        maxEpochs = 100, 
        learningRate = 0.01,
    )
    @test metrics[:test][end][:accuracy] > 0.6
    @test metrics[:test][end][:errorRate] < 0.4
    @test metrics[:test][end][:recall] > 0.6
    @test metrics[:test][end][:specificity] > 0.6
    @test metrics[:test][end][:posPredValue] > 0.6
    @test metrics[:test][end][:negPredValue] > 0.6
    @test metrics[:test][end][:f1Score] > 0.6
    
end

@testset "trainClassANN crossvalidation" begin
    # Set seed for repetibility
    Random.seed!(42)
    
    n_samples = 200  # Total number of samples
    n_features = 4    # Number of input features

    # Generate dummy dataset. The target is based on 
    # whether the first column is greater than 0.5
    train_input  = rand(Float32, n_samples, n_features + 1)    
    train_output = vec(train_input[:, 1] .> 0.5)

    # Create indices for kfolds
    crossvalidation_indices = crossvalidation(train_output, 10)

    # Small test topology
    topology = [3]

    # Train the small topology. The results should be
    # at least moderately good, or at least sligtly
    # better than random chance.
    bestAnn, bestAnnMetrics, metrics = trainClassANN(
        topology,
        (train_input, train_output),
        crossvalidation_indices;
        validationRatio = 0.2,
        maxEpochs = 100, 
        learningRate = 0.01,
        metric = :f1Score
    )
    @test metrics[:test][:accuracy][:mean] > 0.6
    @test metrics[:test][:errorRate][:mean] < 0.4
    @test metrics[:test][:recall][:mean] > 0.6
    @test metrics[:test][:specificity][:mean] > 0.6
    @test metrics[:test][:posPredValue][:mean] > 0.6
    @test metrics[:test][:negPredValue][:mean] > 0.6
    @test metrics[:test][:f1Score][:mean] > 0.6
    
end