using Statistics;
using Test;
using Dates;

"""
Calculate the accuracy of binary classification or multi-class classification with
one-hot encoded outputs and targets. Each row is a single sample, while each column is a class.

# Arguments
- `outputs::AbstractArray{Bool,2}`: A boolean or real-valued matrix of predicted outputs
- `targets::AbstractArray{Bool,2}`: A boolean matrix of true target values

# Returns
- `Real`: Accuracy as a floating-point number (proportion of correct predictions)
"""
function accuracy(outputs::AbstractArray{Bool}, targets::AbstractArray{Bool})
    @assert size(outputs) == size(targets) "Outputs and targets must have the same shape."

    if ndims(targets) == 1
        return mean(targets[:, 1] .== outputs[:, 1])
    else
        @assert all(sum(outputs, dims=2) .== 1) "Each row in 'outputs' must be a valid one-hot encoding."
        @assert all(sum(targets, dims=2) .== 1) "Each row in 'targets' must be a valid one-hot encoding."
    
        return mean(all(targets .== outputs, dims=2))
    end
end;


"""
Compute a comprehensive confusion matrix and related metrics for binary classification.

# Arguments
- `outputs::AbstractArray{Bool,1}`: A boolean array of predicted outputs
- `targets::AbstractArray{Bool,1}`: A boolean array of true target values

# Returns
- `Dict{Symbol, Any}`: A dictionary with the following metrics:
    - `:accuracy`: Overall accuracy
    - `:errorRate`: Proportion of incorrect predictions
    - `:recall`: Proportion of true positives identified
    - `:specificity`: Proportion of true negatives identified
    - `:posPredValue`: Positive predictive value (precision)
    - `:negPredValue`: Negative predictive value
    - `:f1Score`: Harmonic mean of precision and recall
    - `:confMatrix`: Confusion matrix
"""
function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert size(outputs) == size(targets)
    
    # Compute the basic counts
    tps = sum(outputs .& targets)
    tns = sum((.~outputs) .& (.~targets))
    fns = sum((.~outputs) .& targets)
    fps = sum(outputs .& (.~targets))
    
    total = tns + tps + fns + fps
    
    if total > 0
        accuracy = (tps + tns) / total
        error_rate = (fns + fps) / total    
    else
        accuracy = 1.0
        error_rate = 0.0
    end;

    if fns+tps > 0
        recall = tps / (fns+tps)
    elseif fps == 0
        recall = 1.0
    else
        recall = 0.0
    end;
    if fps+tns > 0
        specificity = tns / (fps+tns)
    elseif fns == 0
        specificity = 1.0
    else
        specificity = 0.0
    end;
    if tps+fps > 0
        pos_pred_value = tps / (tps+fps)
    elseif fns == 0
        pos_pred_value = 1.0
    else
        pos_pred_value = 0.0
    end;
    if tns+fns > 0
        neg_pred_value = tns / (tns+fns)
    elseif fps == 0
        neg_pred_value = 1.0
    else
        neg_pred_value = 0.0
    end;
    if pos_pred_value != 0.0 && recall != 0.0
        f1_score = (2*pos_pred_value*recall) / (pos_pred_value+recall)
    else
        f1_score = 0.0
    end;
    
    # Construct the confusion matrix
    conf_matrix = [
        tns fps; 
        fns tps
    ]
    
    # Return results as a dictionary
    return Dict(
        :accuracy => accuracy,
        :errorRate => error_rate,
        :recall => recall,
        :specificity => specificity,
        :posPredValue => pos_pred_value,
        :negPredValue => neg_pred_value,
        :f1Score => f1_score,
        :confMatrix => conf_matrix
    )
end;

"""
Compute a comprehensive confusion matrix and related metrics for binary classification.

# Arguments
- `outputs::AbstractArray{<:Real,1}`: A real-valued array of predicted outputs.
- `targets::AbstractArray{Bool,1}`: A boolean array of true target values.
- `threshold::Real`: The thresshold to use to consider an output a true or negative class.

# Returns
- `Dict{Symbol, Any}`: A dictionary with the following metrics:
    - `:accuracy`: Overall accuracy
    - `:errorRate`: Proportion of incorrect predictions
    - `:recall`: Proportion of true positives identified
    - `:specificity`: Proportion of true negatives identified
    - `:posPredValue`: Positive predictive value (precision)
    - `:negPredValue`: Negative predictive value
    - `:f1Score`: Harmonic mean of precision and recall
    - `:confMatrix`: Confusion matrix
"""
function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    outputs = outputs .> threshold
    return confusionMatrix(outputs, targets)
end;

"""
Compute a comprehensive confusion matrix and related metrics for multi-class classification.

# Arguments
- `outputs::AbstractArray{Bool,2}`: A boolean array of predicted outputs. Each row
    is a single sample, and each column is a class.
- `targets::AbstractArray{Bool,2}`: A boolean array of true one-hot encoded targets.
- `weighted::Bool`: Whether to scale metrics based on the frequency of each class, or just report
    the mean metrics of all classes.

# Returns
- `Dict{Symbol, Any}`: A dictionary with the following metrics:
    - `:accuracy`: Overall accuracy
    - `:errorRate`: Proportion of incorrect predictions
    - `:recall`: Proportion of true positives identified
    - `:specificity`: Proportion of true negatives identified
    - `:posPredValue`: Positive predictive value (precision)
    - `:negPredValue`: Negative predictive value
    - `:f1Score`: Harmonic mean of precision and recall
    - `:confMatrix`: Confusion matrix
"""
function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)    
    @assert(size(outputs, 2) == size(targets, 2), "The number of columns 'outputs' and 'targets' must match.")
    
    # For the specific case of binary classification, call the other function.
    if size(outputs, 2) == 2
        return confusionMatrix(outputs[:,1], targets[:,1])
    end
        
    # Initialize metrics
    numClass = size(outputs, 2)
    sensitivity = zeros(Float64, numClass)
    specificity = zeros(Float64, numClass)
    ppv = zeros(Float64, numClass)
    npv = zeros(Float64, numClass)
    f1 = zeros(Float64, numClass)
    
    # Compute metrics
    for i in 1:numClass
        if any(targets[:, i])
            metrics = confusionMatrix(outputs[:, i], targets[:, i])
            sensitivity[i] = metrics[:recall]
            specificity[i] = metrics[:specificity]
            ppv[i] = metrics[:posPredValue]
            npv[i] = metrics[:negPredValue]
            f1[i] = metrics[:f1Score]
        end
    end

    # Initialize and populate matrix
    confusion_mat = zeros(Int, numClass, numClass)

    for i in 1:numClass
        for j in 1:numClass
            confusion_mat[i, j] = sum(targets[:, i] .& outputs[:, j])
        end
    end

    # Strategy
    if weighted
        class_weights = sum(targets, dims=1) ./ size(targets, 1)
        agg_sensitivity = sum(sensitivity .* class_weights) ./ size(targets, 2)
        agg_specificity = sum(specificity .* class_weights) ./ size(targets, 2)
        agg_ppv = sum(ppv .* class_weights) ./ size(targets, 2)
        agg_npv = sum(npv .* class_weights) ./ size(targets, 2)
        agg_f1 = sum(f1 .* class_weights) ./ size(targets, 2)
    else
        agg_sensitivity = mean(sensitivity)
        agg_specificity = mean(specificity)
        agg_ppv = mean(ppv)
        agg_npv = mean(npv)
        agg_f1 = mean(f1)
    end

    # accuracy and error
    accuracy_value = accuracy(outputs, targets)
    error_rate = 1.0 - accuracy_value

    return Dict(
        :accuracy => accuracy_value,
        :errorRate => error_rate,
        :recall => agg_sensitivity,
        :specificity => agg_specificity,
        :posPredValue => agg_ppv,
        :negPredValue => agg_npv,
        :f1Score => agg_f1,
        :confMatrix => confusion_mat
    )
end;

"""
Compute a comprehensive confusion matrix and related metrics for multi-class classification.

# Arguments
- `outputs::AbstractArray{<:Real,2}`: A real-valued matrix of predicted outputs. Each row
    is a single sample, and each column is a class.
- `targets::AbstractArray{Bool,2}`: A boolean matrix of true one-hot encoded targets. Each row
    is a single sample, and each column is a class.
- `weighted::Bool`: Whether to scale metrics based on the frequency of each class, or just report
    the mean metrics of all classes.
- `threshold::Real`: The thresshold to use to consider an output a true or negative class.

# Returns
- `Dict{Symbol, Any}`: A dictionary with the following metrics:
    - `:accuracy`: Overall accuracy
    - `:errorRate`: Proportion of incorrect predictions
    - `:recall`: Proportion of true positives identified
    - `:specificity`: Proportion of true negatives identified
    - `:posPredValue`: Positive predictive value (precision)
    - `:negPredValue`: Negative predictive value
    - `:f1Score`: Harmonic mean of precision and recall
    - `:confMatrix`: Confusion matrix
"""
function confusionMatrix(
    outputs::AbstractArray{<:Real,2}, 
    targets::AbstractArray{Bool,2}; 
    weighted::Bool=true,
    threshold::Real=0.5
)
    outputs = outputs .> threshold
    return confusionMatrix(outputs, targets; weighted=weighted)
end;

"""
Prints a comprehensive confusion matrix and related metrics for binary classification.

# Arguments
- `outputs::AbstractArray{Bool,1}`: A boolean array of predicted outputs
- `targets::AbstractArray{Bool,1}`: A boolean array of true target values
"""
function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    metrics = confusionMatrix(outputs, targets)

    TN = metrics[:confMatrix][1, 1]
    FP = metrics[:confMatrix][1, 2]
    FN = metrics[:confMatrix][2, 1]
    TP = metrics[:confMatrix][2, 2]

    accuracy = metrics[:accuracy]
    errorRate = metrics[:errorRate]
    recall = metrics[:recall]
    specificity = metrics[:specificity]
    posPredValue = metrics[:posPredValue]
    negPredValue = metrics[:negPredValue]
    fScore = metrics[:f1Score]

    println("             Predicted")
    println("            Pos     Neg")
    println("Actual Pos  $TP      $FN")
    println("       Neg  $FP      $TN")
    println("------------------------")
    println("Accuracy: $accuracy")
    println("Error rate: $errorRate")
    println("Recall: $recall")
    println("Specificity: $specificity")
    println("Precision: $posPredValue")
    println("Negative Predictive Value: $negPredValue")
    println("F1-Score: $fScore")
end;

"""
Prints a comprehensive confusion matrix and related metrics for binary classification.

# Arguments
- `outputs::AbstractArray{Bool,1}`: A boolean array of predicted outputs
- `targets::AbstractArray{Bool,1}`: A boolean array of true target values
"""
function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    outputs = outputs .> threshold
    return printConfusionMatrix(outputs, targets)
end;

"""
Prints a comprehensive confusion matrix and related metrics for binary classification.

# Arguments
- `outputs::AbstractArray{<:Real,1}`: A real-valued array of predicted outputs.
- `targets::AbstractArray{Bool,1}`: A boolean array of true target values.
- `threshold::Real`: The thresshold to use to consider an output a true or negative class.
"""
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert(all([in(output, unique(targets)) for output in outputs]))

    classes_targets = unique(targets)

    encoded_outputs = oneHotEncoding(outputs, classes_targets)
    encoded_targets = oneHotEncoding(targets, classes_targets)

    return confusionMatrix(encoded_outputs, encoded_targets; weighted=weighted)
end;

function getModelMetrics(
    model_results, 
    phase::Symbol=:test;
    metric_statistic::Symbol=:mean,
    average_results::Bool=false
)    
    metric_names = keys(model_results[1][:metrics][phase])
    if average_results
        metrics_data = Dict{Symbol,Float32}()
    else
        metrics_data = Dict{Symbol,Vector{Float32}}()
    end;

    for metric in metric_names
        metric_result = Vector{Float32}()
        for model_data in model_results
            push!(metric_result, model_data[:metrics][phase][metric][metric_statistic])
        end

        if average_results
            metrics_data[metric] = mean(metric_result)
        else
            metrics_data[metric] = metric_result
        end
    end

    return metrics_data
end;

function plotMetrics(
    model_name, 
    metric_models::Dict{Symbol, Vector{Float32}},
)
    for (metric, values) in metric_models
        p = bar(
            values, 
            title="$model_name $metric Comparison", 
            xlabel="Index", 
            ylabel="Value", 
            legend=false
        )
        display(p)
    end
end;

function plotMetricStatistics(
    model_name, 
    metrics::Dict{Symbol, Dict{Symbol, Float32}},
)
    for (metric, metric_statistics) in metrics
        p = bar(
            string.(collect(keys(metric_statistics))),
            collect(values(metric_statistics)),
            title="$model_name $metric Comparison", 
            ylabel="Value", 
            legend=false
        )
        display(p)
    end
end;

function plotMetricsAllModels(metrics::Dict{Symbol, Dict{Symbol, Float32}})
    # Get names of all metrics.
    metric_names = keys(first(values(metrics))) |> collect
    # Get names of all models.
    model_names = collect(keys(metrics))
    
    # Loop through each metric and create a bar plot
    for metric in metric_names
        # Extract data for the current metric
        values = [metrics[model][metric] for model in model_names]
        
        # Create a bar plot
        p = bar(
            string.(model_names), 
            values, 
            label=string(metric), 
            title="Comparison of Models: $metric",
            xlabel="Models",
            ylabel=string(metric) * " Values",
            legend=false,
            bar_width=0.6
        )
        display(p)
    end
end;

function meanTime(training_times::AbstractArray{<:Any,1})
    return Dates.Millisecond(round(mean([Dates.value(t) for t in training_times])))
end;

function plotTimes(
    model_name::String,
    metric_times::AbstractArray{<:Any,1},
)
    bar(
        [Dates.value(t) / 1e3 for t in metric_times],
        title="$model_name training times Comparison", 
        ylabel="Training time (s)", 
        legend=false
    )
end;

function plotTimes(
    model_name::AbstractArray{String,1},
    metric_times::AbstractArray{<:Any,1},
)
    bar(
        model_name,
        [Dates.value(t) / 1e3 for t in metric_times],
        title="Training times Comparison", 
        ylabel="Training time (s)", 
        legend=false
    )
end;

@testset "accuracy" begin
    # Test 1: 1D arrays
    outputs_1d = Bool[1, 0, 1, 1]
    targets_1d = Bool[1, 0, 0, 1]
    @test accuracy(outputs_1d, targets_1d) ≈ 0.75

    # Test 2: Perfect accuracy (single row)
    outputs = Bool[1 0 0; 0 1 0; 0 0 1]
    targets = Bool[1 0 0; 0 1 0; 0 0 1]
    @test accuracy(outputs, targets) ≈ 1.0

    # Test 3: Mixed accuracy
    outputs = Bool[1 0 0; 0 1 0; 0 1 0]
    targets = Bool[1 0 0; 0 1 0; 0 0 1]
    @test accuracy(outputs, targets) ≈ 2/3

    # Test 4: Invalid outputs (not one-hot encoded)
    # Some samples have multiple classes
    outputs = Bool[1 1 0; 0 1 0; 0 0 1]
    targets = Bool[1 0 0; 0 1 0; 0 0 1]
    @test_throws AssertionError accuracy(outputs, targets)
end

@testset "confusionMatrix binary" begin
    # Test 1: Perfect prediction (all true positives)
    outputs = [true, true, true, true]
    targets = [true, true, true, true]
    result = confusionMatrix(outputs, targets)
    @test result[:accuracy] == 1.0
    @test result[:errorRate] == 0.0
    @test result[:recall] == 1.0
    @test result[:specificity] == 1.0
    @test result[:posPredValue] == 1.0
    @test result[:negPredValue] == 1.0
    @test result[:f1Score] == 1.0
    @test result[:confMatrix] == [0 0; 0 4]

    # Test 2: Perfect prediction (all true negatives)
    outputs = [false, false, false, false]
    targets = [false, false, false, false]
    result = confusionMatrix(outputs, targets)
    @test result[:accuracy] == 1.0
    @test result[:errorRate] == 0.0
    @test result[:recall] == 1.0
    @test result[:specificity] == 1.0
    @test result[:posPredValue] == 1.0 # Undefined but handled
    @test result[:negPredValue] == 1.0
    @test result[:f1Score] == 1.0 # Undefined but handled
    @test result[:confMatrix] == [4 0; 0 0]

    # Test 3: Random prediction with mixed results
    outputs = [true, true, false, false]
    targets = [true, false, true, false]
    result = confusionMatrix(outputs, targets)
    @test result[:accuracy] == 0.5
    @test result[:errorRate] == 0.5
    @test result[:recall] == 0.5
    @test result[:specificity] == 0.5
    @test result[:posPredValue] == 0.5
    @test result[:negPredValue] == 0.5
    @test result[:f1Score] ≈ 0.5
    @test result[:confMatrix] == [1 1; 1 1]

    # Test 4: Edge case (all false positives)
    outputs = [true, true, true, true]
    targets = [false, false, false, false]
    result = confusionMatrix(outputs, targets)
    @test result[:accuracy] == 0.0
    @test result[:errorRate] == 1.0
    @test result[:recall] == 0.0
    @test result[:specificity] == 0.0
    @test result[:posPredValue] == 0.0
    @test result[:negPredValue] == 0.0
    @test result[:f1Score] == 0.0
    @test result[:confMatrix] == [0 4; 0 0]

    # Test 5: Edge case (all false negatives)
    outputs = [false, false, false, false]
    targets = [true, true, true, true]
    result = confusionMatrix(outputs, targets)
    @test result[:accuracy] == 0.0
    @test result[:errorRate] == 1.0
    @test result[:recall] == 0.0
    @test result[:specificity] == 0.0 # undefined but handled
    @test result[:posPredValue] == 0.0
    @test result[:negPredValue] == 0.0
    @test result[:f1Score] == 0.0
    @test result[:confMatrix] == [0 0; 4 0]

    # Test 6: Input size mismatch (should throw an assertion error)
    outputs = [true, false]
    targets = [true]
    @test_throws AssertionError confusionMatrix(outputs, targets)

    # Test 7: Empty inputs (edge case)
    outputs = Bool[]
    targets = Bool[]
    result = confusionMatrix(outputs, targets)
    @test result[:accuracy] == 1.0 # No errors, no mismatches
    @test result[:errorRate] == 0.0
    @test result[:confMatrix] == [0 0; 0 0]

    # Test 8: Mix of true and false
    outputs = [true, false, true, false, false]
    targets = [true, false, false, true, false]
    result = confusionMatrix(outputs, targets)
    @test result[:accuracy] == 0.6
    @test result[:errorRate] == 0.4
    @test result[:confMatrix] == [2 1; 1 1]
end

@testset "confusionMatrix multiclass" begin
    # Test 1: Perfect prediction (one-hot encoded, all correct)
    outputs = Bool[
        true false false;
        false true false;
        false false true
    ]
    targets = Bool[
        true false false;
        false true false;
        false false true
    ]
    result = confusionMatrix(outputs, targets)
    @test result[:accuracy] == 1.0
    @test result[:errorRate] == 0.0
    @test result[:recall] == 1.0
    @test result[:specificity] == 1.0
    @test result[:posPredValue] == 1.0
    @test result[:negPredValue] == 1.0
    @test result[:f1Score] == 1.0
    @test result[:confMatrix] == [1 0 0; 0 1 0; 0 0 1]

    # Test 2: Completely incorrect predictions
    outputs = Bool[
        false true  false;
        false false true;
        true  false false
    ]
    targets = Bool[
        true false false;
        false true false;
        false false true
    ]
    result = confusionMatrix(outputs, targets)
    @test result[:accuracy] == 0.0
    @test result[:errorRate] == 1.0
    @test result[:recall] == 0.0
    @test result[:specificity] == 0.5 
    @test result[:posPredValue] == 0.0
    @test result[:negPredValue] == 0.5
    @test result[:f1Score] == 0.0
    @test result[:confMatrix] == [0 1 0; 0 0 1; 1 0 0]
end