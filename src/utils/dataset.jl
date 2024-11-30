using DataFrames
using Statistics
using Printf
using Plots
using StatsPlots

function dataset_to_matrix(
    df::DataFrame, 
    target_column::Symbol, 
    input_type::Type=Float64
)
    @assert(target_column in propertynames(df), "Target column doesn't exist in DataFrame.")

    # Convert the input columns to a matrix of the specified type
    input_matrix = Matrix{input_type}(df[:, Not(target_column)])
    
    # Extract the target column as a vector. We don't convert to
    # any type as the output can be of arbtriary type.
    output_vector = df[:, target_column]
    
    return input_matrix, output_vector
end

function value_counts(df::DataFrame, target_variable::Symbol)
    counts = combine(groupby(df, target_variable), nrow => :Count)
    counts[!, :Percentage] = 100 .* counts[!, :Count] ./ sum(counts[!, :Count])
    return counts
end

function plot_value_counts(
    df::DataFrame, 
    target_variable::Symbol; 
    sort::Bool=true,
    kwargs...
)
    counts = value_counts(df, target_variable)
    if sort
        sort!(counts, :Count, rev=true)
    end

    p = bar(
        string.(counts[!, target_variable]), 
        counts[!, :Count],
        title = "Value Counts for $(string(target_variable))",
        ylabel = "Count",
        label = "Count",
        xticks = :auto,
        rotate = true,
        legend = false
    )

    # Disable scientific notation
    yformatter = x -> string(Int(round(x)))
    plot!(p, yformatter=yformatter)

    # Add percentage annotations below bars
    for i in 1:nrow(counts)
        annotate!(p, 
            i-0.45, 
            counts[i, :Count] - maximum(counts[!, :Count]) * 0.05,  # Offset above bar
            text("$(round(counts[i, :Percentage], digits=1))%", :black, :center)
        )
    end
    
    return p
end

function count_nulls(df::DataFrame)
    return DataFrame(
        Column = names(df), 
        Nulls = [sum(ismissing.(df[!, col])) for col in names(df)],
        Percentage = [100 * sum(ismissing.(df[!, col])) / nrow(df) for col in names(df)]
    )
end

function plot_null_counts(df::DataFrame; sort::Bool=true, kwargs...)
    null_counts = count_nulls(df)    
    if sort
        sort!(null_counts, :Nulls, rev=true)
    end

    p = bar(
        string.(null_counts[!, :Column]), 
        null_counts[!, :Nulls],
        title = "Null Counts by Column",
        ylabel = "Count",
        label = "Count",
        xticks = :auto,
        rotate = true,
        legend = false
    )

    # Disable scientific notation
    yformatter = x -> string(Int(round(x)))
    plot!(p, yformatter=yformatter)

    # Add percentage annotations below bars
    for i in 1:nrow(null_counts)
        annotate!(p, 
            i-0.45, 
            null_counts[i, :Nulls] - maximum(null_counts[!, :Nulls]) * 0.05,  # Offset above bar
            text("$(round(null_counts[i, :Percentage], digits=1))%", :black, :center)
        )
    end
    
    return p
end

function plot_heatmap(df::DataFrame, target_variable::Symbol)
    correlation_matrix = cor(Matrix(df[:, Not(target_variable)]))
    features = names(df[:, Not(target_variable)])

    heatmap(
        features, 
        features,
        correlation_matrix,
        color = :viridis,
        title = "Correlation Heatmap",
        aspect_ratio = :equal,
        xlabel = "Features",
        ylabel = "Features",
        size = (1000, 1000),
        colorbar_title = "Correlation",
        c = :coolwarm,
        xticks = (1:length(features), features),
        xrotation = 45,
        yticks = (1:length(features), features)
    )
end

function plot_boxplots(df::DataFrame; kwargs...)
    # Make boxplots only on numeric columns.
    numeric_cols = names(df, eltype.(eachcol(df)) .<: Number)
    
    if isempty(numeric_cols)
        error("No numeric columns found in the DataFrame.")
    end
    
    p = plot(
        layout=(1, length(numeric_cols)), 
        size=(200*length(numeric_cols), 400)
    )
    
    for (i, col) in enumerate(numeric_cols)
        boxplot!(
            df[!, col], 
            xlabel=string(col),
            subplot=i, 
            legend=false,
            kwargs...
        )
    end
    
    return p
end

"""
    scatterplot_dataset(x, y; col_x1=1, col_x2=2, colors, target_names=nothing)

Generates a scatter plot of a dataset with two selected columns of features, where each point 
is colored by its class. The method supports binary and multi-class datasets.

### Arguments:
- `x::AbstractMatrix{T}`: A 2D matrix (or any type of 2D array) representing the feature matrix 
    of the dataset, where each row is a data point and each column is a feature.
- `y::AbstractMatrix{T}`: A 2D matrix (or any type of 2D array) representing the labels of the 
    dataset, where each row corresponds to a data point and each column corresponds to the binary indicator for the class (one-hot encoding format).
- `col_x1::Int=1`: The index of the first feature to plot on the x-axis (default is `1`).
- `col_x2::Int=2`: The index of the second feature to plot on the y-axis (default is `2`).
- `colors::Vector{ColorTypes.Color}`: A vector containing the colors to be used for each class. 
    It should have the same length as the number of unique classes.
- `target_names::Vector{String}=nothing`: An optional vector of strings representing the class 
    labels. If provided, it must have the same length as the number of unique classes.
"""
function scatterplot_dataset(x, y; col_x1=1, col_x2=2, colors, target_names=nothing)
    num_classes = length(unique(colors))

    if !isnothing(target_names)
        @assert num_classes == length(target_names)
        label = target_names
    else
        label = [string(i) for i in 1:num_classes]
    end

    fig = plot()
    if (num_classes == 2)
        possitive_class = y[:,1].==1
        scatter!(fig, x[possitive_class,col_x1], x[possitive_class,col_x2], markercolor=colors[1], label=label[1])
        scatter!(fig, x[.!possitive_class,col_x1], x[.!possitive_class,col_x2], makercolor=colors[2], label=label[2])
    else
        for i in 1:num_classes
            index_class = y[:,i].==1
            scatter!(fig, x[index_class, col_x1], x[index_class, col_x2], markercolor=colors[i], label=label[i])
        end
    end
end

"""
Plots a histogram of a variable with a scaled PD (Probability of Default) rate overlay.

# Arguments
- df::DataFrame: The dataset containing the variables.
- target_var::Symbol: The target variable (e.g., :Default).
- study_var::Symbol: The variable to study (e.g., :LIMIT_BAL).
- n_bins::Int: The number of bins for the histogram (default is 30).

# Returns
- A plot with the histogram and scaled PD rate.
"""
function plot_histogram_with_pd_rate(df::DataFrame, target_var::Symbol, study_var::Symbol; n_bins::Int = 30)
    # Calculate bin edges and centers
    bin_edges = range(minimum(df[!, study_var]), stop=maximum(df[!, study_var]), length=n_bins + 1)
    bin_centers = [0.5 * (bin_edges[i] + bin_edges[i+1]) for i in 1:n_bins]

    # Initialize vectors for mean PD rate and frequencies
    mean_default_per_bin = Vector{Float64}(undef, n_bins)
    frequencies = Vector{Int}(undef, n_bins)

    # Calculate mean PD rate and frequencies for each bin
    for i in 1:n_bins
        lower = bin_edges[i]
        upper = bin_edges[i+1]
        values_in_bin = df[(df[!, study_var] .>= lower) .& (df[!, study_var] .< upper), :]

        frequencies[i] = nrow(values_in_bin)

        if !isempty(values_in_bin)
            mean_default_per_bin[i] = mean(values_in_bin[!, target_var]) * frequencies[i]  # Scale by frequency
        else
            mean_default_per_bin[i] = NaN
        end
    end

    # Plot the histogram using the data column, not the symbol
    @df df histogram(df[!, study_var], bins=n_bins, title="Distribución de $study_var con PD Rate", 
                     xlabel=string(study_var), ylabel="Frecuencia", legend=false)

    # Overlay the PD rate line
    plot!(bin_centers, mean_default_per_bin, lw=2, label="PD Rate (escala ajustada)", color=:red)
end

@testset "dataset_to_matrix" begin
    # Dummy dataframe
    df = DataFrame(
        ID = 1:10,
        Age = rand(18:65, 10),
        Salary = rand(30000:100000, 10),
    )

    # Change dataframe to a 2D matrix of floats.
    inputs, outputs = dataset_to_matrix(df,  :Salary, Float64)

    @test typeof(inputs)  <: AbstractArray{Float64, 2}
    @test typeof(outputs) <: AbstractArray{Int64, 1}
end