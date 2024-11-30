using Test;

"""
Perform one-hot encoding on a 1D array of data.

# Arguments
- `data::AbstractArray{<:Any,1}`: The input 1D array of data to be encoded.
- `classes::AbstractArray{<:Any,1}`: The array of unique classes or categories 
    present in the input data.

# Returns
- `::Matrix{Bool}`: A boolean matrix where each row represents an input data 
    point, and each column represents a unique class. The value at each position 
    is 1 if the data point belongs to the corresponding class, and 0 otherwise.
"""
function oneHotEncoding(data::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    encoded = zeros(Bool, length(data), length(classes))

    for (i, value) in enumerate(data)
        class_index = findfirst(==(value), classes)
        if class_index === nothing
            throw(ArgumentError("Value $value not found in classes array."))
        else
            encoded[i, class_index] = 1
        end
    end

    return encoded
end;

"""
Perform one-hot encoding on a 1D array of data, using all the different unique
elements as classes. Remember to use oneHotEncoding on the whole dataset before
splitting; otherwise we may miss classes that appear in one set but not in the
other.

# Arguments
- `data::AbstractArray{<:Any,1}`: The input 1D array of data to be encoded.

# Returns
- `::Matrix{Bool}`: A boolean matrix where each row represents an input data 
    point, and each column represents a unique class. The value at each position 
    is 1 if the data point belongs to the corresponding class, and 0 otherwise.
"""
oneHotEncoding(data::AbstractArray{<:Any,1}) = oneHotEncoding(data, unique(data))

"""
Convert a one-hot encoded 2D array back into its original categorical labels.

This function reverses one-hot encoding by identifying the index of the maximum 
value in each row of the input matrix. It assumes that each row represents an 
observation and each column represents a class.

# Arguments
- `targets::AbstractArray{Bool,2}`: A 2D boolean array where each row is a 
    one-hot encoded representation of a categorical label.

# Returns
- `::AbstractVector{Int}`: A vector of integer indices corresponding to the 
    original labels for each row of the input matrix.
"""
function labelEncoding(targets::AbstractArray{Bool,2})
    return vec([idx[2] for idx in argmax(targets, dims=2)])
end;

"""
Convert a boolean array of single-class labels back into their original 
representation.

# Arguments
- `targets::AbstractArray{Bool,1}`: A 1D boolean array representing single-class 
    labels.

# Returns
- `::AbstractVector{Bool}`: A vector of boolean values that directly correspond 
    to the original labels.
"""
function labelEncoding(targets::AbstractArray{Bool,1})
    return vec(targets)
end;

@testset "OneHotEncoding" begin
    # Test Simple example
    data = ["cat", "dog", "bird", "cat", "dog"]
    classes = ["cat", "dog", "bird"]
    expected = [
        1 0 0;
        0 1 0;
        0 0 1;
        1 0 0;
        0 1 0;
    ]
    @test oneHotEncoding(data, classes) == expected

    # Test data contains value not in classes
    @test_throws ArgumentError oneHotEncoding(["cat", "dog", "hamster"], ["cat", "dog"])
end