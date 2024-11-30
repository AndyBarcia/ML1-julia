using Printf;

"""
Prints the metrics in a formatted table for a specified phase (e.g., `:train`, `:test`, etc.).

# Arguments
- `metrics::Dict{Symbol, Dict{Symbol, T}}`: 
    A dictionary where the keys are phases (e.g., `:train`, `:test`) and the values are dictionaries 
    containing metric information.  Each metric is represented by a dictionary with keys such as 
    `:mean`, `:max`, `:min`, and `:std`.

- `phase::Symbol`: 
    The phase for which the metrics should be printed. Defaults to `:test`.

# Behavior
- If the values in the `metrics[phase]` dictionary are dictionaries (i.e., `T <: Dict`), the function 
    assumes the metrics have statistics like mean, max, min, and std. It prints these in a formatted table:
    ```
    Metric          | Mean            | Max             | Min             | Std             
    --------------- | --------------- | --------------- | ---------------
    metric_name     | mean_value      | max_value       | min_value       | std_value
    ```

- If the values in the `metrics[phase]` dictionary are scalars (e.g., `Real`), it prints only the 
    metric name and its value:
    ```
    Metric          | Value           
    --------------- | ---------------
    metric_name     | metric_value    
    ```
"""
function printMetrics(metrics::Dict{Symbol, Dict{Symbol, T}}, phase::Symbol=:test) where T
    if T<:Dict
        println("Metric          | Mean            | Max             | Min             | Std")
        println("----------------|-----------------|-----------------|-----------------|---------------")
        for (key, values) in metrics[phase]
            println(@sprintf("%-15s | %-15.10f | %-15.10f | %-15.10f | %-17.10f", string(key), values[:mean]*100, values[:max]*100, values[:min]*100, values[:std]*100))
        end
    else
        println("Metric          | Value           |")
        println("----------------|-----------------|")
        for (key, value) in metrics[phase]
            if isa(value, Real)
                println(@sprintf("%-15s | %-15.10f |", string(key), value*100))
            end
        end
    end
end