function process_dataset(original_path::AbstractString, dataset::AbstractString;
        label_mapping=identity,
        header=false,
        missingstring=nothing,
        class_index=-1,
        variable_indices=1:0,
        ext=".csv",
    )
    # Sanity checks.
    if class_index < 1
        error("class_index should be a positive integer.")
    end
    if isempty(variable_indices) || any(<(0), variable_indices)
        error("variable_indices should contain positive integers.")
    end

    # Read data.
    input_df = CSV.read(original_path, DataFrame, header=header, missingstring=missingstring)

    # Build output DataFrame.
    output_df = DataFrame()
    output_df[!, :class] = map(label_mapping, input_df[:, class_index])    
    for (i, variable_index) in enumerate(variable_indices)
        column_name = Symbol("var", i)
        output_df[!, column_name] = input_df[!, variable_index]
    end

    # Drop rows with missing values.
    dropmissing!(output_df)

    # Write to disk.
    output_path = joinpath(dirname(original_path), dataset * ext)
    if ext == ".csv"
        CSV.write(output_path, output_df, delim=',', writeheader=true)
    elseif ext == ".csv.gz"
        open(GzipCompressorStream, output_path, "w") do stream
            CSV.write(stream, output_df, delim=",", writeheader=true)
        end
    else
        error("Unknown file extension option '$(ext)'")
    end

    # Delete the original file.
    rm(original_path)

    @info "Saved $(dataset) to $(output_path). Source dataset $(original_path) has been deleted."

    return output_path
end

"""
`list_datasets()`

List available datasets in SparseSVM.
"""
list_datasets() = DATASETS[]

"""
`dataset(str)`

Load a dataset named `str`, if available. Returns data as a `DataFrame` where
the first column contains labels/targets and the remaining columns correspond to
distinct features.
"""
function dataset(str)
    filename_of(file_path) = basename(file_path) |> Base.Fix2(split, '.') |> first
    matches_filename(file_path, filename) = filename_of(file_path) == filename

    # Locate dataset file.
    dataset_path = @datadep_str(DATADEPNAME)
    files = readdir(dataset_path, join=true)
    filter!(contains(".csv"), files)
    filter!(Base.Fix2(matches_filename, str), files)
    if isempty(files)
        error("File for dataset $(str) not found.")
    end
    dataset_file = first(files)

    # Read dataset file as a DataFrame.
    df = if splitext(dataset_file)[2] == ".csv"
        CSV.read(dataset_file, DataFrame)
    else # assume .csv.gz
        open(GzipDecompressorStream, dataset_file, "r") do stream
            CSV.read(stream, DataFrame)
        end
    end

    return df
end
