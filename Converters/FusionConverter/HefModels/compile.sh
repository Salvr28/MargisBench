#!/bin/bash


source ../hailo_virtualenv/bin/activate
echo "[PARSING]"
echo ""
for file in ../ONNXModels/*.onnx; do
    echo ""
    echo "PARSING ${file} ..."
    echo ""
    filename=$(echo $file | cut -d "." -f 3 | cut -d "/" -f 3)
    hailo parser onnx --hw-arch hailo8 --har-path ./har_files/${filename}.har $file
    rm *.log
done

echo "PARSING COMPLETED CORRECTLY! PASSING TO OPTIMIZATION..."
echo "[OPTIMIZATION]"
for file in ./har_files/*.har; do
    echo ""    
    echo "QUANTIZING ${file} ..."
    echo ""
    filename=$(echo $file | cut -d "." -f 2 | cut -d "/" -f 3)

    #This regex help to extract the name of the model to construct the .npy paths.
    #The first part prints the filename and the lines from the folder CalibrationArrays
    #The second command performs a longest prefix match between the first string and each line took by the ls command, removes "pruned" or "distilled" and removes the hypotethically final _, prints the lines
    
    calib_set_filename="$(printf "%s\n%s\n" $filename "$(ls ../CalibrationArrays/)" | sed -n '1h; 1!{G; s/^\(.*\).*\n\1.*$/\1/; s/_[a-zA-Z]*$//; s/_$//; p}' | sort | tail -n 1)_calibration_data.npy"
    echo ""
    echo "CALIBRATION SET FILENAME: ${calib_set_filename}"
    hailo optimize --hw-arch hailo8 --output-har-path "./har_files_quantized/${filename}_quantized.har" --calib-set-path "../CalibrationArrays/${calib_set_filename}" --model-script "./avgpool1.alls" $file 
    rm *.log
done

echo "OPTIMIZATION COMPLETED CORRECTLY! PASSING TO COMPILE THE MODELS..."
echo "[COMPILE]"
for file in ./har_files_quantized/*.har; do
    echo ""    
    echo "COMPILING ${file} ..."
    echo ""
    filename=$(echo $file | cut -d "." -f 2 | cut -d "/" -f 3)
    hailo compiler --hw-arch hailo8 --output-dir ./hef_files $file 
    rm *.log
    rm ./hef_files/*.log
    rm ./hef_files/*.har
done