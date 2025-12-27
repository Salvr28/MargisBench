#!/bin/bash


source ../hailo_virtualenv/bin/activate
echo "[PARSING]"
echo ""
for file in ../ONNXModels/*.onnx; do
    echo "PARSING ${file} ..."
    echo ""
    filename=$(echo $file | cut -d "." -f 3 | cut -d "/" -f 3)
    hailo parser onnx --hw-arch hailo8 --har-path ./har_files/${filename}.har $file
    rm *.log
done
echo "PARSING COMPLETED CORRECTLY! PASSING TO OPTIMIZATION..."
echo "[OPTIMIZATION]"
for file in ./har_files/*.har; do
    echo "QUANTIZING ${file} ..."
    echo ""
    filename=$(echo $file | cut -d "." -f 2 | cut -d "/" -f 3)
    hailo optimize --hw-arch hailo8 --output-har-path ./har_files_quantized/${filename}_quantized.har --use-random-calib-set --model-script "./avgpool1.alls" $file 
    rm *.log
done
echo "OPTIMIZATION COMPLETED CORRECTLY! PASSING TO COMPILE THE MODELS..."
echo "[COMPILE]"
for file in ./har_files_quantized/*.har; do
    echo "COMPILING ${file} ..."
    echo ""
    filename=$(echo $file | cut -d "." -f 2 | cut -d "/" -f 3)
    hailo compiler --hw-arch hailo8 --output-dir ./hef_files $file 
    rm *.log
    rm ./hef_files/*.log
done