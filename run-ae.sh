#!/bin/bash

help() {
    echo "Usage: $0 MODE [OPTION]"
    echo ""
    echo "Modes:"
    echo "  ablation      Run other Spotlight variants on the complete suite of"
    echo "                benchmarks."
    echo "  main-edge     Run the complete set of benchmarks for edge-scale."
    echo "  main-cloud    Run the complete set of benchmarks for cloud-scale. Can"
    echo "                take multiple days to complete!"
    echo "  single        Run a single configuration."
    echo ""
    echo "Options:"
    echo "  -h"
    echo "  --help        Print this message."
    exit
}

help_main() {
    echo "Usage: $0 $1 [OPTION]"
    echo ""
    echo "Options:"
    echo "  -h"
    echo "  --help        Print this message."
    echo "  --trials      Number of trials to run each algorithm for."
    exit
}

help_single() {
    echo "Usage: $0 single --model MODEL --target TARGET --technique NAME --scale SCALE [OPTION]"
    echo ""
    echo "Options:"
    echo "  -h"
    echo "  --help        Print this message."
    echo "  --model       DL model. Either VGG16, RESNET, MOBILENET, MNASNET,"
    echo "                or TRANSFORMER."
    echo "  --scale       Scale of accelerator. Either Edge or Cloud."
    echo "  --technique   Either the search algorithm to run or the hand-designed"
    echo "                baseline. Either Spotlight, Spotlight-GA, Spotlight-R,"
    echo "                Spotlight-R, Spotlight-V, Spotlight-F, Eyeriss, NVDLA,"
    echo "                or MAERI."
    echo "  --target      Optimization target. Either EDP or delay."
    echo "  --trials      Number of trials to run each algorithm for."
    exit
}

run_single() {
    technique=$1
    shift
    model=$1
    shift
    target=$1
    shift
    trials=$1
    shift
    scale=$1
    shift
    progress_bar=$1
    shift

    extra_flags=""
    hw_trials=10
    sw_trials=10
    optimizer_type="hw"

    result_dir="results/$scale/$technique/$target/$model"
    mkdir -p $result_dir

    case $scale in
        Edge)
            real_scale=""
            ;;
        Cloud)
            real_scale="--space-template=datacenter --max-invalid=1000"
            ;;
        *)
            echo "Invalid scale: $scale"
            help_single
            ;;
    esac

    case $technique in
        Spotlight-GA)
            real_technique="ga_hw_sw_search"
            extra_flags="--hw-batch-size=20 --sw-batch-size=20 --scale-trials"
            ;;
        Spotlight-R)
            real_technique="random_hw_sw_search"
            extra_flags="--scale-trials"
            ;;
        Spotlight-V)
            real_technique="bo_hw_sw_search"
            extra_flags="--exclude-feat=original,intuitive,data-driven"
            ;;
        Spotlight-F)
            real_technique="bo_hw_sw_search"
            extra_flags="--dataflow=fixed"
            ;;
        Spotlight)
            real_technique="bo_hw_sw_search"
            ;;
        Eyeriss)
            real_technique="bo_sw_search"
            optimizer_type="sw"
            if [[ $scale == "Edge" ]]; then
                extra_flags="--hw-point={'num_simd_lane':1,'bit_width':16,'bandwidth':256,'l0_buf_size':16384,'l1_buf_size':262144,'subclusters':[15,18]}"
            else
                extra_flags="--hw-point={'num_simd_lane':1,'bit_width':16,'bandwidth':256,'l0_buf_size':16777216,'l1_buf_size':268435456,'subclusters':[60,72]}"
            fi
            ;;
        NVDLA)
            real_technique="bo_sw_search"
            optimizer_type="sw"
            if [[ $scale == "Edge" ]]; then
                extra_flags="--hw-point={'num_simd_lane':1,'bit_width':8,'bandwidth':256,'l0_buf_size':32768,'l1_buf_size':524288,'subclusters':[16,16]}"
            else
                extra_flags="--hw-point={'num_simd_lane':1,'bit_width':8,'bandwidth':256,'l0_buf_size':1048576,'l1_buf_size':16777216,'subclusters':[120,120]}"
            fi
            ;;
        MAERI)
            real_technique="bo_hw_sw_search"
            if [[ $scale == "Edge" ]]; then
                extra_flags="--simd-low=1 --simd-high=1 --prec-low=8 --prec-high=8 --bw-low=64 --bw-high=64 --l1-low=65 --l1-high=65 --l2-low=80 --l2-high=80 --pe-low=374 --pe-high=374"
            else
                extra_flags="--simd-low=1 --simd-high=1 --prec-low=8 --prec-high=8 --bw-low=64 --bw-high=64 --l1-low=1024 --l1-high=1024 --l2-low=2048 --l2-high=2048 --pe-low=14336 --pe-high=14336"
            fi
            ;;
        *)
            echo "Invalid technique: $technique"
            help_single
            ;;
    esac

    case $model in
        VGG16|RESNET|MOBILENET|MNASNET|TRANSFORMER)
            ;;
        *)
            echo "Invalid model: $model"
            help_single
            ;;
    esac

    case $target in
        EDP)
            real_target="edp"
            ;;
        Delay)
            real_target="delay"
            ;;
        *)
            echo "Invalid target: $target"
            help_single
            ;;
    esac

    real_progress_bar=""
    if [[ $progress_bar -eq 1 ]]; then
        real_progress_bar="--$optimizer_type-progress-bar"
    fi
    python src/main.py $real_progress_bar --model=$real_technique --trials=$trials --layers=$model --target=$real_target --hw-trials=$hw_trials --sw-trials=$sw_trials $real_scale $extra_flags $@ > $result_dir/out.txt
}

mode=""
if [[ $# -lt 1 ]]; then
    help
fi

mode=$1
shift

if [[ $mode == "single" ]]; then
    technique=""
    model=""
    scale=""
    target=""
    trials=1

    while [[ $# -gt 0 ]]; do
        case $1 in
            --technique)
                technique=$2
                shift
                ;;
            -h|--help)
                help_single
                ;;
            --model)
                model=$2
                shift
                ;;
            --scale)
                scale=$2
                shift
                ;;
            --target)
                target=$2
                shift
                ;;
            --trials)
                trials=$2
                shift
                ;;
            *)
                break
                ;;
        esac
        shift
    done

    run_single $technique $model $target $trials $scale 1 $@

elif [[ $mode == "main-"* ]]; then
    trials=1

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                help_main $mode
                ;;
            --trials)
                trials=$2
                shift
                ;;
            *)
                break
                ;;
        esac
        shift
    done

    scale=""
    if [[ $mode == *"-edge" ]]; then
        scale="Edge"
    elif [[ $mode == *"-cloud" ]]; then
        scale="Cloud"
    fi

    for model in VGG16 RESNET MOBILENET MNASNET TRANSFORMER; do
        for target in EDP Delay; do
            for technique in Spotlight Eyeriss NVDLA MAERI; do
                echo "$technique-$target-$model"
                run_single $technique $model $target $trials $scale 0 $@ &
            done
        done
        wait
    done
elif [[ $mode == "ablation" ]]; then
    trials=1

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                help_main $mode
                ;;
            --trials)
                trials=$2
                shift
                ;;
            *)
                break
                ;;
        esac
        shift
    done

    for model in VGG16 RESNET MOBILENET MNASNET TRANSFORMER; do
        for target in EDP Delay; do
            for technique in Spotlight-GA Spotlight-R Spotlight-V Spotlight-F; do
                echo "$technique-$target-$model"
                run_single $technique $model $target $trials "Edge" 0 $@ &
            done
        done
        wait
    done
else
    help
fi