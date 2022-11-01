#!/bin/bash

help() {
    echo "Usage: $0 MODE [OPTION]"
    echo ""
    echo "Modes:"
    echo "  full          Run the complete set of tests. Can take multiple days"
    echo "                to complete!"
    echo "  single        Run a single type of run."
    echo ""
    echo "Options:"
    echo "  -h"
    echo "  --help        Print this message."
    exit
}

help_full() {
    echo "Usage: $0 full [OPTION]"
    echo ""
    echo "Options:"
    echo "  -h"
    echo "  --help        Print this message."
    echo "  --trials      Number of trials to run each algorithm for."
    exit
}

help_single() {
    echo "Usage: $0 --model MODEL --target TARGET [OPTION]"
    echo ""
    echo "Options:"
    echo "  --algorithm   Search algorithm to run. Either Spotlight,"
    echo "                Spotlight-GA, Spotlight-R, Spotlight-V, or"
    echo "                Spotlight-F."
    echo "  -h"
    echo "  --help        Print this message."
    echo "  --model       DL model. Either VGG16, RESNET, MOBILENET, MNASNET,"
    echo "                or TRANSFORMER."
    echo "  --target      Optimization target. Either EDP or delay."
    echo "  --trials      Number of trials to run each algorithm for."
    exit
}

run_single() {
    algorithm=$1
    shift
    model=$1
    shift
    target=$1
    shift
    trials=$1
    shift

    extra_flags=""
    hw_trials=100
    sw_trials=100

    result_dir="results/$algorithm/$target/$model"
    mkdir -p $result_dir

    case $algorithm in
        Spotlight-GA)
            real_algorithm="ga_hw_sw_search"
            extra_flags="--hw-batch-size=20 --sw-batch-size=20 --scale-trials"
            ;;
        Spotlight-R)
            real_algorithm="random_hw_sw_search"
            extra_flags="--scale-trials"
            ;;
        Spotlight-V)
            real_algorithm="bo_hw_sw_search"
            extra_flags="--exclude-feat=original,intuitive,data-driven"
            ;;
        Spotlight-F)
            real_algorithm="bo_hw_sw_search"
            extra_flags="--dataflow=fixed"
            ;;
        Spotlight)
            real_algorithm="bo_hw_sw_search"
            ;;
        *)
            echo "Invalid algorithm: $algorithm"
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

    python src/main.py --hw-progress-bar --model=$real_algorithm --trials=$trials --layers=$model --target=$real_target --hw-trials=$hw_trials --sw-trials=$sw_trials $extra_flags $@ > $result_dir/out.txt
}

mode=""
if [[ $# -lt 1 ]]; then
    help
fi

mode=$1
shift

if [[ $mode == "single" ]]; then
    algorithm="Spotlight"
    model=""
    target=""
    trials=1

    while [[ $# -gt 0 ]]; do
        case $1 in
            --algorithm)
                algorithm=$2
                shift
                ;;
            -h|--help)
                help_single
                ;;
            --model)
                model=$2
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

    run_single $algorithm $model $target $trials $@

elif [[ $mode == "full" ]]; then
    trials=1

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                help_full
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

    for algorithm in Spotlight Spotlight-GA Spotlight-R Spotlight-V Spotlight-F; do
        for target in EDP Delay; do
            for model in VGG16 RESNET MOBILENET MNASNET TRANSFORMER; do
                echo "$algorithm-$target-$model"
                run_single $algorithm $model $target $trials $@
            done
        done
    done
else
    help
fi