#!/bin/bash

help() {
    echo "Usage: $0 MODE [OPTION]"
    echo ""
    echo "Modes:"
    echo "  ablation      Run other Spotlight variants on the complete suite of"
    echo "                benchmarks."
    echo "  general       Run Spotlight in generalized settings."
    echo "  main-edge     Run the complete set of benchmarks for edge-scale."
    echo "  main-cloud    Run the complete set of benchmarks for cloud-scale. Can"
    echo "                take multiple days to complete!"
    echo "  single        Run a single configuration."
    echo ""
    echo "Options:"
    echo "  -h"
    echo "  --help        Print this message."
    echo "  --slurm       Whether to submit jobs to SLURM or not. Must specify"
    echo "                --slurm-queue."
    echo "  --slurm-queue The SLURM queue to submit to. Required if using SLURM."
    echo "  --trials      Number of trials to run each algorithm for."
    exit
}

help_main() {
    echo "Usage: $0 $1 [OPTION]"
    echo ""
    echo "Options:"
    echo "  -h"
    echo "  --help        Print this message."
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
    hw_trials=100
    sw_trials=100
    optimizer_type="hw"
    runtime_limit="07:59:59"

    result_dir="results/$scale/$technique/$target/$model"
    mkdir -p $result_dir

    case $scale in
        Edge)
            real_scale=""
            ;;
        Cloud)
            real_scale="--space-template=datacenter --max-invalid=1000"
            runtime_limit="47:59:59"
            ;;
        *)
            echo "Invalid scale: $scale"
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


    case $technique in
        Spotlight-Multi)
            real_technique="bo_hw_sw_search"
            extra_flags="--hw-point=MoRVMnT_$real_target"
            optimizer_type="sw"
            ;;
        Spotlight-General)
            real_technique="bo_hw_sw_search"
            extra_flags="--hw-point=MoRV_$real_target"
            optimizer_type="sw"
            ;;
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
            runtime_limit="00:59:59"
            if [[ $scale == "Edge" ]]; then
                extra_flags="--hw-point=eyeriss_edge"
            else
                extra_flags="--hw-point=eyeriss_cloud"
            fi
            ;;
        NVDLA)
            real_technique="bo_sw_search"
            optimizer_type="sw"
            runtime_limit="00:59:59"
            if [[ $scale == "Edge" ]]; then
                extra_flags="--hw-point=nvdla_edge"
            else
                extra_flags="--hw-point=nvdla_cloud"
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

    real_progress_bar=""
    if [[ $progress_bar -eq 1 ]]; then
        real_progress_bar="--$optimizer_type-progress-bar"
    fi

    if [[ $slurm == 1 ]]; then
        key=$scale-$technique-$target-$model
        read -r -d '' template << EOM
#!/bin/bash
#SBATCH -J $key
#SBATCH -o logs/$key.o%j
#SBATCH -e logs/$key.e%j
#SBATCH -p $slurm_queue
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t $runtime_limit

python src/main.py $real_progress_bar --model=$real_technique --trials=$trials --layers=$model --target=$real_target --hw-trials=$hw_trials --sw-trials=$sw_trials $real_scale --output-dir=$result_dir --output-filename=out.txt --output-to-file $extra_flags $@
EOM

        echo "$template" > scripts/$key.sh
        sbatch scripts/$key.sh
    else
        python src/main.py $real_progress_bar --model=$real_technique --trials=$trials --layers=$model --target=$real_target --hw-trials=$hw_trials --sw-trials=$sw_trials $real_scale --output-dir=$result_dir --output-filename=out.txt --output-to-file $extra_flags $@ > /dev/null
    fi
}

mode=""
if [[ $# -lt 1 ]]; then
    help
fi

mode=$1
shift

trials=1
slurm=0
slurm_queue=""
help=0
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            help=1
            ;;
        --trials)
            trials=$2
            shift
            ;;
        --slurm)
            slurm=1
            ;;
        --slurm-queue)
            slurm_queue=$2
            shift
            ;;
        *)
            break
            ;;
    esac
    shift
done

if [[ $slurm == 1 ]]; then
    if [[ $slurm_queue == "" ]]; then
        help_single
    fi
    mkdir -p logs scripts
fi

if [[ $mode == "single" ]]; then
    if [[ $help -eq 1 ]]; then
        help_single
    fi

    technique=""
    model=""
    scale=""
    target=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --technique)
                technique=$2
                shift
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
    if [[ $help -eq 1 ]]; then
        help_main $mode
    fi

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
                if [[ $slurm == 1 ]]; then
                    wait
                fi
            done
        done
        wait
    done
elif [[ $mode == "ablation" ]]; then
    if [[ $help -eq 1 ]]; then
        help_main $mode
    fi

    for model in VGG16 RESNET MOBILENET MNASNET TRANSFORMER; do
        for target in EDP Delay; do
            for technique in Spotlight-GA Spotlight-R Spotlight-V Spotlight-F; do
                echo "$technique-$target-$model"
                run_single $technique $model $target $trials "Edge" 0 $@ &
                if [[ $slurm == 1 ]]; then
                    wait
                fi
            done
        done
        wait
    done
elif [[ $mode == "general" ]]; then
    if [[ $help -eq 1 ]]; then
        help_main $mode
    fi

    for target in EDP Delay; do
        for model in VGG16 RESNET MOBILENET MNASNET TRANSFORMER; do
            echo "Spotlight-Multi-$target-$model"
            run_single "Spotlight-Multi" $model $target $trials "Edge" 0 $@ &
            if [[ $slurm == 1 ]]; then
                wait
            fi
        done
        wait
    done

    for model in MNASNET TRANSFORMER; do
        for target in EDP Delay; do
            echo "Spotlight-General-$target-$model"
            run_single "Spotlight-General" $model $target $trials "Edge" 0 $@ &
            if [[ $slurm == 1 ]]; then
                wait
            fi
        done
    done
    wait
else
    help
fi
