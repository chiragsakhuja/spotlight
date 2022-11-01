#!/bin/bash

help() {
    echo "Usage: $0 MODE [OPTION]"
    echo ""
    echo "Options:"
    echo "  MODE          The type of data to compare. Either main-Edge,"
    echo "                main-cloud, general, or ablation."
    echo "  -h"
    echo "  --help        Print this message."
    exit
}

scale=""
compare_list=()
case $1 in
    main-edge)
        scale="Edge"
        compare_list=(Spotlight Eyeriss NVDLA MAERI)
        ;;
    main-cloud)
        scale="Cloud"
        compare_list=(Spotlight Eyeriss NVDLA MAERI)
        ;;
    general)
        scale="Edge"
        compare_list=(Spotlight Eyeriss NVDLA MAERI Spotlight-Multi Spotlight-General)
        ;;
    ablation)
        scale="Edge"
        compare_list=(Spotlight Spotlight-GA Spotlight-R Spotlight-V Spotlight-F)
        ;;
    *)
        echo "Invalid comparison: $1"
        help
        ;;
esac
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            help
            ;;
        *)
            break
            ;;
    esac
    shift
done


echo "Path,Min,Max,Median,Median Normalized to Spotlight"

for target in EDP Delay; do
    for model in VGG16 RESNET MOBILENET MNASNET TRANSFORMER; do
        for algorithm in "${compare_list[@]}"; do
            datafile="results/$scale/$algorithm/$target/$model/out.txt"
            if [[ -f $datafile ]]; then
                measurements=$(grep opt_hw $datafile | tr -s ' ' | cut -d' ' -f3 | sort -g)
                min=$(echo "$measurements" | head -n1)
                max=$(echo "$measurements" | tail -n1)
                median=$(echo "$measurements" | awk '
                {
                    count[NR] = $1;
                }
                END {
                    if (NR % 2 == 1) {
                        printf("%.3e\n", count[(NR + 1) / 2]);
                    } else {
                        printf("%.3e\n", (count[(NR / 2)] + count[(NR / 2) + 1]) / 2.0);
                    }
                }')

                if [[ $algorithm == "Spotlight" ]]; then
                    denominator=$median
                fi

                median_norm=$(echo $median | awk '{ printf("%.2f",$0 / '"$denominator)}")
                printf "$scale/$algorithm/$target/$model,$min,$max,$median,"
                if [[ ! -z $denominator ]]; then
                    echo "$median_norm"
                else
                    echo ""
                fi
            fi
        done
    done
done