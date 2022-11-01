#!/bin/bash

echo "Path,Min,Max,Median,Median Normalized to Spotlight"

for target in EDP Delay; do
    for model in VGG16 RESNET MOBILENET MNASNET TRANSFORMER; do
        for algorithm in Spotlight Spotlight-GA Spotlight-R Spotlight-V Spotlight-F; do
            datafile="results/$algorithm/$target/$model/out.txt"
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
                printf "$algorithm/$target/$model,$min,$max,$median,"
                if [[ ! -z $denominator ]]; then
                    echo "$median_norm"
                else
                    echo ""
                fi
            fi
        done
    done
done